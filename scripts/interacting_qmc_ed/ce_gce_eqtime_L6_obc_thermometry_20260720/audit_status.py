#!/usr/bin/env python3
"""Read-only hourly audit of the complete L=6 OBC thermometry workflow."""

from __future__ import annotations

import csv
import math
import re
import statistics
import subprocess
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
M = HERE / "manifests"
S = HERE / "status_source"
LOGS = Path("/home/9pm/nUHubbard_obc_dev/logs")
PRIMARY = (
    "equal_time_kinetic_per_site_qmc.tsv", "equal_time_double_occupancy_per_site_qmc.tsv",
    "equal_time_nn_spin_qmc.tsv", "equal_time_nn_connected_charge_qmc.tsv",
)
NAMES = ("ceL6OAtr","ceL6OA20","ceL6OPos","ceL6OPil","ceL6OPN","ceL6OPH","muL6OAttr","muL6OSpin","gceL6OAttr","gceL6OSpin")


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle: return list(csv.DictReader(handle, delimiter="\t"))


def median(values: list[float]) -> float:
    return statistics.median(values) if values else math.nan


def status_samples(root: Path) -> list[int]:
    values=[]
    for path in root.glob("ranks/rank_*/checkpoint.jls.status"):
        try:
            data=dict(line.split("=",1) for line in path.read_text(errors="replace").splitlines() if "=" in line)
            values.append(int(data.get("nsamples",data.get("completed_batches","-1"))))
        except (OSError,ValueError): pass
    return [value for value in values if value >= 0]


def ce_manifest(path: Path) -> dict[str, object]:
    rows=read(path); finals=rank_full=site_full=checkpoint_full=status_full=tables_full=0; progress=[]
    for row in rows:
        root=Path(row["outdir"]); expected=int(row["expected_ranks"])
        ranks=len(list(root.glob("ranks/rank_*/checkpoint_complete.txt")))
        sites=len(list(root.glob("ranks/rank_*/equal_time_site_density_qmc.tsv")))
        checkpoints=len(list(root.glob("ranks/rank_*/checkpoint.jls")))
        statuses=len(list(root.glob("ranks/rank_*/checkpoint.jls.status")))
        tables=sum((root/name).is_file() for name in PRIMARY)
        strict=(root/"obc_thermometry_complete.txt").is_file() and ranks==expected and sites==expected and tables==4
        finals+=strict; rank_full+=ranks==expected; site_full+=sites==expected; checkpoint_full+=checkpoints==expected; status_full+=statuses==expected; tables_full+=tables==4
        if not strict: progress.extend(status_samples(root))
    target=int(rows[0]["max_batches"])*int(rows[0]["batch_nsamples"]) if rows else 0
    return {"name":path.name,"rows":len(rows),"final":finals,"rank":rank_full,"site":site_full,"checkpoint":checkpoint_full,"status":status_full,"tables":tables_full,"median":median(progress),"target":target}


def base(row: dict[str,str], mu_field: str) -> str:
    prefix="attractive_hubbard_obc_rect" if row["family"]=="attractive" else "hubbard_spin_hs_obc_rect"
    return f"{prefix}_U{float(row['U']):.2f}_tp0.00_mu{float(row[mu_field]):.2f}_Lx{int(row['Lx'])}_Ly{int(row['Ly'])}_b{float(row['beta']):.2f}-{int(row['sid'])}"


def unique_rows(paths: list[Path], root_field: str) -> list[dict[str,str]]:
    result={}
    for path in paths:
        if not path.is_file(): continue
        for row in read(path): result.setdefault(row[root_field],row)
    return list(result.values())


def probes(paths: list[Path]) -> dict[str,object]:
    rows=unique_rows(paths,"out_parent"); complete=ckfull=rankfull=0; nerrors=[]
    for row in rows:
        parent=Path(row["out_parent"]); run=parent/base(row,"mu_probe"); done=parent/f"complete_{base(row,'mu_probe')}"
        checkpoint_paths=list(run.glob("checkpoint_pID-*.jld2"))+list(done.glob("checkpoint_pID-*.jld2"))
        ckfull+=len({path.name for path in checkpoint_paths})==32
        ranks=len(list(done.glob("simulation_info_sID-*_pID-*.toml"))); rankfull+=ranks==32
        achieved=parent/"probe_achieved_density.tsv"
        good=(done/"global_stats.csv").is_file() and ranks==32 and achieved.is_file(); complete+=good
        if good:
            try: nerrors.append(abs(float(read(achieved)[0]["delta_N"])))
            except (IndexError,KeyError,ValueError): pass
    return {"rows":len(rows),"complete":complete,"checkpoint":ckfull,"rank":rankfull,"median_Nerr":median(nerrors)}


def production(paths: list[Path]) -> dict[str,object]:
    rows=unique_rows(paths,"out_parent"); finals=checkpointfull=rankfull=sitefull=tablesfull=densityok=failed=bracketok=superseded=0
    for row in rows:
        bracket_valid=False
        try:
            lo,hi=sorted((float(row["mu_bracket_low"]),float(row["mu_bracket_high"])))
            fitted=float(row["mu_fitted"]); final_mu=float(row["mu_final"])
            bracket_valid=(
                hi-lo <= 0.020000000001
                and lo-1e-12 <= fitted <= hi+1e-12
                and lo-1e-12 <= final_mu <= hi+1e-12
            )
        except (KeyError,ValueError):
            bracket_valid=False
        bracketok+=bracket_valid;superseded+=not bracket_valid
        parent=Path(row["out_parent"]); done=parent/f"complete_{base(row,'mu_final')}"; expected=int(row["expected_ranks"])
        run=parent/base(row,"mu_final")
        checkpoints=list(run.glob("checkpoint_pID-*.jld2"))+list(done.glob("checkpoint_pID-*.jld2"))
        checkpointfull+=len({path.name for path in checkpoints})==expected
        ranks=len(list(done.glob("obc_equal_time_rank_pID-*.tsv"))); sites=len(list(done.glob("obc_equal_time_site_rank_pID-*.tsv")))
        tables=sum((done/name).is_file() for name in PRIMARY); ok=False
        achieved=parent/"achieved_density.tsv"
        if achieved.is_file():
            try: ok=abs(float(read(achieved)[0]["delta_N"]))<=float(row["density_tolerance"])
            except (IndexError,KeyError,ValueError): pass
        rankfull+=ranks==expected; sitefull+=sites==expected; tablesfull+=tables==4; densityok+=ok; failed+=(parent/"density_tolerance_failed.txt").is_file()
        finals+=(parent/"dqmc_gce_obc_eqtime_complete.txt").is_file() and ranks==expected and sites==expected and tables==4 and ok and bracket_valid
    return {"rows":len(rows),"final":finals,"checkpoint":checkpointfull,"rank":rankfull,"site":sitefull,"tables":tablesfull,"density":densityok,"density_failed":failed,"bracket_ok":bracketok,"superseded":superseded}


def queue() -> Counter:
    user=subprocess.check_output(["id","-un"],text=True).strip()
    output=subprocess.check_output(["squeue","-r","-h","-u",user,"-o","%T|%j|%a|%P|%q|%D|%C"],text=True)
    counts=Counter()
    for line in output.splitlines():
        state,name,account,partition,qos,nodes,cpus=line.split("|")
        if name in NAMES or name.startswith(("ceL6O","muL6O","gceL6O")):
            counts[(state,name,account,partition,qos,int(nodes),int(cpus))]+=1
    return counts


def live_progress(name: str) -> tuple[float,float,float,int,int]:
    values=[]; cycles=[]; files=0; now=time.time()
    for path in LOGS.glob(f"{name}_*.out"):
        try:
            if now-path.stat().st_mtime>3*3600: continue
            files+=1
            with path.open("rb") as handle:
                size=handle.seek(0,2); handle.seek(max(0,size-2_000_000)); tail=handle.read().decode(errors="replace")
                handle.seek(0); head=handle.read(min(size,200_000)).decode(errors="replace")
            samples=[int(x) for x in re.findall(r"(?:nsamples|measurement(?:s)?)[=: ]+(\d+)",tail,re.I)]
            if samples: values.append(statistics.median(samples[-64:]))
            cycles.extend(int(x) for x in re.findall(r"continuation=(\d+)/\d+",head))
        except OSError: pass
    return median(values),min(values) if values else math.nan,max(values) if values else math.nan,max(cycles,default=0),files


def errors() -> Counter:
    patterns={
        "numerical_zero_phase":re.compile(r"numerical-zero (?:phase|sign)",re.I),
        "jld2":re.compile(r"JLD2|safe JLD2 checkpoint write failed",re.I),
        "mpi":re.compile(r"MPI_ERRORS_ARE_FATAL|Socket closed|failed to TCP connect|No route to host",re.I),
        "drift":re.compile(r"drift too large",re.I), "wrong_rank":re.compile(r"wrong[- ]rank",re.I),
        "forbidden":re.compile(r"forbidden OBC translational|momentum outputs",re.I),
        "fatal":re.compile(r"\bfatal\b|LoadError|OutOfMemory|oom-kill",re.I),
    }
    result=Counter(); now=time.time()
    if not LOGS.is_dir(): return result
    for path in LOGS.glob("*.err"):
        if not path.name.startswith(tuple(name+"_" for name in NAMES)): continue
        try:
            if now-path.stat().st_mtime>3*3600: continue
            with path.open("rb") as handle:
                head=handle.read(64_000); size=handle.seek(0,2); handle.seek(max(len(head),size-256_000)); tail=handle.read()
            text=(head+b"\n"+tail).decode(errors="replace")
        except OSError: continue
        for key,pattern in patterns.items(): result[key]+=len(pattern.findall(text))
    return result


def main() -> None:
    exact=S/"exact_u0_L6_obc_snapshot.tsv"
    exact_rows=len(read(exact)) if exact.is_file() else 0
    refs=M/"L6_PBC_mu_reference_for_OBC.tsv"; ref_rows=len(read(refs)) if refs.is_file() else 0
    print(f"EXACT_U0 ensemble_rows={exact_rows}/80 physical_conditions={exact_rows//2}/40 geometry=36/60/50")
    print(f"PBC_MU_REFERENCES imported={ref_rows}/152 waiting={152-ref_rows}")
    ce_paths=[
        M/"ce_L6_obc_attractive_beta_le10_r32_m50000.tsv", M/"ce_L6_obc_attractive_beta20_r64_m30000.tsv",
        M/"ce_L6_obc_positive_beta_le4_r32_m50000.tsv", M/"ce_L6_obc_positive_pilot_beta5_6p7_10_r32_m10000.tsv",
    ]+sorted(M.glob("ce_L6_obc_positive_admitted_r*_m*.tsv"))
    for path in ce_paths:
        if not path.is_file(): continue
        x=ce_manifest(path)
        print("CE {name} final={final}/{rows} rank={rank}/{rows} site={site}/{rows} tables={tables}/{rows} checkpoint={checkpoint}/{rows} status={status}/{rows} incomplete_checkpoint_median={median}/{target}".format(**x))
    pilot=S/"positive_pilot_status.tsv"
    if pilot.is_file(): print("PILOT_TIERS",dict(Counter(row["tier"] for row in read(pilot))))
    x=probes(sorted(M.glob("gce_mu_probe_L6_obc_*.tsv")))
    print("GCE_PROBES complete={complete}/{rows} checkpoint_full={checkpoint}/{rows} rank_full={rank}/{rows} median_abs_N_error={median_Nerr}".format(**x))
    status=S/"mu_tuning"/"mu_target_status.tsv"
    if status.is_file(): print("MU_TARGETS",dict(Counter(row["status"] for row in read(status))))
    prod_paths=sorted(M.glob("gce_prod_L6_obc_*_confirmed.tsv"))
    prod_paths+=sorted((S/"submission_manifests").glob("gce_prod_L6_obc_*_delta_*.tsv"))
    prod_paths+=sorted((S/"repair_manifests").glob("gceL6O*_repair_*.tsv"))
    x=production(prod_paths)
    print("GCE_PRODUCTION final={final}/{rows} checkpoint_full={checkpoint}/{rows} rank={rank}/{rows} site={site}/{rows} tables={tables}/{rows} density_ok={density}/{rows} density_failed={density_failed} bracket_ok={bracket_ok}/{rows} superseded_bad_bracket={superseded}".format(**x))
    for key,count in sorted(queue().items()):
        print(f"QUEUE count={count} state={key[0]} name={key[1]} account={key[2]} partition={key[3]} qos={key[4]} nodes={key[5]} cpus={key[6]}")
    for name in NAMES:
        med,lo,hi,cycle,files=live_progress(name)
        if files: print(f"LIVE name={name} recent_logs={files} approximate_progress_median={med} range={lo}-{hi} max_continuation={cycle}")
    repair=S/"repair_ledger.tsv"
    print(f"REPAIRS total={len(read(repair)) if repair.is_file() else 0}")
    print("ERRORS",dict(errors()))


if __name__=="__main__": main()
