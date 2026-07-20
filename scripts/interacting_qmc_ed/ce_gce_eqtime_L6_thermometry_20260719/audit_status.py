#!/usr/bin/env python3.11
"""Read-only status audit for all L=6 workflow stages on CADES."""

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
MANIFESTS = HERE / "manifests"
TABLES = [
    "equal_time_observables_qmc.tsv", "equal_time_charge_spin_wedge_qmc.tsv",
    "equal_time_structure_factors_qmc.tsv", "equal_time_neighbor_shells_qmc.tsv",
]


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def status_values(root: Path) -> list[int]:
    values=[]
    for p in root.glob("ranks/rank_*/checkpoint.jls.status"):
        d={}
        for line in p.read_text(errors="replace").splitlines():
            if "=" in line:
                k,v=line.split("=",1); d[k]=v
        try: values.append(int(d.get("nsamples", d.get("completed_batches", "-1"))))
        except ValueError: pass
    return values


def ce_audit(path: Path) -> dict[str, object]:
    rows=read(path); finals=0; rank_full=0; checkpoint_full=0; status_full=0; progress=[]
    for r in rows:
        root=Path(r["outdir"]); expected=int(r["expected_ranks"])
        rank=len(list(root.glob("ranks/rank_*/checkpoint_complete.txt")))
        ck=len(list(root.glob("ranks/rank_*/checkpoint.jls")))
        st=len(list(root.glob("ranks/rank_*/checkpoint.jls.status")))
        strict=(root/"checkpoint_complete.txt").is_file() and rank==expected and all((root/f).is_file() for f in TABLES)
        finals += strict; rank_full += rank==expected; checkpoint_full += ck==expected; status_full += st==expected
        if not strict:
            vals=status_values(root)
            if vals: progress.append(statistics.median(vals))
    return {
        "manifest": path.name, "rows": len(rows), "final": finals, "rank_full": rank_full,
        "checkpoint_full": checkpoint_full, "status_full": status_full,
        "incomplete_median_measurements": statistics.median(progress) if progress else math.nan,
        "target_measurements": int(rows[0]["max_batches"]) * int(rows[0]["batch_nsamples"]),
    }


def probe_run_base(r: dict[str, str]) -> str:
    prefix="attractive_hubbard_rect" if float(r["U"]) < 0 else "hubbard_spin_hs_rect"
    return f"{prefix}_U{float(r['U']):.2f}_tp0.00_mu{float(r['mu_probe']):.2f}_Lx{int(r['Lx'])}_Ly{int(r['Ly'])}_b{float(r['beta']):.2f}-{int(r['sid'])}"


def probe_audit(paths: list[Path]) -> dict[str, object]:
    rows=[]
    for p in paths: rows.extend(read(p))
    seen=set(); unique=[]
    for r in rows:
        if r["out_parent"] in seen: continue
        seen.add(r["out_parent"]); unique.append(r)
    complete=full_ck=0; density_errors=[]
    for r in unique:
        parent=Path(r["out_parent"]); base=probe_run_base(r)
        c=parent/f"complete_{base}"; i=parent/base
        gf=c/"global_stats.csv"
        if gf.is_file():
            complete += 1
            try:
                with gf.open() as f:
                    stats={x["MEASUREMENT"]:(float(x["MEAN_REAL"]),float(x.get("STD") or 0)) for x in csv.DictReader(f,delimiter=" ",skipinitialspace=True) if x.get("MEASUREMENT")}
                density_errors.append(abs(36*stats["density"][0]-int(r["Ntot_target"])))
            except Exception: pass
        full_ck += len(list(i.glob("checkpoint_pID-*.jld2"))) == 32
    return {
        "manifests": len(paths), "rows": len(unique), "complete": complete,
        "full_checkpoint_sets": full_ck,
        "median_abs_N_error_complete": statistics.median(density_errors) if density_errors else math.nan,
    }


def prod_run_base(r: dict[str, str]) -> str:
    prefix="attractive_hubbard_rect" if float(r["U"]) < 0 else "hubbard_spin_hs_rect"
    return f"{prefix}_U{float(r['U']):.2f}_tp0.00_mu{float(r['mu_final']):.2f}_Lx{int(r['Lx'])}_Ly{int(r['Ly'])}_b{float(r['beta']):.2f}-{int(r['sid'])}"


def production_audit(paths: list[Path]) -> dict[str, object]:
    rows=[]
    for p in paths: rows.extend(read(p))
    final=rank_full=tables_full=density_ok=0
    for r in rows:
        parent=Path(r["out_parent"]); complete=parent/f"complete_{prod_run_base(r)}"; export=parent/"export"
        ranks=len(list(complete.glob("simulation_info_sID-*_pID-*.toml")))
        tables=all((export/f).is_file() for f in TABLES)
        rank_full += ranks==int(r["expected_ranks"]); tables_full += tables
        ok=False
        achieved=parent/"achieved_density.tsv"
        if achieved.is_file():
            q=read(achieved)[0]; ok=abs(float(q["delta_N"])) <= float(q["tolerance"])
        density_ok += ok
        final += (parent/"dqmc_gce_eqtime_complete.txt").is_file() and ranks==int(r["expected_ranks"]) and tables and ok
    return {"manifests":len(paths),"rows":len(rows),"final":final,"rank_full":rank_full,"tables_full":tables_full,"density_ok":density_ok}


def queue() -> Counter:
    cmd=["squeue","-r","-u",subprocess.check_output(["id","-un"],text=True).strip(),"-h","-o","%T|%j|%a|%P|%q|%D|%C"]
    out=subprocess.check_output(cmd,text=True)
    c=Counter()
    for line in out.splitlines():
        state,name,account,part,qos,nodes,cpus=line.split("|")
        if name.startswith(("ceL6","muL6","gceL6")):
            c[(state,name,account,part,qos,int(nodes),int(cpus))]+=1
    return c


def error_scan() -> Counter:
    patterns={
        "numerical_zero_sign":re.compile(r"numerical[- ]zero[- ]sign",re.I),
        "jld2":re.compile(r"JLD2|safe JLD2 checkpoint write failed",re.I),
        "mpi":re.compile(r"MPI_ERRORS_ARE_FATAL|Socket closed|failed to TCP connect|No route to host",re.I),
        "drift":re.compile(r"drift too large",re.I),
        "wrong_rank":re.compile(r"wrong[- ]rank",re.I),
        "load_error":re.compile(r"LoadError|ERROR:\s+LoadError",re.I),
        "out_of_memory":re.compile(r"OutOfMemory|out of memory|oom-kill",re.I),
        "fatal":re.compile(r"\bfatal\b",re.I),
    }
    counts=Counter()
    logs=Path("/home/9pm/nUHubbard/logs")
    current_prefixes = (
        "ceL6Attr_", "ceL6A20_", "ceL6Pos_", "ceL6Pilot_",
        "ceL6PosProd_", "ceL6PosHi_", "muL6Attr_", "muL6Spin_",
        "gceL6Attr_", "gceL6Spin_",
    )
    now = time.time()
    for p in logs.iterdir():
        if not p.is_file() or not p.name.endswith(".err") or not p.name.startswith(current_prefixes):
            continue
        # The audit reports current/recent failures; historical legs are
        # retained on disk and in Slurm accounting but should not be rescanned
        # every hour (some warning-heavy files are hundreds of MB).
        try:
            if now - p.stat().st_mtime > 3 * 3600:
                continue
        except OSError:
            continue
        # MPI rank chatter can make a single stderr file hundreds of MB.  A
        # full read made the hourly audit CPU- and I/O-heavy.  Error banners
        # occur at process startup or near termination, so inspect bounded
        # head/tail windows instead.  This is monitoring only; strict-final
        # classification never depends on this scan.
        try:
            with p.open("rb") as f:
                head=f.read(64_000)
                size=f.seek(0, 2)
                f.seek(max(len(head), size-256_000))
                tail=f.read()
            text=(head+b"\n"+tail).decode(errors="replace")
        except OSError:
            continue
        for key,pat in patterns.items(): counts[key]+=len(pat.findall(text))
    return counts


def live_log_progress(job_name: str) -> dict[str, object]:
    """Approximate current per-task progress from the latest rank chatter.

    Status files are checkpoint snapshots and can lag live jobs by an hour.
    The tail estimate is clearly labeled live/approximate and is used only for
    monitoring, never for strict-final classification.
    """
    logs=Path("/home/9pm/nUHubbard/logs")
    now=time.time(); task_medians=[]; cycles=[]
    for path in logs.glob(f"{job_name}_*.out"):
        try:
            if now-path.stat().st_mtime > 3*3600: continue
            with path.open("rb") as f:
                size=f.seek(0,2); f.seek(max(0,size-2_000_000)); tail=f.read().decode(errors="replace")
                f.seek(0); head=f.read(min(size,200_000)).decode(errors="replace")
            vals=[int(x) for x in re.findall(r"(?:batch=\d+\s+)?nsamples=(\d+)",tail)]
            if vals: task_medians.append(statistics.median(vals[-64:]))
            cycles.extend(int(x) for x in re.findall(r"continuation=(\d+)/\d+",head))
        except OSError:
            pass
    return {
        "job":job_name,"task_logs":len(task_medians),
        "live_median":statistics.median(task_medians) if task_medians else math.nan,
        "live_min":min(task_medians) if task_medians else math.nan,
        "live_max":max(task_medians) if task_medians else math.nan,
        "max_continuation":max(cycles) if cycles else 0,
    }


def main() -> None:
    ce_paths=[
        MANIFESTS/"ce_L6_attractive_beta_le10_r32_m50000.tsv",
        MANIFESTS/"ce_L6_attractive_beta20_r64_m30000.tsv",
        MANIFESTS/"ce_L6_positive_beta_le4_r32_m50000.tsv",
        MANIFESTS/"ce_L6_positive_pilot_beta5_6p7_10_r32_m10000.tsv",
    ] + sorted(MANIFESTS.glob("ce_L6_positive_admitted_r*_m*.tsv"))
    for p in ce_paths:
        x=ce_audit(p)
        print("CE {manifest}: final={final}/{rows} rank_full={rank_full}/{rows} checkpoint_full={checkpoint_full}/{rows} status_full={status_full}/{rows} incomplete_median={incomplete_median_measurements}/{target_measurements}".format(**x))
    probes=sorted(MANIFESTS.glob("gce_mu_probe_L6_*.tsv"))
    x=probe_audit(probes)
    print("GCE_PROBES manifests={manifests} complete={complete}/{rows} full_checkpoint_sets={full_checkpoint_sets}/{rows} median_abs_N_error_complete={median_abs_N_error_complete}".format(**x))
    prod=sorted(MANIFESTS.glob("gce_prod_L6_*_confirmed.tsv"))
    if prod:
        x=production_audit(prod)
        print("GCE_PRODUCTION manifests={manifests} final={final}/{rows} rank_full={rank_full}/{rows} tables_full={tables_full}/{rows} density_ok={density_ok}/{rows}".format(**x))
    for k,n in sorted(queue().items()):
        print(f"QUEUE count={n} state={k[0]} name={k[1]} account={k[2]} partition={k[3]} qos={k[4]} nodes={k[5]} cpus={k[6]}")
    for name in ("ceL6Attr","ceL6A20","ceL6Pos","ceL6Pilot","ceL6PosProd","ceL6PosHi"):
        x=live_log_progress(name)
        if x["task_logs"]:
            print("LIVE job={job} task_logs={task_logs} approximate_measurements_median={live_median} range={live_min}-{live_max} max_continuation={max_continuation}".format(**x))
    print("ERRORS", dict(error_scan()))


if __name__ == "__main__":
    main()
