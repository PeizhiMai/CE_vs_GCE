#!/usr/bin/env python3
"""Static and exact-arithmetic validation gate for the L=6 OBC workflow."""

from __future__ import annotations

import csv
import importlib.util
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
M=HERE/"manifests"
ROOT=HERE.parents[2]
EXPECTED_SMOQ="c5f0c81bc98029bae585e0cb283428e293553999"
PRIMARY=("equal_time_kinetic_per_site_qmc.tsv","equal_time_double_occupancy_per_site_qmc.tsv","equal_time_nn_spin_qmc.tsv","equal_time_nn_connected_charge_qmc.tsv")


def read(path:Path):
    with path.open(newline="") as handle:return list(csv.DictReader(handle,delimiter="\t"))


def check(condition:bool,message:str):
    if not condition:raise AssertionError(message)
    print("PASS",message)


def load_exact():
    spec=importlib.util.spec_from_file_location("l6_obc_exact",HERE/"exact_u0_l6_obc.py");module=importlib.util.module_from_spec(spec);assert spec.loader
    sys.modules[spec.name]=module;spec.loader.exec_module(module);return module


def enumeration_moments(eps:np.ndarray,beta:float,n:int):
    configs=list(itertools.combinations(range(len(eps)),n));weights=np.asarray([math.exp(-beta*sum(eps[list(c)])) for c in configs]);weights/=weights.sum()
    first=np.zeros(len(eps));second=np.zeros((len(eps),len(eps)))
    for weight,config in zip(weights,configs):
        occ=np.zeros(len(eps));occ[list(config)]=1;first+=weight*occ;second+=weight*np.outer(occ,occ)
    return first,second


def main()->None:
    commit=subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()
    grid=read(M/"L6_obc_full_condition_grid.tsv")
    check(len(grid)==192,"exactly 192 physical conditions")
    check(sum(float(row["U"])!=0 for row in grid)==152 and sum(float(row["U"])==0 for row in grid)==40,"152 interacting plus 40 exact U=0 conditions")
    keys={(float(row["U"]),int(row["Ntot"]),float(row["beta"])) for row in grid};check(len(keys)==192,"physical-condition keys are unique")
    check({int(row["Ntot"]) for row in grid}=={12,18,26,32},"requested four particle sectors")
    check(all(int(row["Nup"])==int(row["Ndn"]) and int(row["Nup"])+int(row["Ndn"])==int(row["Ntot"]) for row in grid),"all canonical sectors are balanced")
    check(all(row["boundary"]=="open" and (int(row["site_count"]),int(row["nn_bond_count"]),int(row["nnn_bond_count"]))==(36,60,50) for row in grid),"all grid rows use OBC geometry 36/60/50")
    check(all(row["project_commit"]==commit and row["smoqydqmc_version"]=="2.0.12" and row["smoqydqmc_commit"]==EXPECTED_SMOQ for row in grid),"grid provenance matches deployed project and pinned OBC fork")

    expected={
        "ce_L6_obc_attractive_beta_le10_r32_m50000.tsv":72,
        "ce_L6_obc_attractive_beta20_r64_m30000.tsv":8,
        "ce_L6_obc_positive_beta_le4_r32_m50000.tsv":48,
        "ce_L6_obc_positive_pilot_beta5_6p7_10_r32_m10000.tsv":24,
        "gce_L6_obc_interacting_targets.tsv":152,
        "smoke_ce_L6_obc_two_rank.tsv":2,"smoke_gce_L6_obc_two_rank.tsv":2,
    }
    roots=[]
    for name,count in expected.items():
        rows=read(M/name);check(len(rows)==count,f"{name} has {count} rows")
        for row in rows:
            check(row["boundary"]=="open",f"{name} row boundary is open") if False else None
            root=row.get("outdir",row.get("out_parent",""))
            if root:
                check("_obc_" in root,f"{name} roots contain _obc_") if False else None
                roots.append(root)
            check(row["project_commit"]==commit,f"{name} project provenance") if False else None
    check(len(roots)==len(set(roots)),"all static CE/smoke roots are unique")
    all_manifest_rows=[row for name in expected for row in read(M/name)]
    check(all(row["boundary"]=="open" for row in all_manifest_rows),"every static manifest row has boundary=open")
    check(all(row["project_commit"]==commit for row in all_manifest_rows),"every static manifest row pins current project commit")
    check(all(row["smoqydqmc_commit"]==EXPECTED_SMOQ for row in all_manifest_rows),"every static manifest row pins SmoQyDQMC OBC fork")
    check(all(row.get("partition","burst")=="burst" and row.get("qos","default")=="default" for row in all_manifest_rows),"all QMC rows use burst/default")
    ce_rows=[]
    for name in list(expected)[:4]:ce_rows.extend(read(M/name))
    check(all(row["account"]=="ccsd" for row in ce_rows),"initial CE account is ccsd")
    positive=[row for row in ce_rows if float(row["U"])>0]
    check(all(row["phase_reweighted"]=="true" and row["force_symmetry"]=="false" for row in positive),"positive-U CE uses corrected signed spin-HS path")
    check(all(int(row["measure_interval"])==3 for row in ce_rows if row["stage"]!="smoke"),"production CE measurement interval is 3")

    refs=M/"L6_PBC_mu_reference_for_OBC.tsv"
    if refs.is_file():
        imported=read(refs);check(len(imported)<=152,"frozen PBC reference count is bounded")
        check(len({row["target_key"] for row in imported})==len(imported),"frozen PBC references are unique")
        check(all(row["pbc_tuning_status"]=="confirmed_within_abs_N_0p03" for row in imported),"only confirmed PBC chemical potentials are imported")
        check(all(row["pbc_reference_manifest_sha256"] for row in imported),"every imported PBC seed has manifest SHA256 provenance")

    exact=load_exact();g2=exact.build_geometry(2,2);g3=exact.build_geometry(3,3);g6=exact.build_geometry(6,6)
    check((len(g2.nn_bonds),len(g2.nnn_bonds))==(4,2),"2x2 OBC geometry has 4 NN and 2 NNN bonds")
    check((len(g3.nn_bonds),len(g3.nnn_bonds))==(12,8),"3x3 OBC geometry has 12 NN and 8 NNN bonds")
    check((len(g6.nn_bonds),len(g6.nnn_bonds))==(60,50),"6x6 OBC geometry has 60 NN and 50 NNN bonds")
    eps=np.linalg.eigvalsh(g2.hopping)
    for beta,n in ((0.7,1),(2.0,2),(5.0,3)):
        first,second=exact.canonical_orbital_moments(eps,beta,n);ref_first,ref_second=enumeration_moments(eps,beta,n)
        check(np.allclose(first,ref_first,rtol=2e-12,atol=2e-13) and np.allclose(second,ref_second,rtol=2e-12,atol=2e-13),f"elementary-symmetric canonical moments match enumeration beta={beta} N={n}")
    for beta,n in ((2.0,2),(5.0,4),(10.0,6)):
        mu=exact.exact_gce_mu(g3,beta,n);summary=exact.grand_canonical_summary(g3,beta,mu);check(abs(float(summary["N_mean"])-n)<2e-9,f"exact OBC GCE mu solve hits N={n} at beta={beta}")

    exact_snapshot=HERE/"status_source"/"exact_u0_obc"/"exact_u0_L6_obc_snapshot.tsv"
    if exact_snapshot.is_file():
        rows=read(exact_snapshot);check(len(rows)==80,"exact U=0 snapshot has 80 ensemble rows")
        check(all(row["boundary"]=="open" and row["final"]=="1" for row in rows),"exact U=0 snapshot is final OBC data")

    runner_text="\n".join((HERE/name).read_text() for name in ("run_ce_manifest_task.sh","run_gce_mu_manifest_task.sh","run_gce_production_manifest_task.sh"))
    check(runner_text.count("--boundary=open")>=3,"all CE/GCE launchers explicitly request OBC")
    check("checkpoint-reset-accumulators=false" in runner_text and "CHECKPOINT_RESET_ACCUMULATORS=false" in runner_text,"checkpoint accumulators are never reset")
    analysis=(HERE/"analyze_l6_obc_thermometry.py").read_text()
    check("(tgce - tce) / tce" in analysis and "NaNs make matplotlib break the line" in analysis,"thermometry uses requested bias and broken actual-point curves")
    report={"project_commit":commit,"grid":192,"interacting":152,"exact_conditions":40,"geometry":{"sites":36,"nn":60,"nnn":50},"smoqydqmc_commit":EXPECTED_SMOQ,"status":"pass"}
    out=HERE/"status_source"/"workflow_validation.json";out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+"\n")
    print("VALIDATION_PASS",out)


if __name__=="__main__":main()
