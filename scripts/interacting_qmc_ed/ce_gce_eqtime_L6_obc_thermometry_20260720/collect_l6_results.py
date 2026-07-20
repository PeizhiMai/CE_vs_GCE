#!/usr/bin/env python3
"""Collect only strict-final L=6 OBC CE/GCE equal-time thermometry rows."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path
from typing import Iterable

HERE=Path(__file__).resolve().parent
M=HERE/"manifests"
PRIMARY={
    "kinetic":"equal_time_kinetic_per_site_qmc.tsv",
    "double_occupancy":"equal_time_double_occupancy_per_site_qmc.tsv",
    "nn_spin":"equal_time_nn_spin_qmc.tsv",
    "nn_charge_connected":"equal_time_nn_connected_charge_qmc.tsv",
}
SNAPSHOT_FIELDS=(
    "L","boundary","U","ensemble","Ntot","target_density","beta","T","N_mean","density",
    "kinetic","kinetic_err","interaction","interaction_err","total","total_err",
    "double_occupancy","double_occupancy_err","local_moment","local_moment_err",
    "nn_spin","nn_spin_err","nn_charge_connected","nn_charge_connected_err",
    "nnn_spin","nnn_spin_err","nnn_charge_connected","nnn_charge_connected_err",
    "nsamples","batches","nranks","average_phase","average_phase_abs","average_phase_err",
    "mu","mu_L6_PBC_reference","mu_L8_reference","L8_reference_Ntot","final","sign_limited",
    "source","workflow","project_commit","smoqydqmc_version","smoqydqmc_commit",
    "site_count","nn_bond_count","nnn_bond_count",
)
STATUS_FIELDS=(
    "target_key","U","Ntot","beta","T","ensemble","status","rank_coverage","site_coverage",
    "table_coverage","checkpoint_coverage","density_error","problem","root",
)


def read(path:Path)->list[dict[str,str]]:
    with path.open(newline="") as handle:return list(csv.DictReader(handle,delimiter="\t"))


def one(path:Path)->dict[str,str]:
    rows=read(path)
    if len(rows)!=1:raise ValueError(f"{path}: expected one row, found {len(rows)}")
    return rows[0]


def write(path:Path,rows:Iterable[dict[str,object]],fields:Iterable[str])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,delimiter="\t",fieldnames=list(fields),extrasaction="ignore");writer.writeheader();writer.writerows(rows)


def finite(value:object)->float:
    result=float(value)
    if not math.isfinite(result):raise ValueError(f"non-finite value {value!r}")
    return result


def bonds(lx:int=6,ly:int=6):
    nn=[];nnn=[]
    for y in range(ly):
        for x in range(lx):
            i=x+y*lx
            if x+1<lx:nn.append((i,x+1+y*lx))
            if y+1<ly:nn.append((i,x+(y+1)*lx))
            if x+1<lx and y+1<ly:nnn.append((i,x+1+(y+1)*lx))
            if x-1>=0 and y+1<ly:nnn.append((i,x-1+(y+1)*lx))
    assert len(nn)==60 and len(nnn)==50
    return nn,nnn
NN_BONDS,NNN_BONDS=bonds()


def forbidden(root:Path)->list[str]:
    names=("equal_time_charge_spin_wedge_qmc.tsv","equal_time_structure_factors_qmc.tsv","equal_time_neighbor_shells_qmc.tsv","bkt_observables_qmc.tsv")
    found=[str(root/name) for name in names if (root/name).exists()]
    for pattern in ("pair/*","greens/*","current/*","time-displaced/*"):
        found.extend(str(path) for path in root.glob(pattern))
    return found


def phase_sem(values:list[complex])->float:
    magnitudes=[abs(value) for value in values]
    return statistics.stdev(magnitudes)/math.sqrt(len(magnitudes)) if len(magnitudes)>1 else 0.0


def check_ce_pool(root:Path,expected:int,combined:dict[str,dict[str,str]],bond_rows:dict[str,dict[str,str]])->tuple[float,float,float]:
    phase_total=0.0;sample_total=0.0;phase_by_rank=[]
    signed={name:0.0 for name in ("kinetic","double_occupancy","nn_spin")}
    raw_nn=raw_nnn=0.0;site_signed=[0.0]*36
    rank_dirs=sorted(root.glob("ranks/rank_*"))
    if len(rank_dirs)!=expected:raise ValueError(f"rank directories {len(rank_dirs)}/{expected}")
    for rank in rank_dirs:
        kinetic=one(rank/PRIMARY["kinetic"]);n=finite(kinetic["nsamples"]);p=finite(kinetic["phase_sum"])
        if n<=0 or kinetic.get("boundary")!="open":raise ValueError(f"invalid rank primary {rank}")
        phase_total+=p;sample_total+=n;phase_by_rank.append(complex(p/n,0))
        for name in signed:
            q=one(rank/PRIMARY[name]);signed[name]+=finite(q["value"])*finite(q["phase_sum"])
        br={row["shell"]:row for row in read(rank/"equal_time_bond_observables_qmc.tsv")}
        raw_nn+=finite(br["NN"]["charge_corr_raw_signed_sum"])
        raw_nnn+=finite(br["NNN"]["charge_corr_raw_signed_sum"])
        sites=read(rank/"equal_time_site_density_qmc.tsv")
        if len(sites)!=36:raise ValueError(f"rank site coverage {len(sites)}/36")
        for item in sites:site_signed[int(item["site"])-1]+=finite(item["density_signed_sum"])
    if abs(phase_total)<=100*math.ulp(1.0)*max(sample_total,1):raise ValueError("numerical-zero CE phase denominator")
    pooled={name:value/phase_total for name,value in signed.items()}
    site=[value/phase_total for value in site_signed]
    pooled["nn_charge_connected"]=raw_nn/phase_total-sum(site[i]*site[j] for i,j in NN_BONDS)/60
    pooled_nnn=raw_nnn/phase_total-sum(site[i]*site[j] for i,j in NNN_BONDS)/50
    for name,value in pooled.items():
        if not math.isclose(value,finite(combined[name]["value"]),rel_tol=2e-8,abs_tol=2e-10):raise ValueError(f"CE global phase pool mismatch {name}")
    if not math.isclose(pooled_nnn,finite(bond_rows["NNN"]["charge_corr_connected"]),rel_tol=2e-8,abs_tol=2e-10):raise ValueError("CE NNN connected global pool mismatch")
    avg=phase_total/sample_total
    if not math.isclose(avg,finite(combined["kinetic"]["average_phase"]),rel_tol=2e-9,abs_tol=2e-11):raise ValueError("CE average phase mismatch")
    return avg,phase_sem(phase_by_rank),sample_total


def inspect_ce(row:dict[str,str])->tuple[dict[str,object]|None,dict[str,object]]:
    root=Path(row["outdir"]);expected=int(row["expected_ranks"])
    ranks=len(list(root.glob("ranks/rank_*/checkpoint_complete.txt")));sites=len(list(root.glob("ranks/rank_*/equal_time_site_density_qmc.tsv")))
    tables=sum((root/name).is_file() for name in PRIMARY.values());checkpoints=len(list(root.glob("ranks/rank_*/checkpoint.jls")))
    strict=(root/"obc_thermometry_complete.txt").is_file() and ranks==expected and sites==expected and tables==4
    status={"target_key":row["target_key"],"U":row["U"],"Ntot":row["Ntot"],"beta":row["beta"],"T":1/float(row["beta"]),"ensemble":"CE","status":"incomplete","rank_coverage":f"{ranks}/{expected}","site_coverage":f"{sites}/{expected}","table_coverage":f"{tables}/4","checkpoint_coverage":f"{checkpoints}/{expected}","density_error":"","problem":"","root":str(root)}
    if not strict:return None,status
    try:
        bad=forbidden(root)
        if bad:raise ValueError(f"forbidden translational/momentum outputs: {bad[:3]}")
        combined={name:one(root/path) for name,path in PRIMARY.items()}
        for item in combined.values():
            if item.get("boundary")!="open" or int(float(item["nranks"]))!=expected:raise ValueError("bad OBC primary metadata")
            finite(item["value"]);finite(item["stderr"])
        scalar=one(root/"equal_time_observables_qmc.tsv")
        bonds_by_shell={item["shell"]:item for item in read(root/"equal_time_bond_observables_qmc.tsv")}
        if int(bonds_by_shell["NN"]["bond_count"])!=60 or int(bonds_by_shell["NNN"]["bond_count"])!=50:raise ValueError("bad CE OBC bond counts")
        avg,avg_err,nsamples=check_ce_pool(root,expected,combined,bonds_by_shell)
        if float(row["U"])>0 and row.get("phase_reweighted")!="true":raise ValueError("positive-U manifest lacks phase reweighting")
        result={
            "L":6,"boundary":"open","U":float(row["U"]),"ensemble":"CE","Ntot":int(row["Ntot"]),"target_density":float(row["density"]),"beta":float(row["beta"]),"T":1/float(row["beta"]),
            "N_mean":finite(scalar["ntotal"]),"density":finite(scalar["density"]),
            "kinetic":finite(combined["kinetic"]["value"]),"kinetic_err":finite(combined["kinetic"]["stderr"]),
            "interaction":finite(scalar["interaction_per_site"]),"interaction_err":finite(scalar["interaction_stderr"]),"total":finite(scalar["total_per_site"]),"total_err":finite(scalar["total_stderr"]),
            "double_occupancy":finite(combined["double_occupancy"]["value"]),"double_occupancy_err":finite(combined["double_occupancy"]["stderr"]),
            "local_moment":finite(scalar["local_moment_z"]),"local_moment_err":finite(scalar["local_moment_z_stderr"]),
            "nn_spin":finite(combined["nn_spin"]["value"]),"nn_spin_err":finite(combined["nn_spin"]["stderr"]),
            "nn_charge_connected":finite(combined["nn_charge_connected"]["value"]),"nn_charge_connected_err":finite(combined["nn_charge_connected"]["stderr"]),
            "nnn_spin":finite(bonds_by_shell["NNN"]["spin_corr_s_s"]),"nnn_spin_err":finite(bonds_by_shell["NNN"]["spin_corr_s_s_stderr"]),
            "nnn_charge_connected":finite(bonds_by_shell["NNN"]["charge_corr_connected"]),"nnn_charge_connected_err":finite(bonds_by_shell["NNN"]["charge_corr_connected_stderr"]),
            "nsamples":nsamples,"batches":finite(scalar["batches"]),"nranks":expected,"average_phase":avg,"average_phase_abs":abs(avg),"average_phase_err":avg_err,
            "mu":"","mu_L6_PBC_reference":"","mu_L8_reference":"","L8_reference_Ntot":"","final":1,"sign_limited":0,"source":str(root),"workflow":"L6_OBC_CE_QMC_20260720",
            "project_commit":row["project_commit"],"smoqydqmc_version":row["smoqydqmc_version"],"smoqydqmc_commit":row["smoqydqmc_commit"],"site_count":36,"nn_bond_count":60,"nnn_bond_count":50,
        }
        status["status"]="strict_final";return result,status
    except Exception as exc:
        status["status"]="invalid";status["problem"]=str(exc);return None,status


def gce_base(row:dict[str,str])->str:
    prefix="attractive_hubbard_obc_rect" if row["family"]=="attractive" else "hubbard_spin_hs_obc_rect"
    return f"{prefix}_U{float(row['U']):.2f}_tp0.00_mu{float(row['mu_final']):.2f}_Lx6_Ly6_b{float(row['beta']):.2f}-{int(row['sid'])}"


def check_gce_pool(root:Path,expected:int,combined:dict[str,dict[str,str]],bonds_by_shell:dict[str,dict[str,str]])->tuple[complex,float,float]:
    rank_files=sorted(root.glob("obc_equal_time_rank_pID-*.tsv"));site_files=sorted(root.glob("obc_equal_time_site_rank_pID-*.tsv"))
    if len(rank_files)!=expected or len(site_files)!=expected:raise ValueError("GCE rank/site accumulator coverage")
    phase=0j;samples=0.0;phase_rank=[];signed={};site_signed=[0j]*36
    for rank_path,site_path in zip(rank_files,site_files):
        rows={item["name"]:item for item in read(rank_path)};kin=rows["kinetic_per_site"]
        p=complex(finite(kin["phase_sum_real"]),finite(kin["phase_sum_imag"]));n=finite(kin["nsamples"])
        phase+=p;samples+=n;phase_rank.append(p/n)
        for name,item in rows.items():signed[name]=signed.get(name,0j)+complex(finite(item["signed_sum_real"]),finite(item["signed_sum_imag"]))
        sites=read(site_path)
        if len(sites)!=36:raise ValueError("GCE rank site row count")
        for item in sites:
            site_signed[int(item["site"])-1]+=complex(finite(item["density_signed_sum_real"]),finite(item["density_signed_sum_imag"]))
    if abs(phase)<=100*math.ulp(1.0)*max(samples,1):raise ValueError("numerical-zero GCE phase denominator")
    values={name:(value/phase).real for name,value in signed.items()};site=[(value/phase).real for value in site_signed]
    pooled={"kinetic":values["kinetic_per_site"],"double_occupancy":values["double_occupancy_per_site"],"nn_spin":values["nn_spin_s_s"],"nn_charge_connected":values["nn_charge_raw"]-sum(site[i]*site[j] for i,j in NN_BONDS)/60}
    nnn_connected=values["nnn_charge_raw"]-sum(site[i]*site[j] for i,j in NNN_BONDS)/50
    for name,value in pooled.items():
        if not math.isclose(value,finite(combined[name]["value"]),rel_tol=2e-8,abs_tol=2e-10):raise ValueError(f"GCE global phase pool mismatch {name}")
    if not math.isclose(nnn_connected,finite(bonds_by_shell["NNN"]["charge_corr_connected"]),rel_tol=2e-8,abs_tol=2e-10):raise ValueError("GCE NNN connected global pool mismatch")
    return phase/samples,phase_sem(phase_rank),samples


def inspect_gce(row:dict[str,str])->tuple[dict[str,object]|None,dict[str,object]]:
    parent=Path(row["out_parent"]);root=parent/f"complete_{gce_base(row)}";expected=int(row["expected_ranks"])
    ranks=len(list(root.glob("obc_equal_time_rank_pID-*.tsv")));sites=len(list(root.glob("obc_equal_time_site_rank_pID-*.tsv")));tables=sum((root/name).is_file() for name in PRIMARY.values())
    achieved=parent/"achieved_density.tsv";density_error=math.nan;density_ok=False
    if achieved.is_file():
        try:density_error=float(read(achieved)[0]["delta_N"]);density_ok=abs(density_error)<=float(row["density_tolerance"])
        except (IndexError,KeyError,ValueError):pass
    strict=(parent/"dqmc_gce_obc_eqtime_complete.txt").is_file() and ranks==expected and sites==expected and tables==4 and density_ok
    status={"target_key":row["target_key"],"U":row["U"],"Ntot":row["Ntot_target"],"beta":row["beta"],"T":1/float(row["beta"]),"ensemble":"GCE","status":"incomplete","rank_coverage":f"{ranks}/{expected}","site_coverage":f"{sites}/{expected}","table_coverage":f"{tables}/4","checkpoint_coverage":"complete" if root.is_dir() else "0/32","density_error":density_error,"problem":"","root":str(parent)}
    if not strict:return None,status
    try:
        bad=forbidden(root)
        if bad:raise ValueError(f"forbidden translational/momentum outputs: {bad[:3]}")
        combined={name:one(root/path) for name,path in PRIMARY.items()}
        for item in combined.values():
            if item.get("boundary")!="open" or int(item["nranks"])!=expected or item["smoqydqmc_commit"]!=row["smoqydqmc_commit"]:raise ValueError("bad GCE OBC primary provenance")
            finite(item["value"]);finite(item["stderr"])
        scalar=one(root/"equal_time_observables_obc_qmc.tsv");bond_rows={item["shell"]:item for item in read(root/"equal_time_bond_observables_qmc.tsv")}
        if int(bond_rows["NN"]["bond_count"])!=60 or int(bond_rows["NNN"]["bond_count"])!=50:raise ValueError("bad GCE OBC bond counts")
        avg,avg_err,nsamples=check_gce_pool(root,expected,combined,bond_rows)
        nmean=finite(scalar["achieved_N"])
        if not math.isclose(nmean-int(row["Ntot_target"]),density_error,rel_tol=0,abs_tol=2e-10):raise ValueError("achieved-N marker mismatch")
        result={
            "L":6,"boundary":"open","U":float(row["U"]),"ensemble":"GCE","Ntot":int(row["Ntot_target"]),"target_density":float(row["target_density"]),"beta":float(row["beta"]),"T":1/float(row["beta"]),"N_mean":nmean,"density":finite(scalar["density"]),
            "kinetic":finite(combined["kinetic"]["value"]),"kinetic_err":finite(combined["kinetic"]["stderr"]),"interaction":finite(scalar["interaction_per_site"]),"interaction_err":0.0,"total":finite(scalar["total_per_site"]),"total_err":math.nan,
            "double_occupancy":finite(combined["double_occupancy"]["value"]),"double_occupancy_err":finite(combined["double_occupancy"]["stderr"]),"local_moment":finite(scalar["local_moment"]),"local_moment_err":math.nan,
            "nn_spin":finite(combined["nn_spin"]["value"]),"nn_spin_err":finite(combined["nn_spin"]["stderr"]),"nn_charge_connected":finite(combined["nn_charge_connected"]["value"]),"nn_charge_connected_err":finite(combined["nn_charge_connected"]["stderr"]),
            "nnn_spin":finite(bond_rows["NNN"]["spin_corr_s_s"]),"nnn_spin_err":finite(bond_rows["NNN"]["spin_corr_s_s_stderr"]),"nnn_charge_connected":finite(bond_rows["NNN"]["charge_corr_connected"]),"nnn_charge_connected_err":finite(bond_rows["NNN"]["charge_corr_connected_stderr"]),
            "nsamples":nsamples,"batches":"","nranks":expected,"average_phase":avg.real,"average_phase_abs":abs(avg),"average_phase_err":avg_err,"mu":float(row["mu_final"]),"mu_L6_PBC_reference":float(row["mu_L6_PBC_reference"]),"mu_L8_reference":float(row["mu_L8_reference"]),"L8_reference_Ntot":int(row["L8_reference_Ntot"]),"final":1,"sign_limited":0,"source":str(parent),"workflow":"L6_OBC_GCE_QMC_20260720","project_commit":row["project_commit"],"smoqydqmc_version":row["smoqydqmc_version"],"smoqydqmc_commit":row["smoqydqmc_commit"],"site_count":36,"nn_bond_count":60,"nnn_bond_count":50,
        }
        status["status"]="strict_final";return result,status
    except Exception as exc:
        status["status"]="invalid";status["problem"]=str(exc);return None,status


def exact_rows(path:Path)->tuple[list[dict[str,object]],list[dict[str,object]]]:
    source=read(path);data=[];statuses=[]
    if len(source)!=80:raise ValueError(f"exact snapshot has {len(source)}/80 rows")
    for item in source:
        if item["boundary"]!="open" or int(item["site_count"])!=36 or int(item["nn_bond_count"])!=60 or int(item["nnn_bond_count"])!=50:raise ValueError("bad exact OBC provenance")
        row={field:item.get(field,"") for field in SNAPSHOT_FIELDS};row.update({"L":6,"target_density":int(item["Ntot"])/36,"interaction":item.get("interaction",0),"interaction_err":item.get("interaction_err",0),"total":item.get("total",item["kinetic"]),"total_err":item.get("total_err",0),"average_phase_abs":1.0,"nsamples":0,"batches":0,"nranks":0,"workflow":"L6_OBC_U0_EXACT_20260720","final":1,"sign_limited":0})
        data.append(row);statuses.append({"target_key":f"U0_L6OBC_N{int(item['Ntot']):03d}_b{float(item['beta']):g}","U":0,"Ntot":item["Ntot"],"beta":item["beta"],"T":item["T"],"ensemble":item["ensemble"],"status":"exact_final","rank_coverage":"exact","site_coverage":"exact","table_coverage":"4/4","checkpoint_coverage":"exact","density_error":0,"problem":"","root":item["source"]})
    return data,statuses


def manifest_rows(paths:list[Path],root_field:str)->list[dict[str,str]]:
    seen={}
    for path in paths:
        if not path.is_file():continue
        for row in read(path):
            root=row[root_field]
            if root in seen and seen[root]!=row:raise ValueError(f"duplicate conflicting root {root}")
            seen[root]=row
    return list(seen.values())


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--exact-snapshot",type=Path,required=True);parser.add_argument("--snapshot",type=Path,required=True);parser.add_argument("--status",type=Path,required=True);parser.add_argument("--require-complete",action="store_true");args=parser.parse_args()
    data,status=exact_rows(args.exact_snapshot)
    ce_paths=[M/"ce_L6_obc_attractive_beta_le10_r32_m50000.tsv",M/"ce_L6_obc_attractive_beta20_r64_m30000.tsv",M/"ce_L6_obc_positive_beta_le4_r32_m50000.tsv"]+sorted(M.glob("ce_L6_obc_positive_admitted_r*_m*.tsv"))
    ce=manifest_rows(ce_paths,"outdir")
    for row in ce:
        result,state=inspect_ce(row);status.append(state)
        if result:data.append(result)
    limited=M/"ce_L6_obc_positive_sign_limited.tsv"
    limited_keys=set()
    if limited.is_file():
        for row in read(limited):
            limited_keys.add((round(float(row["U"]),10),int(row["Ntot"]),round(float(row["beta"]),10)))
            status.append({"target_key":row["target_key"],"U":row["U"],"Ntot":row["Ntot"],"beta":row["beta"],"T":1/float(row["beta"]),"ensemble":"CE","status":"sign_limited","rank_coverage":f"{row['rank_complete']}/{row['expected_ranks']}","site_coverage":f"{row['rank_site_coverage']}/{row['expected_ranks']}","table_coverage":row["primary_table_coverage"],"checkpoint_coverage":"pilot_final","density_error":"","problem":"abs_average_phase_below_0p002","root":row["outdir"]})
    prod=manifest_rows(sorted(M.glob("gce_prod_L6_obc_*_confirmed.tsv")),"out_parent")
    for row in prod:
        result,state=inspect_gce(row);status.append(state)
        if result:data.append(result)
    # Add waiting placeholders for every interacting ensemble not represented yet.
    grid=read(M/"L6_obc_full_condition_grid.tsv")
    existing={(round(float(row["U"]),10),int(row["Ntot"]),round(float(row["beta"]),10),row["ensemble"]) for row in status}
    for row in grid:
        if float(row["U"])==0:continue
        for ensemble in ("CE","GCE"):
            key=(round(float(row["U"]),10),int(row["Ntot"]),round(float(row["beta"]),10),ensemble)
            if key in existing:continue
            status.append({"target_key":row["target_key"],"U":row["U"],"Ntot":row["Ntot"],"beta":row["beta"],"T":row["T"],"ensemble":ensemble,"status":"waiting_manifest","rank_coverage":"0/0","site_coverage":"0/0","table_coverage":"0/4","checkpoint_coverage":"0/0","density_error":"","problem":"production manifest not yet admitted/confirmed","root":""})
    keys=[(round(float(row["U"]),10),row["ensemble"],int(row["Ntot"]),round(float(row["beta"]),10)) for row in data]
    if len(keys)!=len(set(keys)):raise ValueError("duplicate collected ensemble condition")
    status_keys=[(round(float(row["U"]),10),row["ensemble"],int(row["Ntot"]),round(float(row["beta"]),10)) for row in status]
    if len(status_keys)!=len(set(status_keys)):raise ValueError("duplicate condition status")
    data.sort(key=lambda row:(float(row["U"]),int(row["Ntot"]),float(row["beta"]),row["ensemble"]));status.sort(key=lambda row:(float(row["U"]),int(row["Ntot"]),float(row["beta"]),row["ensemble"]))
    write(args.snapshot,data,SNAPSHOT_FIELDS);write(args.status,status,STATUS_FIELDS)
    terminal={"strict_final","exact_final","sign_limited"};unfinished=[row for row in status if row["status"] not in terminal]
    print(f"COLLECT snapshot_rows={len(data)} status_rows={len(status)}/384 unfinished={len(unfinished)} sign_limited={len(limited_keys)}")
    if args.require_complete and (len(status)!=384 or unfinished):raise SystemExit(f"strict collection incomplete: status={len(status)}/384 unfinished={len(unfinished)}")


if __name__=="__main__":
    main()
