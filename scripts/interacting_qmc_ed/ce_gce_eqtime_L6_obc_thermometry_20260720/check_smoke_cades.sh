#!/usr/bin/env bash
# Strictly validate CE/GCE OBC smoke outputs and mark the production gate.
set -euo pipefail
PROJECT=${PROJECT:-/home/9pm/nUHubbard_obc_dev}
WF=${PROJECT}/scripts/interacting_qmc_ed/ce_gce_eqtime_L6_obc_thermometry_20260720
ROOT=/home/9pm/nUHubbard_obc_runs/ce_gce_eqtime_L6_obc_thermometry_20260720_smoke
PRIMARY=(equal_time_kinetic_per_site_qmc.tsv equal_time_double_occupancy_per_site_qmc.tsv equal_time_nn_spin_qmc.tsv equal_time_nn_connected_charge_qmc.tsv)
ce_ok=0
for u in Um3 Up3; do
  d=${ROOT}/ce/${u}; ranks=$(find "${d}/ranks" -mindepth 2 -maxdepth 2 -name checkpoint_complete.txt 2>/dev/null | wc -l | tr -d ' ')
  sites=$(find "${d}/ranks" -mindepth 2 -maxdepth 2 -name equal_time_site_density_qmc.tsv 2>/dev/null | wc -l | tr -d ' ')
  tables=0; for f in "${PRIMARY[@]}"; do [[ -s "${d}/${f}" ]] && tables=$((tables+1)); done
  valid=no
  if [[ -f "${d}/obc_thermometry_complete.txt" && ${ranks} -eq 2 && ${sites} -eq 2 && ${tables} -eq 4 ]] && \
     python3 - "${d}" <<'PY'
import csv,pathlib,sys
root=pathlib.Path(sys.argv[1])
for name in ("equal_time_kinetic_per_site_qmc.tsv","equal_time_double_occupancy_per_site_qmc.tsv","equal_time_nn_spin_qmc.tsv","equal_time_nn_connected_charge_qmc.tsv"):
 row=list(csv.DictReader(open(root/name),delimiter="\t")); assert len(row)==1 and row[0]["boundary"]=="open" and int(row[0]["nranks"])==2
bond=list(csv.DictReader(open(root/"equal_time_bond_observables_qmc.tsv"),delimiter="\t"))
assert [(r["shell"],int(r["bond_count"])) for r in bond]==[("NN",4),("NNN",2)]
PY
  then ce_ok=$((ce_ok+1)); valid=yes; fi
  echo "CE ${u}: valid=${valid} ranks=${ranks}/2 sites=${sites}/2 tables=${tables}/4"
done

gce_ok=0
for fam in attractive spinHS; do
  d=${ROOT}/gce/${fam}; mapfile -t complete < <(find "${d}" -mindepth 1 -maxdepth 1 -type d -name 'complete_*_obc_*' 2>/dev/null)
  ranks=0; valid=no
  if [[ ${#complete[@]} -eq 1 ]]; then
    ranks=$(find "${complete[0]}" -maxdepth 1 -name 'simulation_info_sID-*_pID-*.toml' | wc -l | tr -d ' ')
    if [[ -s "${complete[0]}/global_stats.csv" && ${ranks} -eq 2 ]] && python3 - "${complete[0]}" <<'PY'
import pathlib,sys,tomllib
root=pathlib.Path(sys.argv[1]); paths=sorted(root.glob("simulation_info_sID-*_pID-*.toml")); assert len(paths)==2
for path in paths:
 d=tomllib.loads(path.read_text())["metadata"]
 assert d["boundary"]=="open" and d["geometry_site_count"]==4 and d["geometry_nn_bond_count"]==4 and d["geometry_nnn_bond_count"]==2
 assert d["smoqydqmc_version"]=="2.0.12"
PY
    then gce_ok=$((gce_ok+1)); valid=yes; fi
  fi
  echo "GCE ${fam}: valid=${valid} complete_dirs=${#complete[@]} rank_info=${ranks}/2"
done

echo "SMOKE CE=${ce_ok}/2 GCE=${gce_ok}/2"
[[ ${ce_ok} -eq 2 && ${gce_ok} -eq 2 ]]
mkdir -p "${WF}/status_source"
printf 'passed_at=%s\nproject_commit=%s\nsmoqydqmc_commit=%s\n' "$(date -Is)" "$(git -C "${PROJECT}" rev-parse HEAD)" "$(git -C "${PROJECT}/external/SmoQyDQMC" rev-parse HEAD)" > "${WF}/status_source/smoke_passed.txt"
