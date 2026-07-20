#!/usr/bin/env bash
set -euo pipefail
PROJECT=${PROJECT:-/home/9pm/nUHubbard}
ROOT=${PROJECT}/runs/ce_gce_eqtime_L6_thermometry_20260719_smoke
ce_ok=0
for u in Um3 Up3; do
  d=${ROOT}/ce/${u}
  n=$(find "${d}/ranks" -mindepth 2 -maxdepth 2 -name checkpoint_complete.txt 2>/dev/null | wc -l | tr -d ' ')
  tables=0
  for f in equal_time_observables_qmc.tsv equal_time_charge_spin_wedge_qmc.tsv equal_time_structure_factors_qmc.tsv equal_time_neighbor_shells_qmc.tsv; do [[ -s "${d}/${f}" ]] && tables=$((tables+1)); done
  [[ -f "${d}/checkpoint_complete.txt" && "${n}" -eq 2 && "${tables}" -eq 4 ]] && ce_ok=$((ce_ok+1))
  echo "CE ${u}: rank_complete=${n}/2 tables=${tables}/4 root_marker=$([[ -f "${d}/checkpoint_complete.txt" ]] && echo yes || echo no)"
done
gce_ok=0
for fam in attractive spinHS; do
  d=${ROOT}/gce/${fam}
  c=$(find "${d}" -maxdepth 1 -type d -name 'complete_*' | head -1 || true)
  n=0; [[ -n "${c}" ]] && n=$(find "${c}" -maxdepth 1 -name 'simulation_info_sID-*_pID-*.toml' | wc -l | tr -d ' ')
  [[ -n "${c}" && -s "${c}/global_stats.csv" && "${n}" -eq 2 ]] && gce_ok=$((gce_ok+1))
  echo "GCE ${fam}: complete_dir=$([[ -n "${c}" ]] && echo yes || echo no) rank_info=${n}/2 global_stats=$([[ -n "${c}" && -s "${c}/global_stats.csv" ]] && echo yes || echo no)"
done
echo "SMOKE CE=${ce_ok}/2 GCE=${gce_ok}/2"
[[ "${ce_ok}" -eq 2 && "${gce_ok}" -eq 2 ]]
