#!/usr/bin/env bash
# Two-rank CADES gate for OBC geometry, MPI, checkpoint/resume, CE, and GCE.
set -euo pipefail

PROJECT=${PROJECT:-/home/9pm/nUHubbard_obc_dev}
EXPECTED_COMMIT=c5f0c81bc98029bae585e0cb283428e293553999
RANKS=${SLURM_NTASKS:-2}
[[ "${RANKS}" -eq 2 ]] || { echo "smoke requires exactly two ranks" >&2; exit 2; }
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  [[ "${SLURM_JOB_ACCOUNT:-}" == ccsd ]] || { echo "smoke must remain on ccsd" >&2; exit 2; }
  [[ "${SLURM_JOB_PARTITION:-}" == burst && "${SLURM_JOB_QOS:-}" == default ]] || {
    echo "smoke requires burst/default" >&2; exit 2;
  }
fi

cd "${PROJECT}"
unset LD_LIBRARY_PATH MPI_PATH MPI_ROOT MPICC MPICXX MPIF77 MPIF90 MPIFC OMPI_MCA_pml OMPI_MCA_btl || true
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,tcp
export OMPI_MCA_btl_tcp_if_include=${OMPI_MCA_btl_tcp_if_include:-mgmt0}
export OMPI_MCA_oob_tcp_if_include=${OMPI_MCA_oob_tcp_if_include:-mgmt0}
export PRTE_MCA_oob_tcp_if_include=${PRTE_MCA_oob_tcp_if_include:-mgmt0}
export PMIX_MCA_ptl_tcp_if_include=${PMIX_MCA_ptl_tcp_if_include:-mgmt0}
export PATH="${PROJECT}/.julia_depot/bin:$HOME/.julia/bin:$HOME/.juliaup/bin:$PATH"
export JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH:-${PROJECT}/.julia_depot:$HOME/.julia}
export JULIA_PROJECT=${JULIA_PROJECT:-${PROJECT}/julia_env}
export JULIA_BINDIR=${JULIA_BINDIR:-/home/9pm/.julia/juliaup/julia-1.12.1+0.x64.linux.gnu/bin}
JULIA_BIN=${JULIA_BIN:-${JULIA_BINDIR}/julia}
MPIEXECJL=${MPIEXECJL:-mpiexecjl}
export JULIA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE JULIA_PKG_PRECOMPILE_AUTO=0 CHECKPOINT_RESET_ACCUMULATORS=false

test "$(git -C external/SmoQyDQMC rev-parse HEAD)" = "${EXPECTED_COMMIT}"
"${JULIA_BIN}" --project="${JULIA_PROJECT}" -e '
    using MPIPreferences, SmoQyDQMC
    MPIPreferences.binary == "OpenMPI_jll" || error("expected OpenMPI_jll")
    Base.pkgversion(SmoQyDQMC) == v"2.0.12" || error("wrong SmoQyDQMC version")
'
"${JULIA_BIN}" --project="${JULIA_PROJECT}" \
  scripts/interacting_qmc_ed/test_square_lattice_geometry.jl

MPI_LOG=$(mktemp)
"${MPIEXECJL}" --project="${JULIA_PROJECT}" -n 2 "${JULIA_BIN}" --project="${JULIA_PROJECT}" -e '
    using MPI
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)
    if rank == 0
        println("mpi_library=$(MPI.Get_library_version())")
        flush(stdout)
    end
    MPI.Barrier(comm)
    for printing_rank in 0:(size - 1)
        if rank == printing_rank
            println("rank=$(rank) size=$(size)")
            flush(stdout)
        end
        MPI.Barrier(comm)
    end
    MPI.Finalize()
' | tee "${MPI_LOG}"
grep -qi 'open mpi' "${MPI_LOG}"
grep -q 'rank=0 size=2' "${MPI_LOG}"
grep -q 'rank=1 size=2' "${MPI_LOG}"

JOB_TOKEN=${SLURM_JOB_ID:-local}
RUN_BASE=${RUN_BASE:-/home/9pm/nUHubbard_obc_runs/obc_ce_gce_validation_20260720/smoke_job${JOB_TOKEN}}
CE_ROOT=${RUN_BASE}/ce_obc_checkpoint
GCE_ATTR_ROOT=${RUN_BASE}/gce_obc_attractive_checkpoint
GCE_SPIN_ROOT=${RUN_BASE}/gce_obc_spin_fresh
[[ "${RUN_BASE}" == *"obc_"* ]] || { echo "smoke root must contain obc_" >&2; exit 2; }
mkdir -p "${RUN_BASE}"

CE_DRIVER=${PROJECT}/scripts/interacting_qmc_ed/benchmark_ce_green_tau_space_mpi_3x3.jl
CE_COMMON=(
  --lx=2 --ly=2 --boundary=open --nup=1 --ndn=1 --u=-3 --beta=1 --dtau=0.2
  --nwarmups=4 --batch-nsamples=1 --max-batches=2 --measure-interval=1
  --stab-interval=2 --cluster-size=3 --num-fourier-points=auto --nfreq=1
  --seed=20260720 --measure-greens=false --measure-bkt=false --measure-equal-time=true
  --measure-equal-time-correlations=true --phase-reweight=false --force-symmetry=true
  --checkpoint-enable=true --checkpoint-file=checkpoint.jls --checkpoint-every-batches=1
  --checkpoint-warmup-chunk=2 --checkpoint-keep=true --checkpoint-reset-accumulators=false
  --checkpoint-sync-timeout-seconds=300 --checkpoint-sync-poll-seconds=2
  --python=python3 --output-dir="${CE_ROOT}"
)
set +e
"${MPIEXECJL}" --project="${JULIA_PROJECT}" -n 2 "${JULIA_BIN}" --project="${JULIA_PROJECT}" "${CE_DRIVER}" \
  "${CE_COMMON[@]}" --runtime-limit-hours=0.000000001
CE_FIRST_RC=$?
set -e
[[ "${CE_FIRST_RC}" -eq 13 ]] || { echo "CE checkpoint leg returned ${CE_FIRST_RC}, expected checkpoint exit 13" >&2; exit 2; }
[[ $(find "${CE_ROOT}/ranks" -mindepth 2 -maxdepth 2 -name checkpoint.jls | wc -l | tr -d ' ') -eq 2 ]]
[[ $(find "${CE_ROOT}/ranks" -mindepth 2 -maxdepth 2 -name checkpoint.jls.status | wc -l | tr -d ' ') -eq 2 ]]
"${MPIEXECJL}" --project="${JULIA_PROJECT}" -n 2 "${JULIA_BIN}" --project="${JULIA_PROJECT}" "${CE_DRIVER}" \
  "${CE_COMMON[@]}" --runtime-limit-hours=0

PRIMARY=(equal_time_kinetic_per_site_qmc.tsv equal_time_double_occupancy_per_site_qmc.tsv equal_time_nn_spin_qmc.tsv equal_time_nn_connected_charge_qmc.tsv)
[[ $(find "${CE_ROOT}/ranks" -mindepth 2 -maxdepth 2 -name checkpoint_complete.txt | wc -l | tr -d ' ') -eq 2 ]]
for file in "${PRIMARY[@]}"; do test -s "${CE_ROOT}/${file}"; done

ATTR_DRIVER=${PROJECT}/scripts/interacting_qmc_ed/run_smoqydqmc_attractive_hubbard_checkpoint.jl
ATTR_SID=$(( (${SLURM_JOB_ID:-720}) % 100000000 ))
GCE_COMMON=(
  "${ATTR_SID}" -3.0 0.0 0.0 2 1.0 4 2 2 1 1.0
)
GCE_TAIL=(true "${GCE_ATTR_ROOT}" 2 equal-time-only 0.2 2 1e-6 false false 1 20260721 1 --boundary=open)
set +e
"${MPIEXECJL}" --project="${JULIA_PROJECT}" -n 2 "${JULIA_BIN}" --project="${JULIA_PROJECT}" "${ATTR_DRIVER}" \
  "${GCE_COMMON[@]}" 0.000000001 "${GCE_TAIL[@]}"
GCE_FIRST_RC=$?
set -e
[[ "${GCE_FIRST_RC}" -eq 13 ]] || { echo "GCE checkpoint leg returned ${GCE_FIRST_RC}, expected checkpoint exit 13" >&2; exit 2; }
GCE_INCOMPLETE=$(find "${GCE_ATTR_ROOT}" -mindepth 1 -maxdepth 1 -type d ! -name 'complete_*' | head -1)
[[ -n "${GCE_INCOMPLETE}" ]]
[[ $(find "${GCE_INCOMPLETE}" -maxdepth 1 -name 'checkpoint_pID-*.jld2' | wc -l | tr -d ' ') -eq 2 ]]
"${MPIEXECJL}" --project="${JULIA_PROJECT}" -n 2 "${JULIA_BIN}" --project="${JULIA_PROJECT}" "${ATTR_DRIVER}" \
  "${GCE_COMMON[@]}" Inf "${GCE_TAIL[@]}"
ATTR_COMPLETE=$(find "${GCE_ATTR_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'complete_*')
[[ -n "${ATTR_COMPLETE}" && $(printf '%s\n' "${ATTR_COMPLETE}" | wc -l | tr -d ' ') -eq 1 ]]
[[ $(find "${ATTR_COMPLETE}" -maxdepth 1 -name 'obc_equal_time_rank_pID-*.tsv' | wc -l | tr -d ' ') -eq 2 ]]
[[ $(find "${ATTR_COMPLETE}" -maxdepth 1 -name 'obc_equal_time_site_rank_pID-*.tsv' | wc -l | tr -d ' ') -eq 2 ]]
for file in "${PRIMARY[@]}"; do test -s "${ATTR_COMPLETE}/${file}"; done

SPIN_DRIVER=${PROJECT}/scripts/interacting_qmc_ed/run_smoqydqmc_hubbard_spin_hs_checkpoint.jl
SPIN_SID=$((ATTR_SID + 1))
"${MPIEXECJL}" --project="${JULIA_PROJECT}" -n 2 "${JULIA_BIN}" --project="${JULIA_PROJECT}" "${SPIN_DRIVER}" \
  "${SPIN_SID}" 3.0 0.0 0.0 2 1.0 1 2 2 1 0.0 Inf true \
  "${GCE_SPIN_ROOT}" 2 equal-time-only 0.2 2 1e-6 false false 1 20260722 0 --boundary=open
SPIN_COMPLETE=$(find "${GCE_SPIN_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'complete_*')
[[ -n "${SPIN_COMPLETE}" && $(printf '%s\n' "${SPIN_COMPLETE}" | wc -l | tr -d ' ') -eq 1 ]]
[[ $(find "${SPIN_COMPLETE}" -maxdepth 1 -name 'obc_equal_time_rank_pID-*.tsv' | wc -l | tr -d ' ') -eq 2 ]]
for file in "${PRIMARY[@]}"; do test -s "${SPIN_COMPLETE}/${file}"; done

cat >"${RUN_BASE}/smoke_complete.txt" <<EOF
status=PASS
smoqydqmc_version=2.0.12
smoqydqmc_commit=${EXPECTED_COMMIT}
mpi_ranks=2
ce_checkpoint_first_rc=${CE_FIRST_RC}
gce_checkpoint_first_rc=${GCE_FIRST_RC}
ce_root=${CE_ROOT}
gce_attractive_root=${GCE_ATTR_ROOT}
gce_spin_root=${GCE_SPIN_ROOT}
EOF
echo "CADES OBC smoke PASS: ${RUN_BASE}"
