# CE unequal-time Green/current-response debugging status (2026-05-28)

Purpose: summarize what has already been done for the canonical-ensemble unequal-time benchmark work in `CanEnsAFQMC`, what is currently validated, what is still broken, and what should be done next.

Repository root used locally:

```text
/Users/cosdis/Desktop/projects/CE_GCE
```

---

## 1. Scope

This note concerns the fixed-sector `3x3` attractive-Hubbard benchmark

```text
Lx = 3
Ly = 3
Nup = 4
Ndn = 4
t = 1
U = -5
beta = 10
```

and the debugging of:

1. unequal-time single-particle Green functions,
2. Matsubara-frequency Green functions,
3. current-current response and superfluid density.

The main goal has been to understand why the canonical-ensemble current/superfluid-density observables disagree with ED even after the equal-time benchmark looks good.

---

## 2. Main scripts and files touched

### CE / benchmark drivers

```text
scripts/interacting_qmc_ed/benchmark_ce_qmc_vs_ed_3x3.jl
scripts/interacting_qmc_ed/benchmark_ce_current_vs_ed_3x3.jl
scripts/interacting_qmc_ed/benchmark_ce_current_vs_ed_3x3_exactsector.jl
scripts/interacting_qmc_ed/benchmark_ce_unequaltime_greens_exactsector_3x3.jl
scripts/interacting_qmc_ed/benchmark_ce_green_matsubara_3x3.jl
scripts/interacting_qmc_ed/ce_unequal_time_current_helpers.jl
scripts/interacting_qmc_ed/validate_ce_unequaltime_single_config.jl
scripts/interacting_qmc_ed/validate_ce_current_estimator_2x2.jl
scripts/interacting_qmc_ed/check_attractive_hs_slice_2x2.jl
```

### ED / reference scripts

```text
scripts/interacting_qmc_ed/ed_hubbard_ce_lowtemp_3x3.py
scripts/interacting_qmc_ed/ed_hubbard_ce_green_matsubara_3x3.py
scripts/interacting_qmc_ed/compare_ce_qmc_vs_ed_3x3.py
scripts/interacting_qmc_ed/compare_ce_current_vs_ed_3x3.py
scripts/interacting_qmc_ed/compare_ce_green_matsubara_vs_ed_3x3.py
scripts/interacting_qmc_ed/compare_ce_bkt_observables_3x3.py
```

### Existing benchmark note

```text
docs/ed_ceqmc_3x3_benchmark_notes.md
```

This new note is specifically about the later unequal-time/current debugging beyond the equal-time benchmark.

---

## 3. What has been done successfully

### 3.1 Equal-time CE vs ED benchmark was built and is in good shape

For the `3x3`, `Nup=Ndn=4`, `U=-5`, `beta=10` benchmark, the equal-time observables were compared between canonical ED and `CanEnsAFQMC`.

Best corrected CE result used:

```text
results/interacting_qmc_ed/ce_qmc_vs_ed_3x3_beta10_dtau0025_spincomplex
```

ED reference:

```text
results/interacting_qmc_ed/ed_ce_lowtemp_3x3_t1_Um5_Nup4_Ndn4_beta10
```

Representative agreement:

| observable | ED | CE-QMC |
|---|---:|---:|
| `E/site` | `-3.005734582` | `-3.013223443 ± 0.031813641` |
| `K/site` | `-1.291147958` | `-1.312686796 ± 0.010453411` |
| `V/site` | `-1.714586623` | `-1.700536647 ± 0.028634934` |
| `double occ/site` | `0.342917325` | `0.340107329 ± 0.005726987` |

Conclusion: the **equal-time CE machinery is basically working** once the attractive-`U` setup is corrected.

---

### 3.2 The attractive-`U` HS convention was corrected

Earlier CE benchmarks for attractive `U` were using the wrong HS choice.

Important diagnostic:

```text
scripts/interacting_qmc_ed/check_attractive_hs_slice_2x2.jl
```

This showed that for the attractive Hubbard model in this codebase the correct setup is:

```text
useChargeHST = false
sys_type = ComplexF64
```

and not the older real charge-HS setup.

The one-slice `2x2` check found:

- `useChargeHST=true, sys_type=Float64` -> huge mismatch
- `useChargeHST=false, sys_type=ComplexF64` -> machine-precision agreement

Conclusion: all attractive-`U` unequal-time/current benchmarks should use

```text
useChargeHST = false
sys_type = ComplexF64
forceSymmetry = true
```

unless there is a specific reason to test something else.

---

### 3.3 A major unequal-time indexing bug was found and fixed

The unequal-time CE code had an off-by-one in the backward product:

- old behavior effectively used `suffix[slice + 2]`
- corrected behavior uses `suffix[slice + 1]`

This bug affected the construction of `B(β,τ)` and therefore unequal-time Green functions and current response.

Files patched around this fix:

```text
scripts/interacting_qmc_ed/benchmark_ce_unequaltime_greens_exactsector_3x3.jl
scripts/interacting_qmc_ed/benchmark_ce_green_matsubara_3x3.jl
scripts/interacting_qmc_ed/ce_unequal_time_current_helpers.jl
scripts/interacting_qmc_ed/validate_ce_unequaltime_single_config.jl
```

Conclusion: this was a real bug and removing it materially improved the unequal-time benchmarks.

---

### 3.4 Free-fermion unequal-time Green in tau space is now validated

#### `U=0`, `dtau=0.05`

Validated against an exact-sector reference:

```text
results/interacting_qmc_ed/ce_unequaltime_greens_exactsector_3x3_u0_dtau005_mc32_fix1/greens_tau0_summary.tsv
```

Representative agreement:

| tau | exact | formula | delta |
|---:|---:|---:|---:|
| `0.05` | `0.5189578631688836` | `0.5188670623875307` | `-9.08e-5` |
| `0.25` | `0.4122386728376326` | `0.41223276741032167` | `-5.91e-6` |
| `1.0` | `0.36218032904535147` | `0.36218321479181936` | `+2.89e-6` |
| `5.0` | `16.490372450261987` | `16.490372450435196` | `+1.73e-10` |

#### `U=0`, `dtau=0.1`

The earlier `U=0` ED reference for Matsubara work turned out to be wrong because it used an interacting-style truncated low-energy sparse diagonalization path, which is not appropriate for the free benchmark.

That ED script was fixed so that for `U=0` it now uses an exact factorized free-fermion canonical reference:

```text
scripts/interacting_qmc_ed/ed_hubbard_ce_green_matsubara_3x3.py
```

New exact free reference:

```text
results/interacting_qmc_ed/ed_ce_green_matsubara_3x3_u0_beta10_dtau01_exact
```

Compared with:

```text
results/interacting_qmc_ed/ce_green_matsubara_3x3_u0_beta10_dtau01_mc40/greens_tau0_qmc.tsv
```

at `tau=0.1`:

- QMC: `0.48671112088032986`
- exact ED: `0.4866771033764165`
- delta: `+3.40e-05`

and the max absolute deviation over the full tau grid was also about `3.4e-05`.

Conclusion: the **free-case tau-space unequal-time Green benchmark is now in good shape**.

---

### 3.5 Free-fermion current response / stiffness benchmark is now validated

At `U=0`, `beta=10`, `dtau=0.05`, the current-response benchmark matches an exact-sector reference very well.

CE formula result:

```text
results/interacting_qmc_ed/ce_current_formula_3x3_u0_dtau005_mc32/summary.tsv
```

Exact-sector reference:

```text
results/interacting_qmc_ed/ce_current_vs_exactsector_3x3_u0_dtau005_mc32/summary.tsv
```

Comparison:

| observable | CE formula | exact-sector |
|---|---:|---:|
| `Lambda_L` | `0.7792326747377988` | `0.7792355645287168` |
| `Lambda_T` | `0.6679070776950428` | `0.6679161981692725` |
| `rho_s` | `0.027831399260689005` | `0.02782984158986107` |

Conclusion: the **free-fermion current-response path is basically correct** after the unequal-time indexing fix.

---

## 4. What has been done but is still not satisfactory

### 4.1 Interacting current response still disagrees with ED

Longer interacting current-response run:

```text
results/interacting_qmc_ed/ce_current_formula_3x3_um5_dtau0025_mc3000_w500_fix1/summary.tsv
```

ED reference:

```text
results/interacting_qmc_ed/ed_ce_lowtemp_3x3_t1_Um5_Nup4_Ndn4_beta10/summary.tsv
```

Comparison:

| observable | QMC | ED | QMC - ED |
|---|---:|---:|---:|
| `Lambda_L` | `0.6413672845 ± 0.0055895059` | `0.5909226659` | `+0.0504446187` |
| `Lambda_T` | `0.2608306377 ± 0.0237268089` | `0.0648384509` | `+0.1959921867` |
| `rho_s` | `0.0951341617 ± 0.0060957369` | `0.1315210537` | `-0.0363868920` |

Conclusion: although the free-case current benchmark is fine, the **interacting CE current-response / superfluid-density benchmark is still wrong**.

The largest discrepancy is in the transverse current response.

---

### 4.2 Interacting single-particle Green in Matsubara frequency is still badly wrong

Recent long interacting benchmark:

```text
results/interacting_qmc_ed/ce_green_matsubara_3x3_um5_beta10_dtau01_mc3000_w500_fixref
```

Comparison with current ED reference:

```text
results/interacting_qmc_ed/ce_green_matsubara_vs_ed_3x3_um5_beta10_dtau01_mc3000_w500_fixref/greens_iwn_comparison.tsv
```

At `tau=0.1`:

- QMC: `0.427864934214034 ± 0.001064934425780457`
- ED: `0.5165259002177045`
- delta: `-0.0886609660036705`

At the lowest Matsubara frequency:

- QMC: `-0.2821901915874258 - 0.0427665213338087 i`
- ED: `0.4216249233419165 - 0.4697995213864929 i`

so

- `delta_real = -0.7038151149293423`
- `delta_imag = +0.4270330000526842`

Conclusion: the **interacting Matsubara Green benchmark remains severely inconsistent with ED**.

---

## 5. Main diagnosis at this point

The debugging so far strongly suggests:

1. **Equal-time CE measurements are not the core problem.**
   - Energy, double occupancy, charge correlations, and other equal-time quantities are in reasonable agreement with ED.

2. **The free-case unequal-time tau-space construction is not the main problem anymore.**
   - After the indexing fix and the `U=0` ED-reference correction, that benchmark is good.

3. **The remaining problem is in the interacting unequal-time one-particle Green construction, and therefore in any observable that depends on it.**
   - This includes Matsubara Green functions.
   - It likely also contaminates current-current response and superfluid density.

4. **The current CE Matsubara construction should not yet be interpreted as the standard canonical time-ordered Green function.**
   - The present implementation was good enough for the restricted free tau-space check, but it is evidently not correct for the interacting Matsubara benchmark.

---

## 6. What remains to be done

### Priority 1: derive and implement the proper canonical time-ordered one-particle Green function

This is now the most important remaining task.

The implementation should explicitly handle:

1. the **addition branch** (`N -> N+1`),
2. the **removal branch** (`N -> N-1`),
3. the correct **time ordering**,
4. the correct **anti-periodic structure** in imaginary time.

The current Matsubara benchmark failure indicates that the present object being Fourier transformed is not yet the correct canonical Matsubara Green function.

---

### Priority 2: revalidate at `U=0`, `dtau=0.1` in Matsubara frequency

Before returning to the interacting case, the corrected canonical Matsubara construction should first pass the free benchmark:

```text
Lx = 3, Ly = 3, Nup = Ndn = 4, U = 0, beta = 10, dtau = 0.1
```

Target:

- QMC Matsubara data should match the exact free reference from

```text
results/interacting_qmc_ed/ed_ce_green_matsubara_3x3_u0_beta10_dtau01_exact
```

If that fails, the interacting benchmark should not be trusted yet.

---

### Priority 3: rerun the interacting Matsubara benchmark only after Priority 2 passes

Once the free Matsubara check passes, rerun:

```text
U = -5
beta = 10
dtau = 0.1
nwarmups = 500
measurements = 3000
```

and compare against:

```text
results/interacting_qmc_ed/ed_ce_green_matsubara_3x3_um5_beta10_dtau01
```

Only then will it make sense to decide whether the interacting discrepancy is:

1. still a CE unequal-time bug, or
2. something subtler in the interacting reference/observable definition.

---

### Priority 4: return to current response / superfluid density after the one-particle Matsubara Green is fixed

Because the interacting current-response benchmark still fails and depends on unequal-time structure, the cleanest route is:

1. fix the one-particle canonical time-ordered Green function first,
2. then rebuild the current-response path from that corrected unequal-time structure,
3. then repeat the ED comparison for
   - `Lambda_L`,
   - `Lambda_T`,
   - `rho_s`.

At present, the current-response mismatch is too large to interpret physically.

---

## 7. Practical status summary

### Validated

- attractive-`U` HS convention:
  - `useChargeHST = false`
  - `sys_type = ComplexF64`
- equal-time CE benchmark on `3x3`
- free-case tau-space unequal-time Green benchmark
- free-case current-response benchmark

### Not validated / still broken

- interacting Matsubara Green benchmark
- interacting current-current response
- interacting CE superfluid density

### Working conclusion

The project is **past the equal-time debugging stage** and **past the simplest free unequal-time check**, but **not yet past the interacting unequal-time/Matsubara stage**.

The next real technical milestone is:

> make the canonical Matsubara one-particle Green function correct in the free case, then re-test the interacting case.

---

## 8. Follow-up finding from 2026-05-28 debugging

The interacting single-particle unequal-time Green function has an additional
normalization issue that is invisible to equal-time fixed-sector measurements.

For the attractive-`U` spin-HS setup used by `CanEnsAFQMC`

```text
useChargeHST = false
sys_type = ComplexF64
```

the unnormalized local HS sum differs from the unshifted Hubbard interaction by
a sector-dependent scalar per time slice:

```text
HS_sum(Nup,Ndn) / physical_sum(Nup,Ndn)
    ∝ exp[Delta_tau * U * (Nup + Ndn) / 2].
```

This cancels for equal-time observables in a fixed `(Nup,Ndn)` sector, and it
also cancels for number-conserving current-current propagation. It does **not**
cancel for the single-particle addition/removal branches:

```text
addition branch, Nup -> Nup+1 over tau:  multiply raw HS estimator by exp(-U*tau/2)
removal branch,  Nup -> Nup-1 over tau:  multiply raw HS estimator by exp(+U*tau/2)
```

The helper

```text
scripts/interacting_qmc_ed/ce_unequal_time_current_helpers.jl
```

now contains `spin_hs_sector_normalization_ratio(...)`, and

```text
scripts/interacting_qmc_ed/benchmark_ce_green_matsubara_3x3.jl
```

uses the addition-branch correction for its tau-space `Ctau` output.

This fixes a major part of the interacting tau-space comparison, but it does
**not** by itself fix the `G(iω_n)` output. The current `matsubara_from_Cτ`
routine is still only a naive finite-interval transform of the addition branch.
The canonical resolvent/standard Matsubara object still needs a separate,
explicit addition+removal implementation before the frequency-space benchmark
should be trusted.

### Implemented canonical-recursion unequal-time branch

The helper file now also implements the direct canonical procedure:

```text
CanonicalUnequalTimeGreenCache(...)
canonical_unequal_time_greens(...)
canonical_unequal_time_local_greens(...)
```

For each HS configuration and spin sector it:

1. diagonalizes the full one-body propagator `F = B(beta,0)`,
2. computes elementary-symmetric canonical coefficient ratios,
3. builds the addition branch
   `Tr_N[Gamma(Y) c_i Gamma(X) c_j^dagger] / Z_N`,
4. builds the removal branch
   `Tr_N[Gamma(Y) c_j^dagger Gamma(X) c_i] / Z_N`,
5. optionally applies the spin-HS sector-normalization factor above.

The Matsubara/tau benchmark now uses this canonical branch code and writes:

```text
greens_tau0_qmc.tsv          # addition branch
greens_tau0_remove_qmc.tsv   # removal branch
```

The exact-sector unequal-time benchmark has also been switched to this canonical
implementation for the formula side, with `physical_normalization=false` so that
it compares against the raw HS-sector trace.

Quick checks:

```text
U=0, beta=10, dtau=0.1, slices 1,10,50:
canonical formula agrees with exact-sector trace to ~1e-9 or better.
```

Caveat: the present eigenbasis coefficient implementation is less stable very
close to `tau=beta` in the free benchmark than the older stabilized fugacity
projection formula.  This is a numerical-stability issue in the coefficient
backend, not a change in the canonical definition.  The first target should be
the physically relevant tau range and the ED addition/removal branch checks;
frequency-space `G(iω_n)` remains marked as not-yet-canonical because the script
still applies a naive transform to the addition branch only.

### Switched the check target from `G(iω_n)` to `G(r,τ)` / `G(k,τ)`

Following the decision to ignore the Matsubara transform for now, the tau-space
benchmark now writes translationally averaged real- and momentum-space Green
functions for the addition branch:

```text
G(r,τ) = (1/V) sum_j G[j+r,j;τ]
G(k,τ) = sum_r exp(-i k·r) G(r,τ)
```

New/updated outputs from
`scripts/interacting_qmc_ed/benchmark_ce_green_matsubara_3x3.jl`:

```text
greens_r_tau_add_qmc.tsv
greens_k_tau_add_qmc.tsv
greens_r_tau_remove_qmc.tsv      # diagnostic only
greens_k_tau_remove_qmc.tsv      # diagnostic only
greens_tau0_qmc.tsv              # local addition = G(r=0,τ)
greens_tau0_remove_qmc.tsv       # local removal diagnostic
```

The exact-sector algebra benchmark now also writes full addition-branch
comparisons:

```text
greens_r_tau_add_exactsector.tsv
greens_k_tau_add_exactsector.tsv
```

A physical ED tau-space reference was added to
`scripts/interacting_qmc_ed/ed_hubbard_ce_green_matsubara_3x3.py`; despite the
old filename, it now also writes:

```text
greens_r_tau_add_ed.tsv
greens_k_tau_add_ed.tsv
```

and `scripts/interacting_qmc_ed/compare_ce_green_tau_space_vs_ed_3x3.py` compares
QMC and ED `G(r,τ)`/`G(k,τ)` tables directly.

Checks run:

```text
Exact-sector, U=0, beta=10, dtau=0.1, slices 1,10,50:
  max |delta G(r,tau)| = 3.43e-9
  max |delta G(k,tau)| = 1.19e-8

Exact-sector, U=-5, beta=10, dtau=0.1, slices 1,10,50:
  max |delta G(r,tau)| = 5.42e-11
  max |delta G(k,tau)| = 1.91e-10

Physical free ED vs QMC, U=0, beta=10, dtau=0.1, slices 0..50:
  max |delta G(r,tau)| = 3.43e-9
  max |delta G(k,tau)| = 1.19e-8
```

Known caveat: the present eigenbasis coefficient backend is still not reliable
for the very last slices close to `τ=β` in the ill-conditioned free 3x3 check.
The `G(r,τ)`/`G(k,τ)` comparison is therefore clean through `τ<=β/2` in the
current implementation.  A later production-quality implementation should avoid
forming the dense long-time propagator explicitly near `τ≈β`.

### Full-spectrum physical ED benchmark implemented for 3x3 addition Green function

A full-state canonical ED mode has been added to
`scripts/interacting_qmc_ed/ed_hubbard_ce_green_matsubara_3x3.py`:

```text
--full-spectrum
```

For the up-spin addition branch it includes **all** eigenstates in exactly the
connected canonical sectors:

```text
sampled sector:  (Nup,Ndn)   = (4,4), dim = 15876
addition sector: (Nup+1,Ndn) = (5,4), dim = 15876
```

It does not sum over unrelated grand-canonical sectors.  The implementation uses
2D translation momentum blocks, with fermionic translation signs included, so the
full 15876-dimensional sectors are diagonalized as nine 1764-dimensional blocks.
It writes the same tau-space files as the QMC side:

```text
greens_r_tau_add_ed.tsv
greens_k_tau_add_ed.tsv
greens_tau0_ed.tsv
```

Validation:

```text
Full-spectrum U=0 ED vs free/factorized ED, beta=10, dtau=0.1:
  max |delta G(r,tau)| = 2.25e-10
  max |delta G(k,tau)| = 4.31e-10
```

Full `U=-5` ED reference generated at:

```text
results/interacting_qmc_ed/full_ed_3x3_um5_beta10_dtau01
```

with metadata:

```text
E0(4,4) = -27.053044819148102
E0(5,4) = -28.638529079057164
Z_shifted(4,4) = 1.001755076791757
```

A first physical CE-QMC comparison was run with 500 warmup sweeps and 500 measured
samples:

```text
results/interacting_qmc_ed/qmc_gktau_3x3_um5_beta10_dtau01_w500_m500
results/interacting_qmc_ed/compare_gktau_3x3_um5_w500_m500_final
```

Comparison over slices `0..50` (`tau <= 5`) gives large absolute deviations at
large tau because the single-particle estimator variance grows strongly, but the
ED curve is within the current QMC error bars:

```text
space  max_z_abs  rms_z_abs
G(r)   2.05       0.89
G(k)   2.26       0.92
```

The local trace comparison excluding the deterministic `tau=0` point also stays
within about two standard errors over the same tau window.  This is now a real
physical ED-vs-CE-QMC benchmark for `U=-5`, but production-quality statistics
would still require more samples or variance reduction for large `tau`.

### Simple MPI independent-chain driver added

A simple MPI wrapper has been implemented for the 3x3 canonical unequal-time
Green-function benchmark:

```text
scripts/interacting_qmc_ed/benchmark_ce_green_tau_space_mpi_3x3.jl
scripts/interacting_qmc_ed/combine_ce_green_tau_space_rank_outputs.py
```

The MPI model is deliberately simple:

```text
one MPI rank = one independent Markov chain
```

Each rank uses a rank-offset seed and writes cumulative outputs to:

```text
<output-dir>/ranks/rank_00000/
<output-dir>/ranks/rank_00001/
...
```

Rank-local files are rewritten after every completed measurement chunk, so if a
Slurm walltime/preemption happens only the current chunk is lost.  This is not a
full walker/RNG checkpoint yet, but it is a safe first checkpoint-like production
layout for independent-chain MPI.  Rank 0 combines completed rank summaries with
`combine_ce_green_tau_space_rank_outputs.py` into the root output directory.

CADES smoke test passed with two MPI ranks.

The obsolete single-chain CADES job was canceled before writing measurement
files.  A 32-rank replacement job was submitted:

```text
job id: 5390031
script: /home/9pm/nUHubbard/scripts/interacting_qmc_ed/job_ce_gktau_3x3_um5_beta10_dtau01_mpi32_w10000_m100000_cades.sbatch
output: /home/9pm/nUHubbard/runs/ce_gktau_3x3_um5_beta10_dtau01_MPI32_Ntherm10000_Nmeas100000_seed20260528
```

Settings:

```text
32 ranks, one chain per rank
warmup/rank = 10000
measurement chunks/rank = 25
chunk size/rank = 125
total measurements = 32 * 25 * 125 = 100000
```

Follow-up status after the CADES emails:

```text
5390034  ce_gktau3x3_mpi32_test1k   COMPLETED  00:00:56  ExitCode 0:0
5390035  ce_gktau3x3_mpi32_t1kpr    COMPLETED  00:02:07  ExitCode 0:0
5390031  ce_gktau3x3_mpi32_100k     COMPLETED  00:09:55  ExitCode 0:0
```

Important distinction:

```text
5390034: 32 ranks, 100 warmup/rank, about 1000 total measurements
5390035: 32 ranks, 100 warmup/rank, 1000 measurements/rank = 32000 total
5390031: 32 ranks, 10000 warmup/rank, 3125 measurements/rank = 100000 total
```

The QMC jobs themselves completed and wrote combined root-level tau-space files.
The original Slurm post-processing comparison failed only because the comparison
script used Python features/packages not available in CADES default `python3`.
`scripts/interacting_qmc_ed/compare_ce_green_tau_space_vs_ed_3x3.py` has now
been converted to a stdlib-only script and re-run on CADES.

Comparison against the full-spectrum `U=-5` ED reference over slices `0..50`
(`tau <= 5`) gives:

```text
Correct small test, job 5390035, 1000 measurements/rank:
space  nrows  max_abs_delta  rms_abs_delta  max_z_abs  rms_z_abs
G(r)   459    7.56e2         1.13e2         3.22       0.97
G(k)   459    3.37e3         3.38e2         4.99       1.05

Production test, job 5390031, 100000 total measurements:
space  nrows  max_abs_delta  rms_abs_delta  max_z_abs  rms_z_abs
G(r)   459    3.87e2         6.45e1         4.64       1.16
G(k)   459    1.59e3         1.94e2         8.54       1.32
```

If the deterministic/very-low-variance `tau=0` row is excluded (`slice=1..50`):

```text
Job 5390035:
G(r): max_z_abs = 2.40, rms_z_abs = 0.95
G(k): max_z_abs = 3.57, rms_z_abs = 1.00

Job 5390031:
G(r): max_z_abs = 2.97, rms_z_abs = 1.11
G(k): max_z_abs = 5.05, rms_z_abs = 1.19
```

Representative production rows:

```text
G(r=0,tau=0.1): QMC 0.5478951 ± 0.0002363, ED 0.5477966, z = 0.23
G(r=0,tau=1.0): QMC 0.8711009 ± 0.0047887, ED 0.8814427, z = 1.52
G(r=0,tau=5.0): QMC 188.6 ± 207.2 (Re), ED 298.5, z = 0.44

G(k=(2π/3,0),tau=0.1): QMC 0.3372300 ± 0.0014322, ED 0.3379375, z = 0.45
G(k=(2π/3,0),tau=1.0): QMC 1.1120766 ± 0.0165041, ED 1.0977579, z = 0.98
G(k=(2π/3,0),tau=5.0): QMC -423.9 ± 827.1 (Re), ED 666.3, z = 1.16
```

The production run is therefore operational, but the ED benchmark is not yet a
clean pass at high statistics.  The shrinking error bars expose systematic or
underestimated-error issues, especially in small-`k`, small-`tau` momentum-space
rows.  The next debugging step should be to audit the equal-time/tau-near-zero
`G(r)` normalization and improve the MPI error analysis with rank/chunk-level
blocking or jackknife before treating the quoted z-scores as final.

### BKT/superfluid-stiffness benchmark observables added

The active CE unequal-time benchmark now also measures the current-response
quantities needed for a BKT transition-temperature estimate.  The canonical QMC
output file is:

```text
bkt_observables_qmc.tsv
```

and contains:

```text
beta
temperature = 1/beta
bkt_universal_jump_2T_over_pi = 2/(pi*beta)
lambda_longitudinal_qmin0
lambda_transverse_0qmin
Kx_per_site
diamagnetic_minus_Kx_per_site
rho_s_current       = 0.25*(lambda_longitudinal_qmin0 - lambda_transverse_0qmin)
rho_s_diamagnetic  = 0.25*(-Kx_per_site - lambda_transverse_0qmin)
bkt_residual_current      = rho_s_current - 2/(pi*beta)
bkt_residual_diamagnetic = rho_s_diamagnetic - 2/(pi*beta)
```

The MPI independent-chain combiner now combines this file across ranks into the
root output directory.  The rank model is unchanged: one MPI rank is one
independent Markov chain.

For large per-rank statistics, the tau-space QMC driver has been changed from
storing every measured sample in memory to streaming cumulative sums and
sum-of-squares.  This keeps the memory footprint small enough for long
`100000`-measurements/rank runs while preserving the same per-batch cumulative
TSV checkpoint-like output.

The ED side now writes the matching file:

```text
bkt_observables_ed.tsv
```

from `scripts/interacting_qmc_ed/ed_hubbard_ce_green_matsubara_3x3.py`.  A new
`--bkt-only` mode computes the fixed-sector ED BKT quantities without redoing
the one-particle addition Green function.  The comparison helper is:

```text
scripts/interacting_qmc_ed/compare_ce_bkt_observables_3x3.py
```

Important ED finding: for current response/superfluid stiffness, a low-energy
truncated ED spectrum is not reliable even at `beta=10`, because the static
current-current integral has matrix elements to higher excited states.  The
full-spectrum translation-block ED BKT reference for the `3x3`, `(4,4)`,
`U=-5`, `beta=10` benchmark is now saved at:

```text
results/interacting_qmc_ed/full_ed_3x3_um5_beta10_dtau01_bkt/bkt_observables_ed.tsv
```

with:

```text
lambda_longitudinal_qmin0 = 0.6455739792339693
lambda_transverse_0qmin  = 0.2581884150463996
Kx_per_site              = -0.6455739792339751
rho_s_current            = 0.09684639104689242
rho_s_diamagnetic        = 0.09684639104689387
2T/pi                    = 0.06366197723675814
rho_s - 2T/pi            = 0.03318441381013429
```

This replaces the earlier low-energy ED current-response numbers as the proper
ED benchmark for BKT quantities.

A new CADES production job including both `G(r,tau)`/`G(k,tau)` and the BKT
observables was submitted:

```text
job id: 5390088
job name: ce_gktau_bkt3x3_mpi32_100kpr
script: /home/9pm/nUHubbard/scripts/interacting_qmc_ed/job_ce_gktau_bkt_3x3_um5_beta10_dtau01_mpi32_w10000_m100000perrank_cades.sbatch
output: /home/9pm/nUHubbard/runs/ce_gktau_bkt_3x3_um5_beta10_dtau01_MPI32_Ntherm10000_Nmeas100000perrank_seed20260528
```

Settings:

```text
32 ranks, one independent chain per rank
warmup/rank = 10000
measurements/rank = 100000
total measurements = 3200000
chunks/rank = 100
chunk size/rank = 1000
walltime = 24:00:00
```

The job script will run the tau-space ED comparison over `slice=1..50` and the
BKT ED comparison after the MPI QMC finishes and rank 0 combines the outputs.

Completion/results:

```text
job id 5390088: COMPLETED, elapsed = 10:58:57, ExitCode = 0:0
combined samples = 3200000
```

BKT comparison against full-spectrum ED:

```text
observable                      ED                 QMC ± stderr                  z
lambda_longitudinal_qmin0       0.645573979234     0.656874534993 ± 0.000175767  64.3
lambda_transverse_0qmin         0.258188415046     0.275480114338 ± 0.000876537  19.7
Kx_per_site                    -0.645573979234    -0.651991912370 ± 0.000174461  36.8
rho_s_current                   0.096846391047     0.095348605164 ± 0.000224774   6.66
rho_s_diamagnetic               0.096846391047     0.094127949508 ± 0.000224701  12.1
rho_s_current - 2T/pi           0.033184413810     0.031686627927 ± 0.000224774   6.66
```

Tau-space one-particle comparison against full-spectrum ED over `slice=1..50`:

```text
space  nrows  max_abs_delta  rms_abs_delta  max_z_abs  rms_z_abs
G(r)   450    72.07          10.42          11.76      2.47
G(k)   450    248.41         31.27          26.61      3.23
```

The high-statistics result is therefore **not within the nominal error bars**.
The absolute BKT/stiffness differences are small, but the errors are now tiny.
This most likely exposes systematic effects rather than insufficient sampling:
candidate causes are finite-`dtau=0.1` Trotter error in the QMC benchmark,
remaining unequal-time estimator bias at small tau, and/or underestimated
autocorrelation errors from simple rank/chunk combination.  A direct next check
is to repeat the same BKT benchmark at smaller `dtau` (e.g. `0.05` or `0.025`)
before interpreting the discrepancy as a physics/algorithm error.

The `dtau=0.05` check has been submitted on CADES:

```text
job id: 5391031
job name: ce_bkt3x3_mpi64_dt005_30k
script: /home/9pm/nUHubbard/scripts/interacting_qmc_ed/job_ce_gktau_bkt_3x3_um5_beta10_dtau005_mpi64_w5000_m30000perrank_cades.sbatch
output: /home/9pm/nUHubbard/runs/ce_gktau_bkt_3x3_um5_beta10_dtau005_MPI64_Ntherm5000_Nmeas30000perrank_seed20260529
```

Settings:

```text
2 nodes
64 ranks, one independent chain per rank
dtau = 0.05
warmup/rank = 5000
measurements/rank = 30000
total measurements = 1920000
chunks/rank = 30
chunk size/rank = 1000
walltime = 12:00:00
```

A matching full-spectrum ED `G(r,tau)`/`G(k,tau)` reference for `dtau=0.05` was
generated locally and synced to CADES:

```text
results/interacting_qmc_ed/full_ed_3x3_um5_beta10_dtau005
/home/9pm/nUHubbard/runs/full_ed_3x3_um5_beta10_dtau005
```

The job script will compare `G(r,tau)`/`G(k,tau)` over `slice=1..100` and will
compare BKT observables to the same full-spectrum ED stiffness reference used
above.

---

## 2026-05-29 transverse current-current correlator debug update

A targeted single-HS-configuration check exposed a real bug in the finite-`q`
current-current estimator used for the BKT quantities.  The old
`current_corr_pair_q` hand-expanded the Wick contraction bond-by-bond in real
space.  For the `3x3`, `U=-5`, spin-HS benchmark, this did **not** agree with
the general bilinear trace formula for either `q=(qmin,0)` or `q=(0,qmin)`, with
large per-slice deviations.  This directly affected
`lambda_transverse_0qmin` and therefore `rho_s`.

The estimator has been replaced by the matrix bilinear Wick contraction

```text
<J_q(tau) J_-q(0)> / V
  = [ <J_q(tau)> <J_-q(0)> - Tr(J_q G_tau0 J_-q G_0tau) ] / V
```

for same-spin terms, using the same `ce_current_operator_x` matrices as the ED
reference.  The opposite-spin disconnected term remains the product of one-body
current expectations.

Patched file:

```text
scripts/interacting_qmc_ed/ce_unequal_time_current_helpers.jl
```

A new diagnostic script was added:

```text
scripts/interacting_qmc_ed/validate_ce_current_trace_formula_3x3.jl
```

It compares the production estimator against the general trace formula for a
sampled `3x3` HS configuration.  After the patch it passes for both longitudinal
and transverse momenta; representative output:

```text
== longitudinal q=(2.09439510239, 0) ==
slice=  1 |diff|=9.54e-08
slice= 50 |diff|=8.16e-06
== transverse q=(0, 2.09439510239) ==
slice=  1 |diff|=8.53e-08
slice= 50 |diff|=8.73e-06
PASS: max |trace - estimator| = 3.37e-05
```

A small local post-patch current-only smoke run at `dtau=0.05` with 300 samples
is noisy but now has the full-spectrum ED stiffness within the statistical error:

```text
lambda_L        = 0.62664 ± 0.01744    ED 0.64557
lambda_T        = 0.21876 ± 0.06745    ED 0.25819
Kx/site         = -0.62547 ± 0.01741   ED -0.64557
rho_s_current  = 0.10197 ± 0.01772    ED 0.09685
rho_s_dia      = 0.10168 ± 0.01772    ED 0.09685
```

The earlier high-statistics CADES BKT comparisons from before this patch should
therefore be treated as invalid for current-response/stiffness observables,
although their one-particle `G(r,tau)`/`G(k,tau)` files remain useful.

The patched helper was synced to CADES at:

```text
/home/9pm/nUHubbard/scripts/interacting_qmc_ed/ce_unequal_time_current_helpers.jl
```

A small post-patch CADES smoke job was submitted to check the BKT observables:

```text
job id: 5391528
job name: ce_bkt3x3_dt005_patch1k
settings: dtau=0.05, 32 MPI ranks, 100 warmup/rank, 1000 measurements/rank
output: /home/9pm/nUHubbard/runs/ce_gktau_bkt_3x3_um5_beta10_dtau005_MPI32_PATCH_currentfix_Ntherm100_Nmeas1000perrank_seed20260529
comparison: /home/9pm/nUHubbard/runs/compare_ce_bkt_3x3_um5_dtau005_MPI32_PATCH_1000perrank_vs_full_ed.tsv
```

Partial result while job `5391528` was still running, after 2/10 batches per rank
(`6400` combined measurements): all BKT observables were already within about
`1.2 sigma` of full-spectrum ED,

```text
lambda_L       0.649988 ± 0.003919   ED 0.645574   z=1.13
lambda_T       0.254891 ± 0.008927   ED 0.258188   z=0.37
Kx/site       -0.648772 ± 0.003912   ED -0.645574  z=0.82
rho_s_current 0.098774 ± 0.002475   ED 0.096846   z=0.78
rho_s_dia     0.098470 ± 0.002474   ED 0.096846   z=0.66
```

This is strong evidence that the previous high-statistics transverse-current
mismatch came from the real-space bond-expanded current estimator, not from ED
or from insufficient Markov-chain statistics.

After 3/10 batches per rank (`9600` combined measurements), the same patched
CADES smoke job remained consistent with ED:

```text
lambda_L       0.649267 ± 0.003214   ED 0.645574   z=1.15
lambda_T       0.255831 ± 0.007399   ED 0.258188   z=0.32
Kx/site       -0.648052 ± 0.003208   ED -0.645574  z=0.77
rho_s_current 0.098359 ± 0.002055   ED 0.096846   z=0.74
rho_s_dia     0.098055 ± 0.002055   ED 0.096846   z=0.59
```

### Current-fix production-size reruns submitted

After the current estimator fix, two production-size reruns matching the earlier
benchmarks were prepared and submitted on CADES.  Both job scripts first run
`validate_ce_current_trace_formula_3x3.jl`; the startup logs show the validation
passed with `max |trace - estimator| = 2.141e-05` before production sampling.

#### `dtau=0.1`, same size as earlier 32-rank 100k/rank run

```text
job id: 5391551
job name: ce_bkt3x3_dt01_fix100k
script: /home/9pm/nUHubbard/scripts/interacting_qmc_ed/job_ce_gktau_bkt_3x3_um5_beta10_dtau01_mpi32_currentfix_w10000_m100000perrank_cades.sbatch
output: /home/9pm/nUHubbard/runs/ce_gktau_bkt_3x3_um5_beta10_dtau01_MPI32_CURRENTFIX_Ntherm10000_Nmeas100000perrank_seed20260530
comparison G: /home/9pm/nUHubbard/runs/compare_ce_gktau_bkt_3x3_um5_dtau01_MPI32_CURRENTFIX_100000perrank_vs_full_ed
comparison BKT: /home/9pm/nUHubbard/runs/compare_ce_bkt_3x3_um5_dtau01_MPI32_CURRENTFIX_100000perrank_vs_full_ed.tsv
settings: 32 MPI ranks, 10000 warmup/rank, 100000 measurements/rank, 100 chunks/rank, walltime 16h
```

#### `dtau=0.05`, same size as earlier 64-rank 30k/rank run

```text
job id: 5391552
job name: ce_bkt3x3_dt005_fix30k
script: /home/9pm/nUHubbard/scripts/interacting_qmc_ed/job_ce_gktau_bkt_3x3_um5_beta10_dtau005_mpi64_currentfix_w5000_m30000perrank_cades.sbatch
output: /home/9pm/nUHubbard/runs/ce_gktau_bkt_3x3_um5_beta10_dtau005_MPI64_CURRENTFIX_Ntherm5000_Nmeas30000perrank_seed20260530
comparison G: /home/9pm/nUHubbard/runs/compare_ce_gktau_bkt_3x3_um5_dtau005_MPI64_CURRENTFIX_30000perrank_vs_full_ed
comparison BKT: /home/9pm/nUHubbard/runs/compare_ce_bkt_3x3_um5_dtau005_MPI64_CURRENTFIX_30000perrank_vs_full_ed.tsv
settings: 64 MPI ranks, 5000 warmup/rank, 30000 measurements/rank, 30 chunks/rank, walltime 10h
```

### Current-fix production-size reruns completed and benchmarked

The two current-fix CADES reruns completed successfully.

```text
5391551 ce_bkt3x3_dt01_fix100k  COMPLETED  ExitCode=0:0  elapsed=10:08:29
5391552 ce_bkt3x3_dt005_fix30k  COMPLETED  ExitCode=0:0  elapsed=06:10:31
```

#### BKT comparison vs full-spectrum ED

`dtau=0.1`, 32 ranks, 100000 measurements/rank:

```text
lambda_L       ED 0.645573979234   QMC 0.656807632509 ± 0.000175549   z=63.99
lambda_T       ED 0.258188415046   QMC 0.265150245020 ± 0.000543035   z=12.82
Kx/site        ED -0.645573979234  QMC -0.651925509738 ± 0.000174244  z=36.45
rho_s_current ED 0.096846391047   QMC 0.097914346872 ± 0.000145645   z=7.33
rho_s_dia     ED 0.096846391047   QMC 0.096693816179 ± 0.000145525   z=1.05
```

`dtau=0.05`, 64 ranks, 30000 measurements/rank:

```text
lambda_L       ED 0.645573979234   QMC 0.648448561076 ± 0.000226575   z=12.69
lambda_T       ED 0.258188415046   QMC 0.259197202547 ± 0.000725072   z=1.39
Kx/site        ED -0.645573979234  QMC -0.647235449472 ± 0.000226151  z=7.35
rho_s_current ED 0.096846391047   QMC 0.097312839632 ± 0.000193175   z=2.41
rho_s_dia     ED 0.096846391047   QMC 0.097009561731 ± 0.000193138   z=0.84
```

The current estimator fix greatly improved the transverse response.  At
`dtau=0.05`, `lambda_T` is now consistent with ED within `1.4 sigma`, and the
diamagnetic stiffness is within `0.85 sigma`.  Remaining significant deviations
are mainly in `lambda_L` and `Kx/site`, with the expected reduction when going
from `dtau=0.1` to `dtau=0.05`.

#### One-particle tau-space comparison summaries

```text
# dtau=0.1, slice 1..50
G(r): max_z=11.586, rms_z=2.417
G(k): max_z=27.182, rms_z=3.183

# dtau=0.05, slice 1..100
G(r): max_z=5.459, rms_z=1.007
G(k): max_z=7.194, rms_z=1.118
```

As expected, the current-estimator patch does not change the one-particle
Green-function estimator; the `dtau=0.05` one-particle benchmark remains much
better than `dtau=0.1` but still has small high-statistics finite-`dtau`/small-τ
deviations.

### Extra bug-screen after current-fix production reruns

A follow-up screen was performed after the production reruns to check for more
obvious bugs.

1. **Full fixed-sector exact-trace check added.**  A new deterministic validator
   constructs a small `3x3`, `(4,4)`, spin-HS configuration with only four
   Trotter slices, builds the exact fixed-`N` many-body propagators, and compares
   the physical real part of the full two-spin current response against the
   production canonical estimator, including same-spin and opposite-spin
   disconnected pieces:

   ```text
   scripts/interacting_qmc_ed/validate_ce_current_full_exact_smallL_3x3.jl
   ```

   Local result:

   ```text
   L real_diff = 2.887e-15
   T real_diff = 3.775e-15
   PASS
   ```

   CADES result after syncing:

   ```text
   L real_diff = 4.441e-16
   T real_diff = 1.388e-15
   PASS
   ```

   This is a stronger check than the single-particle trace-formula screen: it
   verifies the full fixed-sector CE projection and the cross-spin contribution
   for both longitudinal and transverse responses.

2. **Rank-blocked error estimate check.**  The BKT errors from the MPI combiner
   were compared with a conservative rank-mean standard error.  Ratios
   `SE_rank / SE_combined` were close to one:

   ```text
   dtau=0.1: ratios 0.96--1.04 for lambda_L, lambda_T, Kx, rho_s
   dtau=0.05: ratios 1.09--1.19 for lambda_L, lambda_T, Kx, rho_s
   ```

   So the quoted errors are not obviously underestimated by a large factor from
   rank-to-rank autocorrelation effects.  The `dtau=0.05` rank-blocked errors are
   only about 10--20% larger than the pooled combiner errors.

3. **Latent rectangular-lattice q-vector issue fixed.**  The benchmark/reference
   BKT code used `2*pi/Lx` for both `q=(qmin,0)` and `q=(0,qmin)`.  This has no
   effect for all current square-lattice benchmarks (`3x3`, `Lx=Ly`), but it
   would be wrong for future rectangular lattices.  The active QMC/ED scripts now
   use

   ```text
   qx_min = 2*pi/Lx
   qy_min = 2*pi/Ly
   ```

   for longitudinal and transverse current responses, respectively.  Updated
   files include the active tau-space QMC benchmark and ED reference scripts.

No additional physics-affecting bug was found for the current square-lattice
benchmark.  The remaining high-statistics deviations behave consistently with
finite-`dtau` Trotter error: in particular, `lambda_L` and `Kx` deviations shrink
strongly from `dtau=0.1` to `dtau=0.05`, and the diamagnetic stiffness is already
within error at `dtau=0.05`.

### BKT-only/equal-time-only mode added

The active CE tau-space benchmark now supports skipping one-particle unequal-time
Green-function measurements while still measuring all observables needed for a
BKT `Tc` scan.  New/active flags:

```text
--measure-greens=false       # skip G(tau), G(r,tau), G(k,tau) accumulation/output
--measure-bkt=true           # keep current-current response and rho_s outputs
--measure-equal-time=true    # write equal-time thermodynamic/diamagnetic observables
```

When `--measure-greens=false`, the driver no longer writes the one-particle
files

```text
greens_tau0_qmc.tsv
greens_tau0_remove_qmc.tsv
greens_r_tau_add_qmc.tsv
greens_r_tau_remove_qmc.tsv
greens_k_tau_add_qmc.tsv
greens_k_tau_remove_qmc.tsv
```

and only writes, for this purpose,

```text
equal_time_observables_qmc.tsv
bkt_observables_qmc.tsv
metadata.toml
```

`equal_time_observables_qmc.tsv` contains

```text
kinetic_per_site
interaction_per_site
total_per_site
double_occupancy_per_site
Kx_per_site
diamagnetic_minus_Kx_per_site
```

plus fixed-sector density and run metadata.  `bkt_observables_qmc.tsv` contains
the unequal-time current-response quantities needed for BKT:

```text
lambda_longitudinal_qmin0
lambda_transverse_0qmin
rho_s_current = 0.25*(lambda_L - lambda_T)
rho_s_diamagnetic = 0.25*(-Kx_per_site - lambda_T)
rho_s - 2T/pi
```

The MPI combiner now combines `equal_time_observables_qmc.tsv` across ranks and
skips absent Green-function files cleanly.  Local and CADES smoke tests with
`--measure-greens=false` produced only the expected equal-time/BKT files.

### CE-QMC checkpoint/restart setup for CADES

Checkpoint/restart has now been added to the CE tau-space driver in the same
spirit as the DQMC CADES wrappers.  The checkpoint is per independent MPI rank
(one rank = one Markov chain), because the MPI wrapper launches independent
chains with rank-dependent seeds.

Driver-side checkpoint flags:

```text
--checkpoint-enable=true
--checkpoint-file=checkpoint.jls
--checkpoint-every-batches=1
--checkpoint-freq-hours=0.5
--runtime-limit-hours=<walltime minus safety margin>
--checkpoint-exit-code=13
--checkpoint-keep=true        # useful for MPI wrappers until wrapper sees success
```

Each rank writes, under its rank output directory,

```text
checkpoint.jls          # Julia Serialization payload
checkpoint.jls.status   # small text sidecar with completed_batches/nsamples
```

The serialized payload contains the current `Walker`, Julia RNG state, completed
batch count, total sample count, and all measurement accumulators.  On restart,
the driver restores the walker/RNG/accumulators and skips warmup.  A local
interruption/resume smoke test produced byte-identical `equal_time_observables`
against an uninterrupted two-batch control run.

The MPI wrapper now also passes

```text
--checkpoint-world-size=<nranks>
--checkpoint-root-dir=<absolute root output dir>
```

to each rank.  When a runtime-limit checkpoint stop is reached, a rank writes
its checkpoint and waits for the other rank sidecars to reach the same completed
batch before exiting.  This avoids the common `mpiexecjl` behavior where the
first nonzero rank exit can kill slower ranks before they have checkpointed.

CADES job scripts added:

```text
scripts/interacting_qmc_ed/job_ce_bkt_only_checkpoint_smoke_3x3_cades.sbatch
scripts/interacting_qmc_ed/job_ce_bkt_only_3x3_um5_beta10_dtau01_mpi32_checkpoint_cades.sbatch
```

The smoke script intentionally stops through the checkpoint path, auto-resubmits
with the DQMC-style `RESUBMIT_COUNT=... sbatch script` pattern (no `--export`),
then resumes and completes/combines.  The production template runs the faster
BKT-only/equal-time-only measurement mode:

```text
--measure-greens=false
--measure-bkt=true
--measure-equal-time=true
```

with 32 ranks, 10000 warmups/rank, and 100000 measurements/rank, checkpointing
roughly every 0.5 hours and stopping at 3.75 hours for a 4-hour Slurm walltime.

### 12x12 performance screen after checkpoint fix

The first 12x12 CADES production attempt (`5391996`) timed out before any
checkpoint because checkpointing occurred only after all warmup sweeps.  Warmup
checkpoint/resume was added, but the resulting patched run exposed a separate
performance problem: the default `cluster_size=3` proposal took order minutes
per Markov sweep at `L=12`, `beta=6`, `Nup=Ndn=36`.

CADES one-rank timing screen for one warmup sweep (`V=144`, `L_tau=60`, exact
Metropolis ratio, `forceSymmetry=true`) showed:

```text
cluster_size=3     180.9 s / sweep
cluster_size=12     45.3 s / sweep
cluster_size=36     17.4 s / sweep
cluster_size=144     7.4 s / sweep
cluster_size=144 + lowrank   7.2 s / sweep
```

So exposing and using large cluster proposals is a ~25x warmup/sweep speedup,
but it does not solve the full BKT run.

The unequal-time current/BKT measurement itself is the dominant bottleneck.  For
one `cluster_size=144` sample with the current time-sliced canonical estimator
and the old 3x3-style `num_fourier_points=10`, CADES timing was approximately:

```text
sweep                    7.0 s
canonical density update 2.7 s
B-slice/prefix build     4.1 s
lambda_L current       117.5 s
lambda_T current        89.0 s
total BKT measurement  213.2 s per measured configuration
```

The sparse-current contraction optimization replaced dense `tr(J*G*J*G)` matrix
products by loops over the `2V` current-operator bonds and passes the existing
3x3 single-HS trace validator.  This reduced one BKT measurement from about
251 s to about 213 s at `num_fourier_points=10`, showing that the remaining cost
is dominated by the stable canonical inverse/Fourier loop, not by the current
matrix contraction.

Important correctness note: `num_fourier_points=10` was appropriate for the
3x3 benchmark (`V+1=10`) but is not an exact canonical Fourier projection for a
12x12 lattice.  A correct Fourier projection for `V=144` would require roughly
`V+1=145` points, which would make the current time-sliced BKT estimator even
slower.  The fast built-in static `measure_CurrentResponse` shortcut was tested
against the exact fixed-sector time-sliced current trace on the small 3x3 check
and differs by about `0.5` in the tested configuration, so it cannot replace the
unequal-time BKT estimator.

Conclusion: the current CE-QMC code can be made substantially faster for sweeps
by using large cluster proposals, but the present time-sliced canonical BKT
measurement is not yet suitable for 12x12 production.  A new optimized canonical
current-response estimator (or a different CE-QMC update/measurement algorithm)
is needed before requesting long 12x12 BKT runs.

### Fast exact canonical time-sliced current-response estimator (2026-05-31)

The current-response bottleneck has been addressed at the algorithm level in
`scripts/interacting_qmc_ed/ce_unequal_time_current_helpers.jl`.  The new path
uses the fixed-sector identity

```text
Tr_N[ Γ(V_l) J(q) Γ(U_l) J(-q) ]
  = Tr_N[ Γ(F) (Γ(U_l)^-1 J(q) Γ(U_l)) J(-q) ],    F = V_l U_l,
```

so each imaginary-time slice is measured as a canonical two-bilinear expectation
in the eigenbasis of the full one-body propagator, rather than by looping over
canonical Fourier phases and rebuilding stable inverse Green matrices for every
slice.  The estimator also keeps the sparse current-bond contraction introduced
above, and applies the sparse current operator directly when forming the
transformed bilinear matrices.

Because the existing `CanEnsAFQMC.second_order_corr` was not reliable for the
degenerate eigenvalues that appear in the small exact checks, this path computes
canonical occupations and pair occupations directly from elementary-symmetric
polynomial recursions (`canonical_occ_paircorr_direct`).

Validation after the change:

```text
validate_ce_current_full_exact_smallL_3x3.jl:
  L real_diff = 2.220e-15
  T real_diff = 3.164e-15
  PASS: full fixed-sector exact trace agrees with estimator for physical real parts.

validate_ce_current_trace_formula_3x3.jl:
  PASS: max |trace - estimator| = 3.367e-05
```

CADES one-rank 12x12 timing after syncing this estimator, with `cluster_size=144`,
`V=144`, `L_tau=60`, `Nup=Ndn=36`, `beta=6`:

```text
# num_fourier_points=10 timing script
sweep                    7.02 s
canonical density update 2.65 s
B-slice/prefix build     3.98 s
lambda_L current         3.62 s
lambda_T current         1.55 s
total BKT measurement   11.79 s per measured configuration

# num_fourier_points=145 timing script
sweep                    6.77 s
canonical density update 3.27 s
B-slice/prefix build     4.01 s
lambda_L current         3.48 s
lambda_T current         1.35 s
total BKT measurement   12.10 s per measured configuration
```

Thus the unequal-time current-response part is no longer the 12x12 bottleneck:
`lambda_L + lambda_T` fell from roughly `206 s` per sample to roughly `5 s` per
sample, and total measured-configuration cost fell from roughly `213 s` to
roughly `12 s` for the timing script.  The remaining major cost is now the QMC
sweep/update itself and the prefix/suffix construction, not the current-response
algorithm.

### Batched BKT-momentum current-response optimization (2026-06-01)

The first fast current estimator removed the canonical Fourier loop, but the BKT
measurement still evaluated the longitudinal and transverse momenta as two
separate calls.  Each call rebuilt the same dense prefix product in the full
propagator eigenbasis and refactorized the same slice matrix for a given spin.

`measure_bkt_observables` now calls `measure_current_responses_unequaltime` with
both BKT momenta `[(2π/Lx,0),(0,2π/Ly)]` at once.  For each spin and imaginary-
time slice this reuses:

- the canonical occupations/pair occupations,
- the dense transformed prefix product `Q = U_l P`, and
- one LU factorization of `Q` for both current momenta.

The bilinear-product weights that depend only on `J(-q)` and the fixed-sector
occupations are also precomputed once per momentum, so each slice only contracts
the transformed `J(q)` matrix with a prebuilt weight matrix.

CADES timing for one 12x12 measured configuration (`cluster_size=144`,
`num_fourier_points=145`, same random seed as previous timing):

```text
sweep                         7.01 s
density update                3.29 s
B-slice/prefix build          3.96 s
lambda_L single-call          2.81 s
lambda_T single-call          1.18 s
lambda_L/T batched            1.52 s
total measure, single calls  11.24 s
total measure, batched        8.78 s
```

This is a further `~22%` reduction in total BKT measured-configuration time and
a `~2.6x` reduction in the current-response part relative to two separate fast
calls.  Relative to the original Fourier/inverse implementation, the full
`lambda_L + lambda_T` current-response cost has fallen from roughly `206 s` to
roughly `1.5 s` for this timing case.

### 3x3 ED-benchmark big reruns after batched BKT-current optimization (2026-06-01)

After the batched fixed-`N` current-response optimization was synced to CADES,
the two previous production-size 3x3 ED-benchmark jobs were cloned with new
output directories and submitted again.  These jobs still measure the one-particle
`G(r,τ)`/`G(k,τ)` benchmark plus equal-time/BKT observables; only the BKT/current
measurement path has changed.

```text
dtau = 0.1 job id: 5393117
job name: ce_bkt3x3_d01_bfast100k
script: /home/9pm/nUHubbard/scripts/interacting_qmc_ed/job_ce_gktau_bkt_3x3_um5_beta10_dtau01_mpi32_batchfast_20260601_cades.sbatch
output: /home/9pm/nUHubbard/runs/ce_gktau_bkt_3x3_um5_beta10_dtau01_MPI32_BATCHFAST_Ntherm10000_Nmeas100000perrank_seed20260601
comparison G: /home/9pm/nUHubbard/runs/compare_ce_gktau_bkt_3x3_um5_dtau01_MPI32_BATCHFAST_100000perrank_vs_full_ed
comparison BKT: /home/9pm/nUHubbard/runs/compare_ce_bkt_3x3_um5_dtau01_MPI32_BATCHFAST_100000perrank_vs_full_ed.tsv
settings: 32 MPI ranks, 10000 warmup/rank, 100000 measurements/rank, walltime 16h

dtau = 0.05 job id: 5393118
job name: ce_bkt3x3_d005_bfast30k
script: /home/9pm/nUHubbard/scripts/interacting_qmc_ed/job_ce_gktau_bkt_3x3_um5_beta10_dtau005_mpi64_batchfast_20260601_cades.sbatch
output: /home/9pm/nUHubbard/runs/ce_gktau_bkt_3x3_um5_beta10_dtau005_MPI64_BATCHFAST_Ntherm5000_Nmeas30000perrank_seed20260601
comparison G: /home/9pm/nUHubbard/runs/compare_ce_gktau_bkt_3x3_um5_dtau005_MPI64_BATCHFAST_30000perrank_vs_full_ed
comparison BKT: /home/9pm/nUHubbard/runs/compare_ce_bkt_3x3_um5_dtau005_MPI64_BATCHFAST_30000perrank_vs_full_ed.tsv
settings: 64 MPI ranks, 5000 warmup/rank, 30000 measurements/rank, walltime 10h
```

Initial Slurm state after submission: both jobs were pending with reason
`Priority`.

### Pre-batched-current-response QMC result archive (2026-06-01)

Before using the new `BATCHFAST` 3x3 reruns for timing/benchmark comparisons,
the previous production-size `CURRENTFIX` QMC outputs were copied into an explicit
CADES archive:

```text
/home/9pm/nUHubbard/runs/archive_pre_batchfast_currentfix_3x3_20260601
```

The archive contains the original QMC output directories, G/BKT comparison
outputs, the two Slurm scripts used for the pre-batched jobs, Slurm logs when
available, a `README.txt`, `file_listing.txt`, and `sha256_selected.txt`.

Archived pre-batched result roots:

```text
runs/ce_gktau_bkt_3x3_um5_beta10_dtau01_MPI32_CURRENTFIX_Ntherm10000_Nmeas100000perrank_seed20260530
runs/compare_ce_gktau_bkt_3x3_um5_dtau01_MPI32_CURRENTFIX_100000perrank_vs_full_ed
runs/compare_ce_bkt_3x3_um5_dtau01_MPI32_CURRENTFIX_100000perrank_vs_full_ed.tsv
runs/ce_gktau_bkt_3x3_um5_beta10_dtau005_MPI64_CURRENTFIX_Ntherm5000_Nmeas30000perrank_seed20260530
runs/compare_ce_gktau_bkt_3x3_um5_dtau005_MPI64_CURRENTFIX_30000perrank_vs_full_ed
runs/compare_ce_bkt_3x3_um5_dtau005_MPI64_CURRENTFIX_30000perrank_vs_full_ed.tsv
```

The new `BATCHFAST` reruns use separate output roots with `seed20260601`, so they
will not overwrite the archived pre-batched results.

### BATCHFAST jobs canceled; stable batched projection reruns submitted (2026-06-01)

The first post-fast-estimator `BATCHFAST` reruns were started as jobs `5393117`
and `5393118`, but their rank-1 batch logs immediately showed unphysical BKT
stiffness values of order `1e5--1e7`.  They were canceled after about eight
minutes to avoid wasting CADES allocation:

```text
5393117 ce_bkt3x3_d01_bfast100k    CANCELLED  elapsed=00:07:52
5393118 ce_bkt3x3_d005_bfast30k    CANCELLED  elapsed=00:07:52
```

A local diagnostic showed the eigenbasis fast estimator can become numerically
unstable at the beta=10 3x3 benchmark point: for one sampled configuration the
stable canonical Fourier-projection estimator gave `lambda_L≈0.357` and
`lambda_T≈0.467`, while the eigenbasis fast path gave values of order `1e3`.
The exact-trace beta=1 validators still pass, so this is a low-temperature
numerical-stability problem rather than a Markov-chain ensemble change.

The production BKT measurement path was therefore switched back to the stable
canonical Fourier-projection estimator, but batched over the two BKT momenta so
that each `(Fourier point, imaginary-time slice)` Green-function construction is
reused for both `q=(2π/Lx,0)` and `q=(0,2π/Ly)`.  A local beta=10 smoke test
with 10 samples gave reasonable values:

```text
rho_s_current = 0.0845
rho_s_dia     = 0.0834
```

New stable-batched reruns were submitted:

```text
dtau = 0.1 job id: 5393119
job name: ce_bkt3x3_d01_stab100k
script: /home/9pm/nUHubbard/scripts/interacting_qmc_ed/job_ce_gktau_bkt_3x3_um5_beta10_dtau01_mpi32_stablebatch_20260601_cades.sbatch
output: /home/9pm/nUHubbard/runs/ce_gktau_bkt_3x3_um5_beta10_dtau01_MPI32_STABLEBATCH_Ntherm10000_Nmeas100000perrank_seed20260601
comparison G: /home/9pm/nUHubbard/runs/compare_ce_gktau_bkt_3x3_um5_dtau01_MPI32_STABLEBATCH_100000perrank_vs_full_ed
comparison BKT: /home/9pm/nUHubbard/runs/compare_ce_bkt_3x3_um5_dtau01_MPI32_STABLEBATCH_100000perrank_vs_full_ed.tsv
settings: 32 MPI ranks, 10000 warmup/rank, 100000 measurements/rank, walltime 16h

dtau = 0.05 job id: 5393120
job name: ce_bkt3x3_d005_stab30k
script: /home/9pm/nUHubbard/scripts/interacting_qmc_ed/job_ce_gktau_bkt_3x3_um5_beta10_dtau005_mpi64_stablebatch_20260601_cades.sbatch
output: /home/9pm/nUHubbard/runs/ce_gktau_bkt_3x3_um5_beta10_dtau005_MPI64_STABLEBATCH_Ntherm5000_Nmeas30000perrank_seed20260601
comparison G: /home/9pm/nUHubbard/runs/compare_ce_gktau_bkt_3x3_um5_dtau005_MPI64_STABLEBATCH_30000perrank_vs_full_ed
comparison BKT: /home/9pm/nUHubbard/runs/compare_ce_bkt_3x3_um5_dtau005_MPI64_STABLEBATCH_30000perrank_vs_full_ed.tsv
settings: 64 MPI ranks, 5000 warmup/rank, 30000 measurements/rank, walltime 10h
```

For runtime comparison, the pre-batched stable `CURRENTFIX` jobs completed in:

```text
5391551 ce_bkt3x3_dt01_fix100k  COMPLETED  elapsed=10:08:29
5391552 ce_bkt3x3_dt005_fix30k  COMPLETED  elapsed=06:10:31
```

### Stable reruns canceled; propagated stable estimator sanity checks (2026-06-01)

After deciding that rerunning the slow stable-projection path was not useful,
the submitted stable-batched jobs were canceled before production work:

```text
5393119 ce_bkt3x3_d01_stab100k   CANCELLED+ elapsed=00:15:58
5393120 ce_bkt3x3_d005_stab30k   CANCELLED+ elapsed=00:15:58
```

The current production candidate is now a `propagated` canonical
Fourier-projection estimator for the BKT current response.  It keeps the
canonical Fourier projection, so it is still a canonical-ensemble estimator, but
it avoids recomputing the full stable displaced Green functions at every
imaginary-time slice.  For each Fourier point it computes the exact stable
slice-0 Green functions, propagates the displaced Green functions with the
single-slice `B_l` and `B_l^{-1}` matrices, and refreshes from the stable LDR
formula every `--bkt-refresh-interval` slices.  The intended production setting
is:

```text
--measure-greens=false
--measure-bkt=true
--measure-equal-time=true
--bkt-current-estimator=propagated
--bkt-refresh-interval=10
--bkt-adaptive-refresh=true
--bkt-refresh-tol=1e-6
--bkt-refresh-min=1
--bkt-refresh-max=20
```

Sanity checks completed locally:

```text
validate_ce_current_trace_formula_3x3.jl: PASS, max |trace - estimator| = 3.367e-05
validate_ce_current_full_exact_smallL_3x3.jl: PASS, real_diff <= 4.052e-15
validate_ce_current_propagated_3x3.jl:
  refresh=1  agrees exactly with stable projection for the tested beta=10 sample
  refresh=10 max projected-vs-propagated real-part difference = 4.369e-10
  refresh=10 estimator-kernel timing: projected=0.040s propagated=0.009s, speedup=4.40x
```

The same propagated-vs-projected validator was synced to CADES and passed there:

```text
refresh=1  max projected-vs-propagated real-part difference = 0.000e+00
refresh=10 max projected-vs-propagated real-part difference = 1.112e-09
refresh=10 estimator-kernel timing: projected=0.080s propagated=0.015s, speedup=5.26x
```

A CADES login-node BKT-only CLI smoke test also completed:

```text
warmup_progress warmups_completed=10/10
batch=1 nsamples=10 total/site=-3.1642043224569862 docc/site=0.35302346864298234 rho_s_current=0.084515154708548 rho_s_dia=0.08344045058714082
```

An MPI passthrough smoke test with two independent ranks and
`--measure-greens=false` completed as well, confirming that the MPI wrapper
accepts and forwards the propagated-estimator options.  These are smoke checks
only; no new large 3x3 rerun has been submitted yet after this optimization.

### Adaptive refresh for propagated current estimator (2026-06-01)

The propagated BKT current estimator now also supports a SmoQyDQMC-like adaptive
refresh mode.  This is still a canonical Fourier-projection estimator; the
adaptive part only chooses how frequently to fall back to the stable LDR
displaced-Green reconstruction while measuring the current response.

CLI options:

```text
--bkt-adaptive-refresh=true
--bkt-refresh-tol=1e-6
--bkt-refresh-min=1
--bkt-refresh-max=20
--bkt-refresh-growth-patience=3
```

The recommended lower bound is `--bkt-refresh-min=1`, so that the adaptive
logic can always fall back to measuring a rejected segment endpoint from the
stable LDR reconstruction without using any unstable propagated intermediate
slice.

The CE `--bkt-refresh-tol` is the direct analogue of the SmoQyDQMC `δG_max`
threshold.  The SmoQyDQMC scripts in this repo use `δG_max=1e-6` with
`n_stab=10`; the CE default has therefore been set to
`--bkt-refresh-tol=1e-6`.

Algorithm:

1. Start from an exactly refreshed stable slice.
2. Try a candidate propagation segment of length `h`.
3. At the candidate endpoint, compute the stable LDR displaced Green functions
   and compare the propagated endpoint to the stable endpoint using the maximum
   absolute matrix-element error over `G(τ,τ)`, `G(τ,0)`, and `G(0,τ)`.
4. If the error exceeds `--bkt-refresh-tol`, reject that segment, halve `h`,
   and retry from the previous stable slice.  No rejected propagated slice is
   accumulated into the observable.
5. If the segment is accepted, accumulate propagated intermediate slices and
   measure the segment endpoint with the stable refreshed Green functions.
6. After several very safe segments, increase the attempted interval slowly up
   to `--bkt-refresh-max`; after failed segments, decrease it down to
   `--bkt-refresh-min`.

This is deliberately rollback-safe: if a candidate segment fails the stability
test, the code replays the segment at a shorter interval before adding those
slice contributions to the current-current integral.

Local checks after adding adaptive mode:

```text
fixed    refresh=1  max diff vs stable projection = 0.000e+00
fixed    refresh=10 max diff vs stable projection = 4.369e-10
adaptive refresh=10 max diff vs stable projection = 4.368e-10
adaptive timing on validator: projected=0.039s propagated=0.010s speedup=3.78x

adaptive CLI smoke:
rho_s_current = 0.08451515225431375
rho_s_dia     = 0.08344044995056438
```

CADES checks after syncing the code:

```text
fixed    refresh=1  max diff vs stable projection = 0.000e+00
fixed    refresh=10 max diff vs stable projection = 1.112e-09
adaptive refresh=10 max diff vs stable projection = 1.111e-09
adaptive timing on validator: projected=0.080s propagated=0.020s speedup=3.99x

adaptive CLI smoke:
rho_s_current = 0.08451515262458043
rho_s_dia     = 0.08344045024538552
```

The two-rank CADES MPI smoke test also passed with
`--bkt-adaptive-refresh=true`, confirming that the MPI wrapper forwards the new
options.  No large rerun was submitted after this adaptive-refresh update.

### Adaptive BKT-only 3x3 production reruns submitted (2026-06-01)

After aligning the CE adaptive refresh tolerance with the SmoQyDQMC
`δG_max=1e-6` convention, two BKT/equal-time-only 3x3 CE-QMC reruns were
submitted on CADES.  These runs intentionally skip unequal-time one-particle
Green-function measurement:

```text
common current estimator:
  --measure-greens=false
  --measure-bkt=true
  --measure-equal-time=true
  --bkt-current-estimator=propagated
  --bkt-adaptive-refresh=true
  --bkt-refresh-interval=10
  --bkt-refresh-tol=1e-6
  --bkt-refresh-min=1
  --bkt-refresh-max=20
  --bkt-refresh-growth-patience=3
```

Submitted jobs:

```text
dtau = 0.1 job id: 5393324
job name: ce_bkt3x3_d01_ad100k
script: /home/9pm/nUHubbard/scripts/interacting_qmc_ed/job_ce_bkt_3x3_um5_beta10_dtau01_mpi32_adaptive_bktonly_20260601_cades.sbatch
output: /home/9pm/nUHubbard/runs/ce_bkt_3x3_um5_beta10_dtau01_MPI32_ADAPTIVE_BKTONLY_Ntherm10000_Nmeas100000perrank_seed20260601
comparison BKT: /home/9pm/nUHubbard/runs/compare_ce_bkt_3x3_um5_dtau01_MPI32_ADAPTIVE_BKTONLY_100000perrank_vs_full_ed.tsv
settings: 32 MPI ranks, 10000 warmup/rank, 100000 measurements/rank, walltime 16h

dtau = 0.05 job id: 5393325
job name: ce_bkt3x3_d005_ad30k
script: /home/9pm/nUHubbard/scripts/interacting_qmc_ed/job_ce_bkt_3x3_um5_beta10_dtau005_mpi64_adaptive_bktonly_20260601_cades.sbatch
output: /home/9pm/nUHubbard/runs/ce_bkt_3x3_um5_beta10_dtau005_MPI64_ADAPTIVE_BKTONLY_Ntherm5000_Nmeas30000perrank_seed20260601
comparison BKT: /home/9pm/nUHubbard/runs/compare_ce_bkt_3x3_um5_dtau005_MPI64_ADAPTIVE_BKTONLY_30000perrank_vs_full_ed.tsv
settings: 64 MPI ranks, 5000 warmup/rank, 30000 measurements/rank, walltime 10h
```

Initial Slurm state immediately after submission:

```text
5393324 PENDING ce_bkt3x3_d01_ad100k  (Priority)
5393325 PENDING ce_bkt3x3_d005_ad30k  (Priority)
```
