# 3x3 PBC ED-CEQMC benchmark notes for attractive Hubbard model

Purpose: document the fixed-sector `3x3` periodic benchmark between low-temperature exact diagonalization and `CanEnsAFQMC`, so it can be reproduced and extended to a canonical superfluid-density benchmark later.

Repository root used locally:

```text
/Users/cosdis/Desktop/projects/CE_GCE
```

Important scripts:

```text
scripts/interacting_qmc_ed/ed_hubbard_ce_lowtemp_3x3.py
scripts/interacting_qmc_ed/benchmark_ce_qmc_vs_ed_3x3.jl
scripts/interacting_qmc_ed/compare_ce_qmc_vs_ed_3x3.py
```

Main result folders:

```text
results/interacting_qmc_ed/ed_ce_lowtemp_3x3_t1_Um5_Nup4_Ndn4_beta10
results/interacting_qmc_ed/ce_qmc_vs_ed_3x3_beta10
results/interacting_qmc_ed/ce_qmc_vs_ed_3x3_beta10_dtau0025
```

---

## 1. Benchmark parameters

Canonical fixed-spin sector:

```text
Lx = 3
Ly = 3
boundary condition = periodic in x and y
Nsite = 9
Nup = 4
Ndn = 4
t = 1
U = -5
beta = 10
mu = 0 (unused in the fixed-sector CE run)
```

The particle density is fixed to

\[
n = \frac{N_\uparrow + N_\downarrow}{N_{\rm site}} = \frac{8}{9}.
\]

For `CanEnsAFQMC`, the attractive-Hubbard runs used:

```text
useChargeHST = false
sys_type = ComplexF64
forceSymmetry = true
```

This matters. A later explicit one-slice `2x2` check showed that for negative `U` the
standard attractive-Hubbard HS sum is reproduced by the spin decomposition with
complex fields, not by the real charge-HS setup that had been used in the earliest
CE benchmark attempts.

---

## 2. ED method used here

For the fixed `(Nup,Ndn)=(4,4)` sector on `3x3`, the Hilbert dimension is

\[
\binom{9}{4}^2 = 15876.
\]

A full dense diagonalization is still expensive enough that the benchmark was done with a low-temperature eigenstate truncation:

1. Build the sparse Hamiltonian in the fixed `(Nup,Ndn)` sector.
2. Use `eigsh(..., which="SA")` to obtain low-energy eigenpairs.
3. Increase the requested number of eigenpairs until the retained spectrum covers an energy window
   \[
   E - E_0 \le 3.0.
   \]
4. Evaluate thermal averages at `beta=10` using only those retained states.

This is effectively exact for the benchmark point because the omitted Boltzmann tail is tiny:

```text
tail_weight_estimate ≈ 1.48e-9
retained_eigenstates = 52
```

The ED script writes:

```text
summary.tsv
correlations.tsv
retained_spectrum.tsv
```

The equal-time correlation observables were matched to the quantities already measured by `CanEnsAFQMC`:

- charge correlation
  \[
  C_n(\delta r)=\frac1N\sum_i \langle n_{i+\delta r}n_i\rangle
  \]
- spin-z correlation
  \[
  C_{S^z}(\delta r)=\frac1N\sum_i \langle (n_{i+\delta r,\uparrow}-n_{i+\delta r,\downarrow})(n_{i,\uparrow}-n_{i,\downarrow})\rangle
  \]
- s-wave pair correlation
  \[
  P_s(\delta r)=\frac1N\sum_i \langle \Delta^\dagger_{i+\delta r}\Delta_i + \Delta^\dagger_i\Delta_{i+\delta r}\rangle,
  \quad \Delta_i=c_{i\downarrow}c_{i\uparrow}.
  \]

For `3x3`, the CE correlation sampler only has the four symmetry-reduced displacements:

```text
(0,0), (0,1), (1,0), (1,1)
```

---

## 3. CE-QMC method used here

The canonical `CanEnsAFQMC` driver was set up to measure:

- kinetic, potential, and total energy
- double occupancy per site
- charge correlation
- spin-z correlation
- s-wave pair correlation

The relevant code path is:

```text
DensityMatrix
measure_Energy
CorrFuncSampler
measure_ChargeCorr
measure_SpinCorr
measure_PairCorr
```

The benchmark driver accumulates samples in batches and rewrites:

```text
summary.tsv
correlations.tsv
```

after each batch, so intermediate comparisons can be made.

---

## 4. Commands used

### ED reference

```bash
~/.venvs/myenv/bin/python scripts/interacting_qmc_ed/ed_hubbard_ce_lowtemp_3x3.py \
  --outdir results/interacting_qmc_ed/ed_ce_lowtemp_3x3_t1_Um5_Nup4_Ndn4_beta10
```

### CE-QMC, first production pass (`Delta_tau = 0.05`)

```bash
./scripts/run_julia_local.sh --project=julia_env \
  scripts/interacting_qmc_ed/benchmark_ce_qmc_vs_ed_3x3.jl
```

### CE-QMC, smaller-Trotter-step probe (`Delta_tau = 0.025`)

```bash
./scripts/run_julia_local.sh --project=julia_env \
  scripts/interacting_qmc_ed/benchmark_ce_qmc_vs_ed_3x3.jl \
  --dtau=0.025 \
  --batch-nsamples=128 \
  --max-batches=12 \
  --nwarmups=256 \
  --measure-interval=4 \
  --output-dir=results/interacting_qmc_ed/ce_qmc_vs_ed_3x3_beta10_dtau0025
```

### Comparison tables

```bash
~/.venvs/myenv/bin/python scripts/interacting_qmc_ed/compare_ce_qmc_vs_ed_3x3.py \
  --ed-dir results/interacting_qmc_ed/ed_ce_lowtemp_3x3_t1_Um5_Nup4_Ndn4_beta10 \
  --qmc-dir results/interacting_qmc_ed/ce_qmc_vs_ed_3x3_beta10_dtau0025 \
  --outdir results/interacting_qmc_ed/ce_qmc_vs_ed_3x3_beta10_dtau0025/comparison
```

---

## 5. ED reference values

From:

```text
results/interacting_qmc_ed/ed_ce_lowtemp_3x3_t1_Um5_Nup4_Ndn4_beta10/summary.tsv
results/interacting_qmc_ed/ed_ce_lowtemp_3x3_t1_Um5_Nup4_Ndn4_beta10/correlations.tsv
```

Summary:

| observable | ED |
|---|---:|
| E/site | -3.005734582 |
| K/site | -1.291147958 |
| V/site | -1.714586623 |
| double occ/site | 0.342917325 |

Correlations:

| dx | dy | charge | spin-z | pair |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1.574723538 | 0.203054240 | 0.685834649 |
| 0 | 1 | 0.603541708 | -0.043285941 | 0.357093216 |
| 1 | 0 | 0.603541708 | -0.043285941 | 0.357093216 |
| 1 | 1 | 0.780555185 | -0.007477619 | 0.241470841 |

---

## 6. CE-QMC benchmark results

### `Delta_tau = 0.05`

From:

```text
results/interacting_qmc_ed/ce_qmc_vs_ed_3x3_beta10/comparison/summary_comparison.tsv
results/interacting_qmc_ed/ce_qmc_vs_ed_3x3_beta10/comparison/correlation_comparison.tsv
```

Summary comparison:

| observable | ED | CE-QMC | CE stderr | CE - ED |
|---|---:|---:|---:|---:|
| E/site | -3.005734582 | -2.988249299 | 0.005435531 | +0.017485282 |
| K/site | -1.291147958 | -1.384614283 | 0.006084746 | -0.093466324 |
| V/site | -1.714586623 | -1.603635016 | 0.004347374 | +0.110951607 |
| double occ/site | 0.342917325 | 0.320727003 | 0.000869475 | -0.022190321 |

Interpretation: this first pass is internally stable, but it disagrees with ED beyond Monte Carlo error bars. This older run used the wrong attractive-`U` HS convention and should now be treated only as a diagnostic.

### `Delta_tau = 0.025`

From:

```text
results/interacting_qmc_ed/ce_qmc_vs_ed_3x3_beta10_dtau0025/comparison/summary_comparison.tsv
results/interacting_qmc_ed/ce_qmc_vs_ed_3x3_beta10_dtau0025/comparison/correlation_comparison.tsv
```

Summary comparison:

| observable | ED | CE-QMC | CE stderr | CE - ED |
|---|---:|---:|---:|---:|
| E/site | -3.005734582 | -2.992624366 | 0.008064451 | +0.013110216 |
| K/site | -1.291147958 | -1.332431926 | 0.009550859 | -0.041283968 |
| V/site | -1.714586623 | -1.660192440 | 0.005081230 | +0.054394184 |
| double occ/site | 0.342917325 | 0.332038488 | 0.001016246 | -0.010878837 |

Representative correlations:

| dx | dy | observable | ED | CE-QMC | CE stderr | CE - ED |
|---:|---:|---|---:|---:|---:|---:|
| 0 | 1 | charge | 0.603541708 | 0.603973931 | 0.004359878 | +0.000432223 |
| 0 | 1 | pair | 0.357093216 | 0.332408469 | 0.019732933 | -0.024684747 |
| 1 | 1 | charge | 0.780555185 | 0.779438175 | 0.004930289 | -0.001117011 |
| 1 | 1 | pair | 0.241470841 | 0.196495594 | 0.016192954 | -0.044975247 |

Interpretation: reducing `Delta_tau` improves the benchmark substantially. However, this older run still used the wrong attractive-`U` HS convention and should be superseded by the corrected spin-HS/complex benchmark below.

### Corrected attractive-`U` benchmark: `useChargeHST=false`, `sys_type=ComplexF64`, `Delta_tau = 0.025`

From:

```text
results/interacting_qmc_ed/ce_qmc_vs_ed_3x3_beta10_dtau0025_spincomplex/comparison/summary_comparison.tsv
results/interacting_qmc_ed/ce_qmc_vs_ed_3x3_beta10_dtau0025_spincomplex/comparison/correlation_comparison.tsv
```

Summary comparison:

| observable | ED | CE-QMC | CE stderr | CE - ED |
|---|---:|---:|---:|---:|
| E/site | -3.005734582 | -3.013223443 | 0.031813641 | -0.007488862 |
| K/site | -1.291147958 | -1.312686796 | 0.010453411 | -0.021538838 |
| V/site | -1.714586623 | -1.700536647 | 0.028634934 | +0.014049976 |
| double occ/site | 0.342917325 | 0.340107329 | 0.005726987 | -0.002809995 |

Representative correlations:

| dx | dy | observable | ED | CE-QMC | CE stderr | CE - ED |
|---:|---:|---|---:|---:|---:|---:|
| 0 | 1 | charge | 0.603541708 | 0.606630450 | 0.004248916 | +0.003088742 |
| 0 | 1 | pair | 0.357093216 | 0.345084640 | 0.010207710 | -0.012008575 |
| 1 | 1 | charge | 0.780555185 | 0.780423499 | 0.003007225 | -0.000131686 |
| 1 | 1 | pair | 0.241470841 | 0.232241097 | 0.008793781 | -0.009229744 |

Interpretation: once the attractive-`U` HS convention is corrected, the CE benchmark agrees with ED much better. The large discrepancy seen in the earliest CE runs was mainly a wrong-model issue, not just Monte Carlo noise.

---

## 7. Current-response / superfluid-density status

An exact-sector CE current benchmark is now available in

```text
scripts/interacting_qmc_ed/benchmark_ce_current_vs_ed_3x3_exactsector.jl
```

This benchmark evaluates the current response by lifting each sampled one-body HS
slice to the exact fixed-`N` many-body sector for one spin species, and then
computing the integrated current-current response directly in that sector. The
underlying estimator was validated on a fully enumerated `2x2` HS benchmark.

For the corrected attractive-`U` setup

```text
useChargeHST = false
sys_type = ComplexF64
Delta_tau = 0.025
```

the current benchmark result is:

```text
results/interacting_qmc_ed/ce_current_vs_ed_3x3_exactsector_dtau0025_12b4_spincomplex/summary.tsv
```

with

| observable | ED | CE-QMC | CE stderr | CE - ED |
|---|---:|---:|---:|---:|
| Lambda_L | 0.590922666 | 0.677815035 | 0.013362777 | +0.086892369 |
| Lambda_T | 0.064838451 | 0.227636771 | 0.048815412 | +0.162798320 |
| rho_s current | 0.131521054 | 0.112544566 | 0.013824425 | -0.018976488 |

Interpretation: after fixing the attractive-`U` HS convention, the CE current benchmark moves much closer to the ED superfluid-density value. The remaining discrepancy in `rho_s` is now at the level of about `1.4 sigma` for this sampling budget, although the individual longitudinal and transverse components are still noisier than the equal-time observables.
