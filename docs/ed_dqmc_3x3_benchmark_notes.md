# 3x3 PBC ED-DQMC benchmark notes for attractive Hubbard model

Purpose: document the 3x3 periodic-boundary benchmark between ED and SmoQyDQMC so it can be reproduced later and adapted to a canonical-ensemble QMC benchmark.

Repository root used locally:

```text
/Users/cosdis/Desktop/projects/CE_GCE
```

Important scripts:

```text
scripts/interacting_qmc_ed/ed_hubbard_gce_lowtemp_3x3.py
scripts/interacting_qmc_ed/run_smoqydqmc_attractive_hubbard.jl
scripts/interacting_qmc_ed/run_smoqydqmc_attractive_hubbard_checkpoint.jl
scripts/interacting_qmc_ed/compute_smoqydqmc_superfluid_density.py
```

Main result table:

```text
results/interacting_qmc_ed/benchmark_3x3_t1_tp0_Um5_mu_m1_beta10_ed_dqmc.tsv
```

---

## 1. Hamiltonian convention

The preferred production convention is to keep the interaction term in particle-hole-symmetric form:

\[
H = K + U\sum_i (n_{i\uparrow}-1/2)(n_{i\downarrow}-1/2) - \mu\sum_{i\sigma} n_{i\sigma}.
\]

Use:

```text
ph_sym_form = true
```

The equivalent conventional chemical potential would be

\[
\mu_{\rm conv}=\mu+U/2,
\]

up to the constant shift \(U N_{\rm site}/4\). For this project, keep the Hamiltonian itself in the particle-hole-symmetric form and compare ED/QMC using the same convention.

The `ph_sym_form=false` benchmark below is kept only as a diagnostic because it was previously requested. It is not the preferred production convention.

---

## 2. Benchmark parameters

```text
Lx = 3
Ly = 3
boundary condition = periodic in x and y
Nsite = 9
t = 1
t' = 0
U = -5
mu = -1
beta = 10
Delta_tau = 0.05
```

DQMC statistics used:

```text
N_therm = 5000
N_measurements = 50000
N_bins = 50
N_updates = 5
```

`N_updates` means the number of full update sweeps between measurements.

---

## 3. DQMC current measurement used for superfluid density

The SmoQyDQMC driver initializes the x-current/current measurement as:

```julia
initialize_correlation_measurements!(
    measurement_container = measurement_container,
    model_geometry = model_geometry,
    correlation = "current",
    time_displaced = false,
    integrated = true,
    pairs = [(1, 1)]
)
```

Here `HOPPING_ID=1` is the `+x` nearest-neighbor hopping. `integrated = true` gives the imaginary-time-integrated static response

\[
\Lambda_{xx}(\mathbf q,0)
=\frac{1}{N}\int_0^\beta d\tau\,
\langle J_x(\mathbf q,\tau)J_x(-\mathbf q,0)\rangle.
\]

The finite-size superfluid density was computed as

\[
\rho_s(L,T)=\frac14\left[\Lambda_{xx}(q_{\min},0,0)-\Lambda_{xx}(0,q_{\min},0)\right],
\quad q_{\min}=2\pi/L.
\]

The postprocessor also reports

\[
\rho_s^{\rm dia}=\frac14\left[\frac{-K_x}{N}-\Lambda_{xx}(0,q_{\min},0)\right].
\]

For conductivity, this measurement is not enough; full time-displaced current data would be needed. For superfluid density, the integrated current response is the desired object.

---

## 4. ED method used for the 3x3 GCE benchmark

A full all-sector dense finite-temperature ED trace for 3x3 is expensive because the largest fixed-spin sector has dimension 15876. For the beta=10 benchmark, the ED reference used low-temperature sector truncation:

1. Find the minimum energy in every `(N_up,N_dn)` sector using sparse diagonalization.
2. Select sectors with sector ground energy within a cutoff of the global ground energy.
3. Fully diagonalize every selected sector.
4. Evaluate thermodynamics and current response with exact Lehmann sums inside the selected sectors.

This is effectively exact at beta=10 for the selected cases because omitted sectors have very small Boltzmann weight, but it is not an all-sector finite-temperature trace.

ED current response used the Lehmann form of the integrated current correlator:

\[
\Lambda_{xx} = \frac{1}{NZ}\sum_{m,n}
|\langle m|J_x|n\rangle|^2
\frac{e^{-\beta E_n}-e^{-\beta E_m}}{E_m-E_n},
\]

with the diagonal limit handled as \(\beta e^{-\beta E_m}\).

---

## 5. Commands used to reproduce this benchmark

### ED, `ph_sym_form=true`

```bash
~/.venvs/myenv/bin/python scripts/interacting_qmc_ed/ed_hubbard_gce_lowtemp_3x3.py \
  --ph-sym-form true \
  --cutoff 1.6 \
  --outdir results/interacting_qmc_ed/ed_lowtemp_3x3_t1_Um5_mu_m1_beta10_phtrue_cut1p6
```

Selected sectors:

| N_up | N_dn | dim | sector Emin |
|---:|---:|---:|---:|
| 1 | 1 | 81 | -13.326582598824665 |
| 2 | 2 | 1296 | -12.633410792673853 |
| 0 | 1 | 9 | -11.750000000000000 |
| 1 | 0 | 9 | -11.750000000000000 |
| 2 | 1 | 324 | -11.741034950308684 |
| 1 | 2 | 324 | -11.741034950308666 |

### ED, `ph_sym_form=false` diagnostic

```bash
~/.venvs/myenv/bin/python scripts/interacting_qmc_ed/ed_hubbard_gce_lowtemp_3x3.py \
  --ph-sym-form false \
  --cutoff 2.1 \
  --outdir results/interacting_qmc_ed/ed_lowtemp_3x3_t1_Um5_mu_m1_beta10_phfalse_cut2p1
```

Selected sectors:

| N_up | N_dn | dim | sector Emin |
|---:|---:|---:|---:|
| 9 | 9 | 1 | -27.000000000000000 |
| 8 | 8 | 81 | -26.091721009493920 |
| 7 | 7 | 1296 | -25.050443144957104 |
| 8 | 9 | 9 | -25.000000000000004 |
| 9 | 8 | 9 | -25.000000000000004 |

### DQMC, `ph_sym_form=true`

```bash
./scripts/run_julia_local.sh --project=julia_env \
  scripts/interacting_qmc_ed/run_smoqydqmc_attractive_hubbard.jl \
  0 -5.0 0.0 -1.0 3 10.0 5000 50000 50 5 true \
  results/interacting_qmc_ed/dqmc_current_3x3_t1_tp0_Um5_mu_m1_beta10_phtrue_n50000 \
  3
```

### DQMC, `ph_sym_form=false` diagnostic

```bash
./scripts/run_julia_local.sh --project=julia_env \
  scripts/interacting_qmc_ed/run_smoqydqmc_attractive_hubbard.jl \
  0 -5.0 0.0 -1.0 3 10.0 5000 50000 50 5 false \
  results/interacting_qmc_ed/dqmc_current_3x3_t1_tp0_Um5_mu_m1_beta10_phfalse_n50000 \
  3
```

### Postprocess DQMC superfluid density

```bash
~/.venvs/myenv/bin/python scripts/interacting_qmc_ed/compute_smoqydqmc_superfluid_density.py \
  results/interacting_qmc_ed/dqmc_current_3x3_t1_tp0_Um5_mu_m1_beta10_phtrue_n50000/attractive_hubbard_rect_U-5.00_tp0.00_mu-1.00_Lx3_Ly3_b10.00-1

~/.venvs/myenv/bin/python scripts/interacting_qmc_ed/compute_smoqydqmc_superfluid_density.py \
  results/interacting_qmc_ed/dqmc_current_3x3_t1_tp0_Um5_mu_m1_beta10_phfalse_n50000/attractive_hubbard_rect_U-5.00_tp0.00_mu-1.00_Lx3_Ly3_b10.00-1
```

The postprocessor reads `stats_pID-0.h5` directly to avoid CSV rounding. This matters for very small current responses.

---

## 6. Benchmark results

### `ph_sym_form=true`: preferred convention

| observable | ED | DQMC | DQMC - ED |
|---|---:|---:|---:|
| E/site | -1.480655701 | -1.481315128 ± 0.001043527 | -0.000659427 |
| density | 0.222439518 | 0.222427252 ± 0.000204957 | -0.000012266 |
| double occ/site | 0.041511463 | 0.041979337 ± 0.000130297 | 0.000467874 |
| compressibility | 0.004341456 | 0.004102076 ± 0.004100877 | -0.000239380 |
| Lambda_L | 0.400818349 | 0.397511055 ± 0.000843052 | -0.003307294 |
| Lambda_T | 0.001214041 | 0.001555032 ± 0.000537116 | 0.000340991 |
| rho_s current | 0.099901077 | 0.098989006 ± 0.000249904 | -0.000912071 |
| -Kx/N | 0.400818349 | 0.400029166 ± 0.000639622 | -0.000789183 |
| rho_s diamagnetic | 0.099901077 | 0.099618533 ± 0.000208808 | -0.000282544 |

Interpretation: energy, density, and diamagnetic superfluid estimator agree at the expected level for this finite-statistics, finite-Delta_tau DQMC run. The current-only estimator is slightly lower; the diamagnetic form is numerically more stable here.

### `ph_sym_form=false`: diagnostic only

| observable | ED | DQMC | DQMC - ED |
|---|---:|---:|---:|
| E/site | -2.999988235 | -2.9999999996 ± 0.0000000123 | -0.000011765 |
| density | 1.999974348 | 1.9999999993 ± 0.0000000022 | 0.000025651 |
| double occ/site | 0.999984146 | 0.9999999993 ± 0.0000000022 | 0.000015853 |
| compressibility | 0.000512989 | 0.0000000071 ± 0.0000000220 | -0.000512982 |
| Lambda_L | 0.000020926 | 0.00000000094 ± 0.00000000418 | -0.000020925 |
| Lambda_T | 0.000006180 | -0.0000000157 ± 0.0000000254 | -0.000006196 |
| rho_s current | 0.000003686 | 0.00000000417 ± 0.00000000645 | -0.000003682 |
| -Kx/N | 0.000020926 | 0.00000000036 ± 0.00000000380 | -0.000020926 |
| rho_s diamagnetic | 0.000003686 | 0.00000000402 ± 0.00000000643 | -0.000003682 |

Interpretation: at `U=-5, mu=-1, beta=10`, the conventional-form system is essentially full. The DQMC chain stayed saturated and did not resolve the tiny hole/current response. This is why `ph_sym_form=false` is not a useful production convention for this parameter point.

---

## 7. Energy reconstruction in DQMC output

Do not use any artificial corrected energy. For `t'=0`, reconstruct the DQMC energy/site as

```text
E/site = hopping_energy(HOPPING_ID=1)
       + hopping_energy(HOPPING_ID=2)
       + hubbard_energy(HUBBARD_ID=1)
       + onsite_energy(ORBITAL_ID=1)
```

For nonzero `t'`, also include `HOPPING_ID=3` and `HOPPING_ID=4`.

For `ph_sym_form=true`, SmoQy `hubbard_energy` corresponds to

\[
U\langle(n_\uparrow-1/2)(n_\downarrow-1/2)\rangle.
\]

The onsite term is the chemical-potential contribution.

---

## 8. How to adapt this to a canonical-ensemble QMC benchmark

For canonical-ensemble QMC, do **not** use the grand-canonical ED trace at fixed `mu`. Instead compare at fixed particle sector.

### Canonical ED target

Choose the same fixed particle numbers as CE QMC:

```text
N_up = fixed by CE QMC
N_dn = fixed by CE QMC
N_total = N_up + N_dn
```

Then ED should compute

\[
Z_{N_\uparrow,N_\downarrow}=\mathrm{Tr}_{N_\uparrow,N_\downarrow} e^{-\beta H_{N_\uparrow,N_\downarrow}}.
\]

No sum over particle sectors. No chemical-potential term is needed for canonical comparisons unless the CE QMC code explicitly includes it as a constant bookkeeping term. If included, \(-\mu N_{\rm total}\) is a constant in a fixed sector and should be handled consistently on both sides.

### Canonical energy with particle-hole-symmetric interaction

For a fixed sector, use

\[
E = \langle K\rangle
+ U\left\langle\sum_i(n_{i\uparrow}-1/2)(n_{i\downarrow}-1/2)\right\rangle.
\]

Equivalently,

\[
E = \langle K\rangle
+ U\left(D - \frac{N_{\rm total}}{2} + \frac{N_{\rm site}}{4}\right),
\]

where

\[
D=\left\langle\sum_i n_{i\uparrow}n_{i\downarrow}\right\rangle.
\]

Per-site double occupancy is \(D/N_{\rm site}\).

### Canonical current response

The same superfluid-density formula applies, but the trace is only over the fixed canonical sector:

\[
\Lambda_{xx}^{N_\uparrow,N_\downarrow}(\mathbf q,0)
=\frac{1}{NZ_{N_\uparrow,N_\downarrow}}
\sum_{m,n\in (N_\uparrow,N_\downarrow)}
|\langle m|J_x(\mathbf q)|n\rangle|^2
\frac{e^{-\beta E_n}-e^{-\beta E_m}}{E_m-E_n}.
\]

Then

\[
\rho_s=\frac14\left[\Lambda_{xx}(q_{\min},0,0)-\Lambda_{xx}(0,q_{\min},0)\right]
\]

or

\[
\rho_s=\frac14\left[\frac{-K_x}{N}-\Lambda_{xx}(0,q_{\min},0)\right].
\]

### Practical checklist for CE QMC comparison

1. Use the same lattice and PBC convention. For safety use `Lx,Ly >= 3`.
2. Use the same Hamiltonian convention: `ph_sym_form=true`.
3. Fix the same `(N_up,N_dn)` in ED and CE QMC.
4. Compare canonical quantities, not grand-canonical averages:
   - `E/site`
   - `double_occ/site`
   - `Kx/site` or `-Kx/N`
   - `Lambda_L`, `Lambda_T`
   - `rho_s_current`, `rho_s_diamagnetic`
5. If CE QMC reports density, it should be exactly `N_total/Nsite` up to normalization.
6. If CE QMC includes a chemical-potential bookkeeping term, add the same constant `-mu*N_total` in ED before comparing energies; otherwise omit it.

---

## 9. Known caveats from this benchmark

- The 3x3 ED values above are low-temperature selected-sector ED, not a full all-sector grand-canonical trace.
- For a canonical benchmark, the fixed sector may be much easier: only diagonalize the chosen `(N_up,N_dn)` sector.
- The `ph_sym_form=false` grand-canonical case at this parameter point is nearly full and not a good stress test for current response.
- For `L=2`, SmoQy periodic-bond convention and standard ED duplicate-bond convention can differ. Avoid `L=2` for final ED/QMC validation.
- Use HDF5 stats rather than rounded CSV when current responses are tiny.
