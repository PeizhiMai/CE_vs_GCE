# CanEnsAFQMC Setup

This workspace is now focused on the canonical AFQMC implementation in [external/CanEnsAFQMC](/Users/cosdis/Desktop/projects/CE_GCE/external/CanEnsAFQMC).

Reference paper:

- [reference/PhysRevE.107.055302.pdf](/Users/cosdis/Desktop/projects/CE_GCE/reference/PhysRevE.107.055302.pdf)
  Title: `Stable recursive auxiliary field quantum Monte Carlo algorithm in the canonical ensemble: Applications to thermometry and the Hubbard model`

## Current target

Reproduce the CE data from Fig. 3 of the preprint for `beta <= 6`.

The local driver for that is:

- [scripts/scan_fig3_ce_energy.jl](/Users/cosdis/Desktop/projects/CE_GCE/scripts/scan_fig3_ce_energy.jl)
- [scripts/scan_fig3_ce_energy_fixed_ntotal.jl](/Users/cosdis/Desktop/projects/CE_GCE/scripts/scan_fig3_ce_energy_fixed_ntotal.jl)
  for exploratory fixed-total-`N` reconstruction across spin sectors

Default scan parameters:

- `Lx = Ly = 6`
- `Nup = Ndn = 18`
- `U = 4.0`
- `beta_list = [0.5, 1.2, 2.4, 3.6, 5.0, 6.0]`
- `dtau = 0.1`, so `L = beta / dtau`

Outputs:

- `results/fig3_ce_energy_scan/metadata.toml`
- `results/fig3_ce_energy_scan/ce_energy.tsv`

The TSV includes per-beta kinetic, potential, and total energy per particle, plus a standard error estimate from the Monte Carlo samples.

## Single-particle disk spectrum

There is now a standalone solver for the non-interacting tight-binding spectrum on a
2D circular lattice with open boundaries:

- [scripts/solve_disk_single_particle_spectrum.jl](/Users/cosdis/Desktop/projects/CE_GCE/scripts/solve_disk_single_particle_spectrum.jl)
- [scripts/disk_spectrum.py](/Users/cosdis/Desktop/projects/CE_GCE/scripts/disk_spectrum.py)

It treats the disk as the set of integer lattice sites satisfying `x^2 + y^2 <= R^2`,
connects nearest neighbors with hopping `-t`, and diagonalizes the resulting
single-particle Hamiltonian exactly.

Example:

```bash
./scripts/run_julia_local.sh scripts/solve_disk_single_particle_spectrum.jl --radius=5
python3 scripts/disk_spectrum.py --radii=3,4,5
```

Outputs are written to `results/disk_spectrum_radius_<R>/`:

- `metadata.toml`
- `sites.tsv`
- `eigenvalues.tsv`

## Canonical density-correlation compressibility scan

There is also an exact finite-temperature driver for the non-interacting disk at fixed
`Nup` and `Ndn`:

- [scripts/scan_disk_canonical_compressibility.jl](/Users/cosdis/Desktop/projects/CE_GCE/scripts/scan_disk_canonical_compressibility.jl)

Example:

```bash
./scripts/run_julia_local.sh scripts/scan_disk_canonical_compressibility.jl \
  --radius=5 --nup=5 --ndn=5 --temperatures=0.25,0.5,1.0,2.0
```

It writes:

- `single_particle_energies.tsv`
- `compressibility_scan.tsv`
- `sites.tsv`
- `metadata.toml`

The integrated compressibility is evaluated as
`kappa = beta / V * sum_ij (<n_i n_j> - <n_i><n_j>)`.
Because the calculation is performed in a strictly canonical sector with fixed
`Nup` and `Ndn`, this global quantity should vanish up to numerical roundoff.

## Local Julia workflow

Use the wrapper script so `juliaup` and Julia package state stay inside this workspace:

```bash
./scripts/run_julia_local.sh scripts/bootstrap_canensafqmc.jl
./scripts/run_julia_local.sh scripts/scan_fig3_ce_energy.jl
./scripts/run_julia_local.sh scripts/scan_fig3_ce_energy_fixed_ntotal.jl --max_workers=1
```

## Notes

- I patched the vendored `CanEnsAFQMC` module to include and export its existing `measure_Energy` implementation.
- The public upstream examples do not include a ready-made Fig. 3 energy scan driver, so this repo now provides one.
- The fixed-total-`N` driver reconstructs sector weights using
  `log Z_sector(beta) = log Z_sector(0) - ∫_0^beta E_sector(beta') d beta'`,
  with `log Z_sector(0) = log(C(V,N_up) C(V,N_dn))`.
- In this sandbox, Julia worker startup may fail with a socket bind error, so the
  fixed-total-`N` script should be run with `--max_workers=1` unless that restriction is lifted.

## Session Notes

### Main outputs

- Half-filled CE scan output:
  - [results/fig3_ce_energy_scan/ce_energy.tsv](/Users/cosdis/Desktop/projects/CE_GCE/results/fig3_ce_energy_scan/ce_energy.tsv)
  - [results/fig3_ce_energy_scan/batch_progress.tsv](/Users/cosdis/Desktop/projects/CE_GCE/results/fig3_ce_energy_scan/batch_progress.tsv)
- Digitized Fig. 3 and overlay:
  - [results/fig3_digitized_overlay/fig3_ce_digitized.tsv](/Users/cosdis/Desktop/projects/CE_GCE/results/fig3_digitized_overlay/fig3_ce_digitized.tsv)
  - [results/fig3_digitized_overlay/fig3_gce_digitized.tsv](/Users/cosdis/Desktop/projects/CE_GCE/results/fig3_digitized_overlay/fig3_gce_digitized.tsv)
  - [results/fig3_digitized_overlay/fig3_ce_vs_run.tsv](/Users/cosdis/Desktop/projects/CE_GCE/results/fig3_digitized_overlay/fig3_ce_vs_run.tsv)
  - [results/fig3_digitized_overlay/fig3_overlay.png](/Users/cosdis/Desktop/projects/CE_GCE/results/fig3_digitized_overlay/fig3_overlay.png)
- Dedicated `beta=6` rerun target folder:
  - [results/fig3_ce_energy_beta6_target003](/Users/cosdis/Desktop/projects/CE_GCE/results/fig3_ce_energy_beta6_target003)

### Half-filled CE results so far

From [results/fig3_ce_energy_scan/ce_energy.tsv](/Users/cosdis/Desktop/projects/CE_GCE/results/fig3_ce_energy_scan/ce_energy.tsv):

- `beta=0.5`: `E/N = -0.159077`, `stderr = 0.001826`
- `beta=1.2`: `E/N = -0.602233`, `stderr = 0.002848`
- `beta=2.4`: `E/N = -0.747273`, `stderr = 0.002721`
- `beta=3.6`: `E/N = -0.802760`, `stderr = 0.003000`
- `beta=5.0`: `E/N = -0.838665`, `stderr = 0.002851`
- `beta=6.0`: `E/N = -0.859225`, `stderr = 0.007129`

Five of the six `beta <= 6` points met the `stderr <= 0.003` target. `beta=6.0` did not, and a deeper single-point rerun was started in [results/fig3_ce_energy_beta6_target003](/Users/cosdis/Desktop/projects/CE_GCE/results/fig3_ce_energy_beta6_target003).

### Digitized Fig. 3 CE values

Extracted by [scripts/digitize_fig3_overlay.py](/Users/cosdis/Desktop/projects/CE_GCE/scripts/digitize_fig3_overlay.py):

- `0.5 -> -0.160067`
- `1.2 -> -0.534951`
- `2.4 -> -0.718655`
- `3.6 -> -0.794478`
- `5.0 -> -0.838471`
- `6.0 -> -0.850752`

The comparison table is [results/fig3_digitized_overlay/fig3_ce_vs_run.tsv](/Users/cosdis/Desktop/projects/CE_GCE/results/fig3_digitized_overlay/fig3_ce_vs_run.tsv).

### `Nup=Ndn=23` test

The `N=23+23` exploratory runs did not match Fig. 3 at all:

- [results/fig3_ce_energy_scan_N46_beta_0.5/ce_energy.tsv](/Users/cosdis/Desktop/projects/CE_GCE/results/fig3_ce_energy_scan_N46_beta_0.5/ce_energy.tsv)
- [results/fig3_ce_energy_scan_N46_beta_1.2/ce_energy.tsv](/Users/cosdis/Desktop/projects/CE_GCE/results/fig3_ce_energy_scan_N46_beta_1.2/ce_energy.tsv)
- [results/fig3_ce_energy_scan_N46_beta_2.4/ce_energy.tsv](/Users/cosdis/Desktop/projects/CE_GCE/results/fig3_ce_energy_scan_N46_beta_2.4/ce_energy.tsv)
- [results/fig3_ce_energy_scan_N46_beta_3.6/ce_energy.tsv](/Users/cosdis/Desktop/projects/CE_GCE/results/fig3_ce_energy_scan_N46_beta_3.6/ce_energy.tsv)
- [results/fig3_ce_energy_scan_N46_beta_5.0/ce_energy.tsv](/Users/cosdis/Desktop/projects/CE_GCE/results/fig3_ce_energy_scan_N46_beta_5.0/ce_energy.tsv)

So the earlier `N46` clue from the companion repo should not be used to interpret Fig. 3.

### Important ensemble caveat

`CanEnsAFQMC` works in a fixed spin-resolved canonical sector `(N_up, N_dn)`.

- [external/CanEnsAFQMC/src/base/systems.jl](/Users/cosdis/Desktop/projects/CE_GCE/external/CanEnsAFQMC/src/base/systems.jl) stores `N` as `Tuple{Int64, Int64}`.
- [external/CanEnsAFQMC/src/utils/ce_recursion.jl](/Users/cosdis/Desktop/projects/CE_GCE/external/CanEnsAFQMC/src/utils/ce_recursion.jl) performs canonical recursion for one `N` at a time.
- For the half-filled Fig. 3 comparison in this workspace, the confirmed choice is `Nup = Ndn`.

This means the existing Fig. 3 scan script should be interpreted as a fixed `(N_up, N_dn)` calculation, not as a fixed-total-`N` canonical average.

### Fixed-total-`N` reconstruction status

There is now a dedicated fixed-total-`N` driver:

- [scripts/scan_fig3_ce_energy_fixed_ntotal.jl](/Users/cosdis/Desktop/projects/CE_GCE/scripts/scan_fig3_ce_energy_fixed_ntotal.jl)

Current validation output:

- Modest run at `beta=0.5`:
  [results/fig3_ce_energy_fixed_ntotal_beta_0.5/fixed_ntotal_energy.tsv](/Users/cosdis/Desktop/projects/CE_GCE/results/fig3_ce_energy_fixed_ntotal_beta_0.5/fixed_ntotal_energy.tsv)
  giving `E/N = -0.158299` with approximate propagated `stderr = 0.001290`.
- Smoke run at `beta=0.5`:
  [results/fig3_ce_energy_fixed_ntotal_smoke/fixed_ntotal_energy.tsv](/Users/cosdis/Desktop/projects/CE_GCE/results/fig3_ce_energy_fixed_ntotal_smoke/fixed_ntotal_energy.tsv)
  giving `E/N = -0.151871`.

This smoke result is only a pipeline check because it used very small per-sector sample counts.

### Paper interpretation caveat

The fixed-total-`N` reconstruction path remains in this repo for comparison work, but it is not the primary interpretation being used for the current Fig. 3 reproduction target.
