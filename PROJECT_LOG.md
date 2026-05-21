# Project Log

## 2026-03-17

- Installed the latest available CE QMC code in this workspace.
- Ran test calculations for the average energy at half-filling and compared them against Fig. 3 of the PRE paper.
- Observed that the CE QMC driver requires both `Nup` and `Ndn` to be specified explicitly.
- Current understanding: to obtain finite-temperature results at fixed total particle number `N_total`, one should average over all spin partitions satisfying `N_total = Nup + Ndn`.
- I could not find an upstream script that explicitly performs this spin-sector averaging.
- Neither using only `Nup = Ndn = N_total / 2` nor doing the averaging manually reproduces Fig. 3 perfectly so far.
- Open issue: clarify whether the paper’s Fig. 3 comparison relies on an additional weighting, convention, or implementation detail beyond the spin-sector average.

## 2026-03-24

- Confirmed that the intended setup for this project is the fixed spin-resolved canonical sector `(Nup, Ndn)`, not a fixed-`N_total` average over spin partitions.
- For the half-filled comparison to Fig. 3, we should use `Nup = Ndn`.
- The earlier fixed-total-`N` averaging idea should be treated as an exploratory side path, not the main interpretation of the CE data.

## 2026-03-24

- After discussion with Steve, I started a new mission for this project.
- New target: compute the compressibility as a function of temperature `T` for a non-interacting tight-binding model on a 2D circular lattice with open boundary conditions.

## 2026-03-24

- Following the conclusion that the relevant setup is fixed `(Nup, Ndn)` with `Nup = Ndn`, I still cannot reproduce the result in Fig. 3.
- Next step: ask Tong for help clarifying the remaining discrepancy.

## 2026-03-31

- I now have the summary of the CE versus GCE behavior in the non-interacting system on the circular lattice.

## 2026-04-01

- As discussed with Steve, I started calculating the non-interacting system on square lattices with both periodic boundary conditions (PBC) and open boundary conditions (OBC).

## 2026-04-06

- I have now finished the non-interacting square-lattice calculations for both boundary conditions at `L = 4, 8, 16, 32`.

## 2026-04-16

- I finished the summary comparing the non-interacting system across disk, square PBC, and square OBC geometries.
- I also finished the exact-diagonalization benchmark of the canonical-ensemble QMC energy versus `beta` on the `4x2` cluster.
- Next phase: move on to the attractive Hubbard model.

## 2026-05-21

- I set up SMOQYDQMC on CADES to properly use MPI and checkpointing for grand-canonical-ensemble checks of the superconducting transition in the attractive-`U` Hubbard model.
- I am now running the calculation to obtain results for `U = -5`.
