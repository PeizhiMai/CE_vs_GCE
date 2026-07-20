# Dependency patches

`CanEnsAFQMC-current-response.patch` records the CE dependency changes used by
the current-response workflow on top of CanEnsAFQMC commit
`21b4f6815d0b836973064ff8401fb2ba9c23b802`.

`CanEnsAFQMC-obc.patch` adds the explicit open-boundary square-lattice hopping
constructor and deterministic 2x2/3x3 bond-count tests on the same pinned base.
The default periodic constructor is unchanged.

Apply it from the repository root with:

```bash
julia --version  # must report 1.12.1 for the pinned validation environment
JULIA_DEPOT_PATH="$PWD/.julia_depot:$HOME/.julia" \
  julia --project=julia_env \
  scripts/interacting_qmc_ed/bootstrap_canensafqmc.jl
```

The patch is kept in the parent repository because the upstream dependency is
owned outside this GitHub account.
