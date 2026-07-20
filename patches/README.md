# Dependency patches

`CanEnsAFQMC-current-response.patch` records the CE dependency changes used by
the current-response workflow on top of CanEnsAFQMC commit
`21b4f6815d0b836973064ff8401fb2ba9c23b802`.

Apply it from the repository root with:

```bash
git -C external/CanEnsAFQMC apply ../../patches/CanEnsAFQMC-current-response.patch
```

The patch is kept in the parent repository because the upstream dependency is
owned outside this GitHub account.
