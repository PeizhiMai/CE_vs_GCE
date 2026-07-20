# L=6 CE/GCE equal-time thermometry (2026-07-19)

Fresh L=6 counterpart to the accepted L=8 workflow.  Historical L=6 U=+4
sign-test outputs are never reused.

## Fixed physics grid

- 6x6 PBC, t=1, t'=0, dtau=0.1.
- Ntot = 12, 18, 26, 32 with balanced CE sectors.
- U = -5, -3, 0, +3, +5.
- beta = 2, 2.2, 2.5, 2.9, 3.3, 4, 5, 6.7, 10, plus beta=20 for U<=0.
- 192 conditions: 152 interacting-QMC and 40 exact U=0 conditions.

`manifests/L8_mu_reference_for_L6.tsv` is the authoritative provenance table.
It contains 152 chemical potentials recovered from final L=8 production
manifests and 40 exactly recomputed L=8 U=0 values.  The L=6-to-L=8 particle
mapping is 12->22, 18->32, 26->46, 32->56.

## Acceptance rules

- Every CE root needs all expected rank completion markers and all four
  combined equal-time tables.
- Positive-U CE is spin-HS, `force_symmetry=false`, `phase_reweighted=true`.
  Rank outputs are pooled by global signed numerator / global phase sum.
- Every GCE production root needs all 32 rank metadata files, all four exports,
  and `abs(N_achieved-N_target) <= 0.03`.
- Checkpoint accumulators are never reset.
- CADES account/partition/QOS are fixed at `ccsd/burst/default`.

## Local validation

```bash
~/.venvs/myenv/bin/python prepare_workflow.py
~/.venvs/myenv/bin/python exact_u0_l6.py \
  --validate-l8 status_source/l8_reference_sources/source_l8_snapshot_20260718.tsv
~/.venvs/myenv/bin/python validate_workflow.py
```

## Staged execution

1. Run the two-rank CE and GCE smoke tests.
2. Submit the four immediate CE arrays and the two initial GCE probe arrays
   with `submit_initial_cades.sh` (arrays are unthrottled).
3. After all positive-U pilots finish, run:

```bash
python3.11 summarize_positive_pilots.py \
  manifests/ce_L6_positive_pilot_beta5_6p7_10_r32_m10000.tsv \
  --out status_source/positive_pilot_status.tsv --manifest-dir manifests \
  --write-production
```

4. After all currently planned GCE probes finish, run `tune_mu.py` with every
   probe manifest generated so far.  It writes symmetric extension probes,
   secant confirmation probes, or confirmed production manifests.  Never
   submit a follow-up manifest without first checking that no root is already
   active or pending.

```bash
python3.11 tune_mu.py manifests/gce_mu_probe_L6_*.tsv \
  --outdir status_source/mu_tuning --manifest-dir manifests --write-next
```

5. Submit admitted positive CE rows with the r32/r64 CE wrappers and confirmed
   GCE rows with the attractive/spinHS production wrappers.

   On CADES, `advance_l6_stages_cades.sh` performs steps 3–5 idempotently.  It
   writes/submits only newly admitted rows, keeps arrays unthrottled on
   `ccsd/burst/default`, and records every root in
   `status_source/submission_ledger.tsv`.  Repeated calls do not duplicate a
   final, submitted, active-manifest, or marker-bearing root.  Checkpoint repair
   remains a separate audited action rather than bypassing the ledger.

6. Strict collection is performed on CADES.  The collector rejects incomplete
   rank/table coverage, out-of-tolerance GCE density, duplicate roots, and an
   incorrect positive-U phase pool.

```bash
python3.11 collect_l6_results.py \
  --exact-snapshot status_source/exact_u0_L6_snapshot.tsv \
  --snapshot status_source/analysis/l6_snapshot_current.tsv \
  --status status_source/analysis/l6_condition_status.tsv \
  --require-complete
```

7. Once all required rows are strict-final, run
   `finalize_l6_results_local.sh` locally.  It collects the remote snapshot,
   generates the 14 L=6 comparison/mismatch/thermometry assets, imports the
   validated 29-slide L=8/L=12 core with artifact-tool, inserts the L=6 section
   before the conclusion, exports the 43-slide deck, renders all slides, runs
   overflow QA, and mirrors timestamped outputs under
   `/Users/cosdis/CE_GCE_no_icloud/results`.

## Analysis contract

Thermometry uses kinetic energy/site, double occupancy/site, nearest-neighbor
spin, and nearest-neighbor connected charge.  The plotted/statistical quantity
is `(T_GCE-T_CE)/T_CE`.  Curves join only actual unique-temperature simulated
points and break at no-solution, ambiguous, flat-calibration, or sign-limited
regions.  NNN observables and other measured quantities are supplemental.
The four thermometer figures use slide-wide landscape canvases with taller
panels.  No dense interpolation curve is displayed: only consecutive actual
simulated points with a unique inferred temperature are joined.
