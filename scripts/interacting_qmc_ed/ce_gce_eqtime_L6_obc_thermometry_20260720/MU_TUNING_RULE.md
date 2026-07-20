# L=6 OBC GCE Chemical-Potential Tuning Rule

**Normative reference:**
`/home/9pm/nUHubbard/scripts/interacting_qmc_ed/ce_gce_eqtime_L6_thermometry_20260719/MU_TUNING_RULE.md`

**OBC adaptation:** The confirmed `mu_L6_PBC_reference` replaces the L=8 value
as the initial OBC seed. Its inherited `mu_L8_reference` remains recorded as
provenance. After initialization, neither reference value constrains the search
center or final bracket; all decisions use measured L=6 OBC densities.

## Dynamic-center search

1. Start at `mu_L6_PBC_reference` and offsets `-0.02` and `+0.02`.
2. Wait for every currently planned probe for the target to finish.
3. If no measured-density bracket exists, select the completed OBC point whose
   achieved density is closest to target and move exactly one directed
   `delta_mu=0.02` step in the required direction.
4. If that point was already sampled, continue in directed 0.02 increments to
   the first unsampled point.
5. Submit only that one point. Never add a mirrored point merely for symmetry.

## Bracket refinement

A bracket must consist of completed OBC probes satisfying

```text
(N_low - N_target) * (N_high - N_target) <= 0
```

and a monotonic increasing density response. If its endpoint separation exceeds
0.02, move the endpoint closest in density to target toward the opposite
endpoint by 0.02 and submit one interior point. Repeat until

```text
abs(mu_high - mu_low) <= 0.02
```

A tighter bracket is acceptable.

## Production admission

Production is admitted only when all conditions hold:

1. Completed probes straddle the target density.
2. Bracket width is at most 0.02.
3. `mu_fitted` lies inside the bracket.
4. The independent confirmation `mu_final` lies inside the same bracket.
5. The confirmation satisfies `abs(N_confirmation-N_target) <= 0.03`.
6. Production independently satisfies `abs(N_achieved-N_target) <= 0.03`.
7. Production has complete expected-rank, site-accumulator, and four-primary-
   table coverage.

A failed production density check requires a new independent retune
confirmation and a new production attempt; accumulators are never reset.

## Superseded results and fail-closed behavior

Any historical production row with bracket width above 0.02, a fitted/final mu
outside its bracket, or failed achieved-density tolerance is non-authoritative.
It is excluded from repair, collection, plots, thermometry, and slides. Running
historical legs need not be canceled, but their output remains excluded.

The rule is independently enforced by `tune_mu.py`,
`run_gce_production_manifest_task.sh`, `repair_stale_l6_roots.py`,
`audit_status.py`, and `collect_l6_results.py`.
