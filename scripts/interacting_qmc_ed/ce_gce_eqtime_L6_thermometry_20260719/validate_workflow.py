#!/usr/bin/env python3.11
"""Static and numerical QA for the L=6 CE/GCE thermometry workflow."""

from __future__ import annotations

import csv
import importlib.util
import math
import py_compile
import subprocess
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
M = HERE / "manifests"
RESULTS = ROOT / "results" / "ce_gce_eqtime_L6_thermometry_20260719"


def rows(name: str) -> list[dict[str, str]]:
    with (M / name).open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"PASS {message}")


def main() -> None:
    py_files = sorted(HERE.glob("*.py"))
    for path in py_files:
        py_compile.compile(str(path), doraise=True)
    check(bool(py_files), f"Python syntax ({len(py_files)} files)")

    shell_files = sorted(HERE.glob("*.sh")) + sorted(HERE.glob("*.sbatch"))
    for path in shell_files:
        subprocess.run(["bash", "-n", str(path)], check=True)
    check(bool(shell_files), f"shell syntax ({len(shell_files)} files)")

    grid = rows("L6_full_condition_grid.tsv")
    refs = rows("L8_mu_reference_for_L6.tsv")
    check(len(grid) == len(refs) == 192, "192 matched CE/GCE conditions and L8 references")
    check(len({(r["U"], r["beta"], r["Ntot"]) for r in grid}) == 192, "condition keys unique")
    check(Counter(r["U"] for r in grid) == Counter({"-5.0": 40, "-3.0": 40, "0.0": 40, "3.0": 36, "5.0": 36}), "U/beta condition counts")
    check(all(int(r["Nup"]) == int(r["Ndn"]) and int(r["Nup"]) + int(r["Ndn"]) == int(r["Ntot"]) for r in grid), "balanced canonical sectors")
    check(all(abs(float(r["T"]) - 1 / float(r["beta"])) < 5e-12 for r in grid), "actual T=1/beta")
    mapping = {12: 22, 18: 32, 26: 46, 32: 56}
    check(all(int(r["L8_reference_Ntot"]) == mapping[int(r["L6_Ntot_target"])] for r in refs), "L6-to-L8 filling map")
    check(all(math.isfinite(float(r["mu_L8_reference"])) for r in refs), "all L8 reference mu values finite")
    check(sum(r["reference_status"] == "final_production" for r in refs) == 152, "152 interacting references from final L8 production")
    check(sum(r["reference_status"] == "exact_recomputed" for r in refs) == 40, "40 exact U=0 references")

    ce_names = [
        "ce_L6_attractive_beta_le10_r32_m50000.tsv",
        "ce_L6_attractive_beta20_r64_m30000.tsv",
        "ce_L6_positive_beta_le4_r32_m50000.tsv",
        "ce_L6_positive_pilot_beta5_6p7_10_r32_m10000.tsv",
    ]
    ce = [r for name in ce_names for r in rows(name)]
    check([len(rows(n)) for n in ce_names] == [72, 8, 48, 24], "CE stage manifest counts 72/8/48/24")
    check(len({r["outdir"] for r in ce}) == 152, "152 unique fresh CE roots")
    check(all(r["account"] == "ccsd" for r in ce), "all CE rows pinned to ccsd")
    check(all(int(r["measure_interval"]) == 3 for r in ce), "CE measurement interval 3")
    check(all(int(r["Lx"]) == int(r["Ly"]) == 6 and int(r["cluster_size"]) == 36 for r in ce), "CE L=6 PBC geometry")
    positives = [r for r in ce if float(r["U"]) > 0]
    check(all(r["phase_reweighted"] == "true" and r["force_symmetry"] == "false" for r in positives), "positive-U CE phase reweighting and no forced symmetry")
    check(all("Up4" not in r["outdir"] and "ce_sign_L6_Up4" not in r["outdir"] for r in ce), "historical L6 U=+4 sign roots not reused")
    # The per-rank seed shift is 1,000,003; stay inside signed Int32 at 64 ranks.
    check(max(int(r["seed"]) + (int(r["expected_ranks"]) - 1) * 1_000_003 for r in ce) < 2_147_483_647, "CE rank seeds fit Int32")

    probe_names = [
        "gce_mu_probe_L6_attractive_L8seed_pm0p02.tsv",
        "gce_mu_probe_L6_positive_spinHS_L8seed_pm0p02.tsv",
    ]
    probes = [r for name in probe_names for r in rows(name)]
    check([len(rows(n)) for n in probe_names] == [240, 216], "GCE initial probe counts 240/216")
    check(len({r["out_parent"] for r in probes}) == 456, "456 unique GCE probe roots")
    check(all(r["account"] == "ccsd" and int(r["nupdates"]) == 3 for r in probes), "GCE probes pinned to ccsd with interval 3")
    grouped: dict[str, list[dict[str, str]]] = {}
    for r in probes:
        grouped.setdefault(r["target_key"], []).append(r)
    check(len(grouped) == 152 and all(len(v) == 3 for v in grouped.values()), "three L8-centered probes for every interacting condition")
    check(all({round(float(x["probe_offset"]), 8) for x in v} == {-0.02, 0.0, 0.02} for v in grouped.values()), "initial probe offsets are mu8 +/-0.02")
    check(all(int(r["L8_reference_Ntot"]) == mapping[int(r["Ntot_target"])] for r in probes), "probe L8 provenance mapping")

    exact_snapshot = RESULTS / "data" / "exact_u0_L6_snapshot.tsv"
    exact_root = RESULTS / "exact_u0"
    with exact_snapshot.open(newline="") as f:
        exact_rows = list(csv.DictReader(f, delimiter="\t"))
    check(len(exact_rows) == 80, "80 exact U=0 ensemble rows")
    exact_dirs = [p.parent for p in exact_root.rglob("exact_complete.txt")]
    check(len(exact_dirs) == 80, "80 deterministic exact U=0 roots")
    required = [
        "equal_time_observables_exact.tsv", "equal_time_charge_spin_wedge_exact.tsv",
        "equal_time_structure_factors_exact.tsv", "equal_time_neighbor_shells_exact.tsv",
    ]
    check(all((p / "exact_complete.txt").is_file() and all((p / f).is_file() for f in required) for p in exact_dirs), "all exact roots have four tables and completion marker")
    exact_status_snapshot = HERE / "status_source" / "exact_u0_L6_snapshot.tsv"
    with exact_status_snapshot.open(newline="") as f:
        check(len(list(csv.DictReader(f, delimiter="\t"))) == 80, "self-contained 80-row exact snapshot for remote collection")

    combiner_path = ROOT / "scripts" / "interacting_qmc_ed" / "combine_ce_green_tau_space_rank_outputs.py"
    spec = importlib.util.spec_from_file_location("ce_combiner", combiner_path)
    module = importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(module)
    synthetic = [
        {"nsamples": "10", "phase_sum": "2", "x": "3", "x_signed_sum": "6", "xe": "0.1", "phase_reweighted": "true"},
        {"nsamples": "30", "phase_sum": "12", "x": "2", "x_signed_sum": "24", "xe": "0.1", "phase_reweighted": "true"},
    ]
    mean, err, n = module.combine_phase_reweighted_rows(synthetic, "x", "xe")
    check(abs(mean - 30 / 14) < 1e-14 and n == 40 and math.isfinite(err), "global signed-numerator/phase-denominator pooling regression")

    text = (HERE / "run_ce_manifest_task.sh").read_text()
    check("--phase-reweight=true" in text and "--force-symmetry=false" in text and "--use-charge-hs=false" in text, "spin-channel positive-U CE command flags")
    check("CHECKPOINT_RESET_ACCUMULATORS=false" in (HERE / "run_gce_production_manifest_task.sh").read_text(), "GCE accumulation reset disabled")
    runner_text = "\n".join((HERE / name).read_text() for name in (
        "run_ce_manifest_task.sh", "run_gce_mu_manifest_task.sh", "run_gce_production_manifest_task.sh"
    ))
    check('exit "${rc:-1}"' not in runner_text and 'if [[ "${rc}" -eq 0 ]]; then rc=1; fi' in runner_text,
          "incomplete roots cannot exit successfully when a driver returns rc=0")
    exporter = (ROOT / "scripts" / "interacting_qmc_ed" / "export_dqmc_equal_time_observables.py").read_text()
    check("average_sign" in exporter and "global_stats.csv" in exporter, "GCE exports retain sign diagnostics and support aggregated final tables")
    collector = (HERE / "collect_l6_results.py").read_text()
    check("global rank phase-sum pool" in collector and "achieved N=" in collector and "duplicate run root" in collector,
          "strict collector checks phase pooling, achieved density, and duplicate roots")
    analysis = (HERE / "analyze_l6_thermometry.py").read_text()
    check('(tgce - tce) / tce' in analysis and 'state == "unique_solution" else np.nan' in analysis,
          "thermometry uses T_CE denominator and line-breaking NaNs at non-unique points")
    deck = (HERE / "build_l6_43slide_deck.mjs").read_text()
    check("@oai/artifact-tool" in deck and "assets.length !== 14" in deck and "expected 43 slides" in deck,
          "artifact-tool deck stage inserts exactly 14 L=6 slides into a 43-slide core")
    finalize = (HERE / "finalize_l6_results_local.sh").read_text()
    check("$HOME/.venvs/myenv/bin/python" in finalize and "slides_test.py" in finalize and "CE_GCE_no_icloud/results" in finalize,
          "finalizer uses requested Python environment, slide QA, and no_icloud mirror")
    print("WORKFLOW_VALIDATION PASS")


if __name__ == "__main__":
    main()
