#!/usr/bin/env python3
"""Advance L=6 OBC GCE density probes to brackets, confirmations, and production."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


PROVENANCE_FIELDS = [
    "boundary", "project_commit", "smoqydqmc_version", "smoqydqmc_commit",
    "canensafqmc_base_commit", "canensafqmc_current_patch_sha256",
    "canensafqmc_obc_patch_sha256", "site_count", "nn_bond_count", "nnn_bond_count",
    "kinetic_normalization", "double_occupancy_normalization", "nn_normalization",
    "nnn_normalization",
]
PROBE_FIELDS = [
    "idx", "target_key", "family", "U_label", "U", "Ntot_target", "target_density",
    "beta", "T", "mu_L6_PBC_reference", "mu_L8_reference", "L8_reference_Ntot",
    "L8_reference_density", "mu_probe", "mu_label", "probe_offset", "probe_role",
    "probe_generation", "Lx", "Ly", "dtau", "ntherm", "nmeasurements", "nbins",
    "nupdates", "account", "partition", "qos", "out_parent", "sid", "seed",
    "tuning_status", "pbc_reference_manifest", "pbc_reference_manifest_sha256", "job_tag",
    *PROVENANCE_FIELDS,
]
PROD_FIELDS = [
    "idx", "target_key", "family", "U_label", "U", "Ntot_target", "target_density",
    "beta", "T", "mu_L6_PBC_reference", "mu_L8_reference", "L8_reference_Ntot",
    "L8_reference_density", "probe_bracket", "mu_bracket_low", "density_bracket_low",
    "mu_bracket_high", "density_bracket_high", "mu_fitted", "mu_final", "mu_label",
    "confirmation_density", "confirmation_density_err", "confirmation_N", "confirmation_N_err",
    "density_tolerance", "tuning_status", "pbc_reference_manifest",
    "pbc_reference_manifest_sha256", "source_probe_manifests", "Lx", "Ly", "dtau",
    "ntherm", "nmeasurements", "nbins", "nupdates", "expected_ranks", "account",
    "partition", "qos", "out_parent", "sid", "seed", "job_tag", "production_attempt",
    *PROVENANCE_FIELDS,
]


def read_tsvs(paths: Iterable[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                missing = [field for field in PROBE_FIELDS if field not in row]
                if missing:
                    raise ValueError(f"{path}: missing fields {missing}")
                rows.append({**row, "_manifest": str(path)})
    roots = [row["out_parent"] for row in rows]
    duplicates = [root for root, count in Counter(roots).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate OBC probe roots: {duplicates[:5]}")
    return rows


def read_first(path: Path) -> dict[str, str]:
    with path.open(newline="") as handle:
        return next(csv.DictReader(handle, delimiter="\t"))


def read_global_stats(path: Path) -> dict[str, tuple[float, float]]:
    stats: dict[str, tuple[float, float]] = {}
    with path.open() as handle:
        for row in csv.DictReader(handle, delimiter=" ", skipinitialspace=True):
            if row.get("MEASUREMENT"):
                stats[row["MEASUREMENT"]] = (float(row["MEAN_REAL"]), float(row.get("STD") or 0.0))
    return stats


def inspect(row: dict[str, str]) -> dict[str, object]:
    parent = Path(row["out_parent"])
    achieved = parent / "probe_achieved_density.tsv"
    complete_dirs = sorted(parent.glob("complete_*_obc_*"))
    density = density_err = nmean = nerr = sign = signerr = math.nan
    complete = len(complete_dirs) == 1 and achieved.is_file()
    if complete:
        a = read_first(achieved)
        nmean, nerr = float(a["achieved_N"]), float(a["achieved_N_err"])
        sites = int(row["site_count"])
        density, density_err = nmean / sites, nerr / sites
        stats = read_global_stats(complete_dirs[0] / "global_stats.csv")
        sign, signerr = stats.get("sgn", stats.get("sign", (math.nan, math.nan)))
    incomplete_dirs = [p for p in parent.iterdir()] if parent.is_dir() else []
    incomplete_dirs = [p for p in incomplete_dirs if p.is_dir() and not p.name.startswith("complete_")]
    checkpoint_count = sum(len(list(path.glob("checkpoint_pID-*.jld2"))) for path in incomplete_dirs)
    return {
        **row, "complete": complete, "complete_dir": str(complete_dirs[0]) if complete_dirs else "",
        "complete_mtime": achieved.stat().st_mtime if complete else math.nan,
        "checkpoint_count": checkpoint_count, "density": density, "density_err": density_err,
        "N_mean": nmean, "N_err": nerr, "sign": sign, "sign_err": signerr,
    }


def stable_int(label: str, base: int) -> int:
    return base + int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "big") % 80_000_000


def mu_label(mu: float) -> str:
    return ("m" if mu < 0 else "p") + f"{abs(mu):.6f}".replace(".", "p")


def beta_label(beta: float) -> str:
    return f"b{beta:.1f}".replace(".", "p")


def write(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def choose_bracket(points: list[dict[str, object]], target_density: float):
    ordered = sorted(points, key=lambda row: float(row["mu_probe"]))
    candidates = []
    for low, high in zip(ordered[:-1], ordered[1:]):
        dl = float(low["density"]) - target_density
        dh = float(high["density"]) - target_density
        if dl == 0:
            return low, low, float(low["mu_probe"]), 0.0
        if dl * dh <= 0:
            dmu = float(high["mu_probe"]) - float(low["mu_probe"])
            ddensity = float(high["density"]) - float(low["density"])
            # Particle number must rise with chemical potential.  A crossing
            # produced only by statistical non-monotonicity is not a valid
            # tuning bracket.
            if abs(dmu) < 1e-12 or ddensity < 1e-8:
                continue
            mu = float(low["mu_probe"]) + (target_density - float(low["density"])) * dmu / ddensity
            slope = ddensity / dmu
            muerr = math.hypot(float(low["density_err"]), float(high["density_err"])) / abs(slope)
            candidates.append((abs(dmu), low, high, mu, muerr))
    if not candidates:
        return None
    _, low, high, mu, muerr = min(candidates, key=lambda item: item[0])
    return low, high, mu, muerr


def make_probe(template: dict[str, object], mu: float, role: str, generation: str, ordinal: int) -> dict[str, object]:
    family = str(template["family"]); beta = float(template["beta"]); n = int(template["Ntot_target"])
    ulabel = str(template["U_label"]); mlab = mu_label(mu)
    identity = f"L6OBC:probe:{template['target_key']}:{role}:{mu:.12f}"
    parent = (
        f"/home/9pm/nUHubbard_obc_runs/gce_mu_L6_obc_thermometry_20260720/{family}/{ulabel}/"
        f"Ntot{n:03d}_n{float(template['target_density']):.6f}_{beta_label(beta)}_mu{mlab}_{role}"
    )
    return {
        **{field: template[field] for field in PROBE_FIELDS if field in template},
        "idx": ordinal, "mu_probe": f"{mu:.12f}", "mu_label": mlab,
        "probe_offset": f"{mu-float(template['mu_L6_PBC_reference']):.12f}",
        "probe_role": role, "probe_generation": generation, "out_parent": parent,
        "sid": stable_int(identity + ":sid", 920_720_000),
        "seed": stable_int(identity + ":seed", 2_020_720_000),
        "tuning_status": generation, "account": "ccsd", "partition": "burst", "qos": "default",
        "job_tag": f"muL6O{ulabel}N{n}{beta_label(beta)}{role}",
    }


def make_production(
    template: dict[str, object], confirmation: dict[str, object], bracket,
    ordinal: int, source_manifests: list[str], production_attempt: int = 1,
) -> dict[str, object]:
    low, high, mu_fit, _ = bracket
    mu = float(confirmation["mu_probe"]); family = str(template["family"])
    beta = float(template["beta"]); n = int(template["Ntot_target"]); ulabel = str(template["U_label"])
    mlab = mu_label(mu); identity = f"L6OBC:production:{template['target_key']}:{mu:.12f}:attempt{production_attempt}"
    parent = (
        f"/home/9pm/nUHubbard_obc_runs/gce_eqtime_L6_obc_thermometry_20260720/{family}/{ulabel}/"
        f"Ntot{n:03d}_n{float(template['target_density']):.6f}_{beta_label(beta)}_mu{mlab}_r32_w5000_m50000_attempt{production_attempt}"
    )
    return {
        "idx": ordinal, "target_key": template["target_key"], "family": family,
        "U_label": ulabel, "U": template["U"], "Ntot_target": n,
        "target_density": template["target_density"], "beta": template["beta"], "T": template["T"],
        "mu_L6_PBC_reference": template["mu_L6_PBC_reference"],
        "mu_L8_reference": template["mu_L8_reference"],
        "L8_reference_Ntot": template["L8_reference_Ntot"],
        "L8_reference_density": template["L8_reference_density"],
        "probe_bracket": f"{float(low['mu_probe']):.12f}:{float(high['mu_probe']):.12f}",
        "mu_bracket_low": f"{float(low['mu_probe']):.12f}",
        "density_bracket_low": f"{float(low['density']):.12f}",
        "mu_bracket_high": f"{float(high['mu_probe']):.12f}",
        "density_bracket_high": f"{float(high['density']):.12f}",
        "mu_fitted": f"{float(mu_fit):.12f}", "mu_final": f"{mu:.12f}", "mu_label": mlab,
        "confirmation_density": f"{float(confirmation['density']):.12f}",
        "confirmation_density_err": f"{float(confirmation['density_err']):.12f}",
        "confirmation_N": f"{float(confirmation['N_mean']):.12f}",
        "confirmation_N_err": f"{float(confirmation['N_err']):.12f}",
        "density_tolerance": "0.03", "tuning_status": "confirmed_within_abs_N_0p03",
        "pbc_reference_manifest": template["pbc_reference_manifest"],
        "pbc_reference_manifest_sha256": template["pbc_reference_manifest_sha256"],
        "source_probe_manifests": ",".join(source_manifests),
        "Lx": 6, "Ly": 6, "dtau": "0.1", "ntherm": 5000,
        "nmeasurements": 50000, "nbins": 100, "nupdates": 3, "expected_ranks": 32,
        "account": "ccsd", "partition": "burst", "qos": "default", "out_parent": parent,
        "sid": stable_int(identity + ":sid", 1_420_720_000),
        "seed": stable_int(identity + ":seed", 1_620_720_000),
        "job_tag": f"gceL6O{ulabel}N{n}{beta_label(beta)}",
        "production_attempt": production_attempt,
        **{field: template[field] for field in PROVENANCE_FIELDS},
    }


def failed_productions(manifest_dir: Path) -> dict[str, dict[str, object]]:
    """Return the latest out-of-tolerance production attempt per target."""
    failures: dict[str, dict[str, object]] = {}
    paths = sorted(manifest_dir.glob("gce_prod_L6_obc_*_confirmed.tsv"))
    paths += sorted((Path(__file__).resolve().parent / "status_source" / "submission_manifests").glob("gce_prod_L6_obc_*_delta_*.tsv"))
    paths += sorted((Path(__file__).resolve().parent / "status_source" / "repair_manifests").glob("gceL6O*_repair_*.tsv"))
    for path in paths:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                parent = Path(row["out_parent"])
                marker = parent / "density_tolerance_failed.txt"
                achieved = parent / "achieved_density.tsv"
                if not marker.is_file() or not achieved.is_file():
                    continue
                measured = read_first(achieved)
                candidate: dict[str, object] = {
                    **row,
                    "failed_mtime": marker.stat().st_mtime,
                    "failed_N": float(measured["achieved_N"]),
                    "failed_density": float(measured["achieved_N"]) / int(row["site_count"]),
                    "failed_mu": float(row["mu_final"]),
                    "production_attempt": int(row.get("production_attempt") or 1),
                }
                prior = failures.get(row["target_key"])
                if prior is None or float(candidate["failed_mtime"]) > float(prior["failed_mtime"]):
                    failures[row["target_key"]] = candidate
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="+", type=Path)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--write-next", action="store_true")
    parser.add_argument("--max-half-width", type=float, default=2.56)
    args = parser.parse_args()

    inspected = [inspect(row) for row in read_tsvs(args.manifest)]
    args.outdir.mkdir(parents=True, exist_ok=True)
    status_fields = PROBE_FIELDS + ["complete", "complete_mtime", "checkpoint_count", "density", "density_err", "N_mean", "N_err", "sign", "sign_err", "complete_dir", "_manifest"]
    write(args.outdir / "mu_probe_task_status.tsv", inspected, status_fields)
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in inspected:
        grouped[str(row["target_key"])].append(row)
    summaries: list[dict[str, object]] = []
    next_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    productions: dict[str, list[dict[str, object]]] = defaultdict(list)
    failures = failed_productions(args.manifest_dir)
    for target_key, rows in sorted(grouped.items()):
        template = rows[0]; target_density = float(template["target_density"]); target_n = int(template["Ntot_target"])
        complete = [row for row in rows if bool(row["complete"])]
        all_complete = len(complete) == len(rows)
        bracket = choose_bracket(complete, target_density)
        confirmations = [
            row for row in complete
            if str(row["probe_role"]).startswith(("confirmation", "production_retune"))
        ]
        failure = failures.get(target_key)
        if failure is not None:
            confirmations = [
                row for row in confirmations
                if str(row["probe_role"]).startswith("production_retune")
                and float(row["complete_mtime"]) > float(failure["failed_mtime"])
            ]
        accepted = [row for row in confirmations if abs(float(row["N_mean"])-target_n) <= 0.03]
        status = "waiting_for_planned_probes"; mu_fit = math.nan
        family = str(template["family"])
        if accepted and bracket is not None:
            confirmation = min(accepted, key=lambda row: abs(float(row["N_mean"])-target_n))
            source_manifests = sorted({str(row["_manifest"]) for row in rows})
            productions[family].append(make_production(
                template, confirmation, bracket, len(productions[family]), source_manifests,
                int(failure["production_attempt"])+1 if failure is not None else 1,
            ))
            status = "confirmed_within_abs_N_0p03"
            mu_fit = float(bracket[2])
        elif failure is not None and all_complete:
            # Fold the longer production density into the local monotonic
            # calibration and require a new independent confirmation.
            failed_point = {
                **template, "mu_probe": failure["failed_mu"],
                "density": failure["failed_density"], "density_err": 0.0,
            }
            retune_bracket = choose_bracket(complete + [failed_point], target_density)
            if retune_bracket is not None:
                mu_fit = float(retune_bracket[2])
            else:
                dmu = float(failure["mu_bracket_high"]) - float(failure["mu_bracket_low"])
                ddensity = float(failure["density_bracket_high"]) - float(failure["density_bracket_low"])
                if dmu <= 0 or ddensity <= 1e-8:
                    status = "production_density_failed_no_monotonic_retune_slope"
                    mu_fit = math.nan
                else:
                    slope = ddensity / dmu
                    mu_fit = float(failure["failed_mu"]) + (
                        target_density - float(failure["failed_density"])
                    ) / slope
            if math.isfinite(mu_fit):
                if any(abs(mu_fit-float(row["mu_probe"])) < 5e-7 for row in confirmations):
                    status = "production_density_failed_duplicate_retune_point"
                else:
                    role = f"production_retune_r{int(failure['production_attempt'])+1}"
                    next_rows[family].append(make_probe(
                        template, mu_fit, role, "retune_after_out_of_tolerance_production",
                        len(next_rows[family]),
                    ))
                    status = "production_density_failed_retune_confirmation_required"
        elif all_complete and bracket is not None:
            mu_fit = float(bracket[2])
            existing_confirmation = [
                float(row["mu_probe"]) for row in confirmations
            ]
            if any(abs(mu_fit-value) < 5e-7 for value in existing_confirmation):
                status = "flat_or_duplicate_confirmation_no_safe_new_point"
            else:
                round_no = len(confirmations)+1; role = f"confirmation_r{round_no}"
                next_rows[family].append(make_probe(template, mu_fit, role, "secant_confirmation", len(next_rows[family])))
                status = "confirmation_required"
        elif all_complete:
            center = float(template["mu_L6_PBC_reference"])
            width = max([abs(float(row["mu_probe"])-center) for row in rows] or [0.02])
            next_width = 0.04 if width < 0.04-1e-12 else width*2
            if next_width > args.max_half_width+1e-12:
                status = "unbracketed_max_width_reached"
            else:
                for sign,label in ((-1,"minus"),(1,"plus")):
                    role=f"extension_{label}_{str(next_width).replace('.', 'p')}"
                    next_rows[family].append(make_probe(template, center+sign*next_width, role, f"symmetric_extension_pm{next_width:g}", len(next_rows[family])))
                status=f"needs_symmetric_extension_pm{next_width:g}"
        summaries.append({
            "target_key": target_key, "family": family, "U_label": template["U_label"], "U": template["U"],
            "Ntot_target": target_n, "target_density": target_density, "beta": template["beta"], "T": template["T"],
            "mu_L6_PBC_reference": template["mu_L6_PBC_reference"], "mu_L8_reference": template["mu_L8_reference"],
            "planned_probes": len(rows), "complete_probes": len(complete), "bracketed": int(bracket is not None),
            "mu_fitted": mu_fit, "confirmations_complete": len(confirmations), "status": status,
        })
    if summaries:
        write(args.outdir / "mu_target_status.tsv", summaries, list(summaries[0]))
    print("TASKS",len(inspected),"complete",sum(bool(row["complete"]) for row in inspected))
    print("TARGETS",len(summaries),dict(Counter(str(row["status"]) for row in summaries)))
    if args.write_next:
        stamp=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        for family in ("attractive","spinHS"):
            if next_rows[family]:
                path=args.manifest_dir/f"gce_mu_probe_L6_obc_{family}_followup_{stamp}.tsv"
                write(path,next_rows[family],PROBE_FIELDS); print(f"wrote {path} ({len(next_rows[family])})")
            if productions[family]:
                path=args.manifest_dir/f"gce_prod_L6_obc_{family}_confirmed.tsv"
                write(path,productions[family],PROD_FIELDS); print(f"wrote {path} ({len(productions[family])})")


if __name__ == "__main__":
    main()
