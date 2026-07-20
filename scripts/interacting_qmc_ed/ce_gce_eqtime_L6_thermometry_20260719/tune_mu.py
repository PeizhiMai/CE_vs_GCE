#!/usr/bin/env python3.11
"""Summarize L=6 GCE density probes and generate safe next-stage manifests.

The initial probes are centered on the authoritative L=8 production chemical
potential.  A target advances only after every currently planned probe is
complete.  Missing brackets expand symmetrically (0.02 -> 0.04 -> 0.08 ...),
while bracketed targets receive a fresh confirmation probe at the secant
estimate.  A production row is emitted only after a confirmation satisfies
|N_GCE-N_target| <= 0.03.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


PROBE_FIELDS = [
    "idx", "U_label", "U", "Ntot_target", "target_density", "beta", "T",
    "mu_L8_reference", "L8_reference_Ntot", "L8_reference_density", "mu_probe",
    "mu_label", "probe_offset", "probe_role", "Lx", "Ly", "dtau", "ntherm",
    "nmeasurements", "nbins", "nupdates", "account", "out_parent", "sid", "seed",
    "target_key", "tuning_status", "source_manifests", "job_tag",
]

PROD_FIELDS = [
    "idx", "U_label", "U", "Ntot_target", "target_density", "beta", "T",
    "mu_L8_reference", "L8_reference_Ntot", "L8_reference_density", "probe_bracket",
    "mu_bracket_low", "density_bracket_low", "mu_bracket_high", "density_bracket_high",
    "mu_fitted", "mu_final", "mu_label", "confirmation_density", "confirmation_density_err",
    "confirmation_N", "confirmation_N_err", "density_tolerance", "tuning_status",
    "source_manifests", "Lx", "Ly", "dtau", "ntherm", "nmeasurements", "nbins",
    "nupdates", "expected_ranks", "account", "out_parent", "sid", "seed", "target_key",
    "job_tag",
]


def read_tsvs(paths: Iterable[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                missing = [k for k in PROBE_FIELDS if k not in row]
                if missing:
                    raise ValueError(f"{path}: missing probe fields {missing}")
                rec = dict(row)
                rec["_manifest"] = str(path)
                rows.append(rec)
    roots = [r["out_parent"] for r in rows]
    if len(roots) != len(set(roots)):
        duplicates = [x for x, n in Counter(roots).items() if n > 1]
        raise ValueError(f"duplicate GCE probe roots across manifests: {duplicates[:5]}")
    return rows


def family(row: dict[str, str]) -> str:
    return "attractive" if float(row["U"]) < 0 else "spinHS"


def run_prefix(row: dict[str, str]) -> str:
    return "attractive_hubbard_rect" if family(row) == "attractive" else "hubbard_spin_hs_rect"


def run_base(row: dict[str, str]) -> str:
    return (
        f"{run_prefix(row)}_U{float(row['U']):.2f}_tp0.00_mu{float(row['mu_probe']):.2f}"
        f"_Lx{int(row['Lx'])}_Ly{int(row['Ly'])}_b{float(row['beta']):.2f}-{int(row['sid'])}"
    )


def read_global_stats(path: Path) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    with path.open() as f:
        for row in csv.DictReader(f, delimiter=" ", skipinitialspace=True):
            key = row.get("MEASUREMENT")
            if key:
                out[key] = (float(row["MEAN_REAL"]), float(row.get("STD") or 0.0))
    return out


def inspect(row: dict[str, str]) -> dict[str, object]:
    parent = Path(row["out_parent"])
    base = run_base(row)
    cdir = parent / f"complete_{base}"
    idir = parent / base
    gfile = cdir / "global_stats.csv"
    density = density_err = sign = sign_err = math.nan
    if gfile.is_file():
        stats = read_global_stats(gfile)
        density, density_err = stats.get("density", (math.nan, math.nan))
        sign, sign_err = stats.get("sgn", (math.nan, math.nan))
    complete = cdir.is_dir() and gfile.is_file() and math.isfinite(density)
    return {
        **row,
        "family": family(row),
        "complete": complete,
        "complete_dir": str(cdir) if cdir.is_dir() else "",
        "incomplete_dir": str(idir) if idir.is_dir() else "",
        "checkpoint_count": len(list(idir.glob("checkpoint_pID-*.jld2"))) if idir.is_dir() else 0,
        "density": density,
        "density_err": density_err,
        "N_mean": density * 36 if math.isfinite(density) else math.nan,
        "N_err": density_err * 36 if math.isfinite(density_err) else math.nan,
        "sign": sign,
        "sign_err": sign_err,
    }


def stable_int(label: str, base: int, span: int = 90_000_000) -> int:
    n = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "big")
    return base + n % span


def mu_label(mu: float) -> str:
    return ("m" if mu < 0 else "p") + f"{abs(mu):.6f}".replace(".", "p")


def beta_label(beta: float) -> str:
    return f"{beta:.1f}".replace(".", "p")


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, delimiter="\t", fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def choose_bracket(points: list[dict[str, object]], target: float):
    pts = sorted(points, key=lambda x: float(x["mu_probe"]))
    candidates = []
    for a, b in zip(pts[:-1], pts[1:]):
        da = float(a["density"]) - target
        db = float(b["density"]) - target
        if da == 0:
            return a, a, float(a["mu_probe"]), 0.0, "exact_probe"
        if da * db <= 0:
            dmu = float(b["mu_probe"]) - float(a["mu_probe"])
            dd = float(b["density"]) - float(a["density"])
            if abs(dd) < 1e-8 or abs(dmu) < 1e-12:
                continue
            mu = float(a["mu_probe"]) + (target - float(a["density"])) * dmu / dd
            slope = dd / dmu
            mu_err = math.hypot(float(a["density_err"]), float(b["density_err"])) / abs(slope)
            candidates.append((abs(dmu), abs(mu - (float(a["mu_probe"]) + float(b["mu_probe"])) / 2), a, b, mu, mu_err))
    if not candidates:
        return None
    _, _, a, b, mu, mu_err = min(candidates, key=lambda x: (x[0], x[1]))
    return a, b, mu, mu_err, "secant_bracket"


def make_probe(template: dict[str, object], mu: float, role: str, status: str, ordinal: int) -> dict[str, object]:
    key = str(template["target_key"])
    fam = str(template["family"])
    beta = float(template["beta"])
    ulabel = str(template["U_label"])
    ntot = int(template["Ntot_target"])
    mlab = mu_label(mu)
    sid = stable_int(f"L6mu:{key}:{role}:{mu:.12f}", 1_020_000_000 if fam == "attractive" else 1_120_000_000)
    seed = stable_int(f"L6museed:{key}:{role}:{mu:.12f}", 1_220_000_000 if fam == "attractive" else 1_320_000_000)
    parent = (
        f"/home/9pm/nUHubbard/runs/gce_mu_L6_L8seed_20260719/{fam}/{ulabel}/"
        f"Ntot{ntot:03d}_n{float(template['target_density']):.6f}_b{beta_label(beta)}_mu{mlab}_{role}"
    )
    offset = mu - float(template["mu_L8_reference"])
    return {
        **{k: template[k] for k in PROBE_FIELDS if k in template},
        "idx": ordinal,
        "mu_probe": f"{mu:.12f}",
        "mu_label": mlab,
        "probe_offset": f"{offset:.12f}",
        "probe_role": role,
        "out_parent": parent,
        "sid": sid,
        "seed": seed,
        "tuning_status": status,
        "job_tag": f"muL6{ulabel}N{ntot}b{beta_label(beta)}{role}",
        "nupdates": 3,
        "account": "ccsd",
    }


def make_production(template: dict[str, object], confirmation: dict[str, object], bracket, ordinal: int) -> dict[str, object]:
    a, b, mu_fit, _, _ = bracket
    mu = float(confirmation["mu_probe"])
    fam = str(template["family"])
    ulabel = str(template["U_label"])
    ntot = int(template["Ntot_target"])
    beta = float(template["beta"])
    mlab = mu_label(mu)
    sid = stable_int(f"L6prod:{template['target_key']}:{mu:.12f}", 1_420_000_000 if fam == "attractive" else 1_520_000_000)
    seed = stable_int(f"L6prodseed:{template['target_key']}:{mu:.12f}", 1_620_000_000 if fam == "attractive" else 1_720_000_000)
    parent = (
        f"/home/9pm/nUHubbard/runs/gce_eqtime_L6_thermometry_20260719/{fam}/{ulabel}/"
        f"Ntot{ntot:03d}_n{float(template['target_density']):.6f}_b{beta_label(beta)}_mu{mlab}_r32_w5000_m50000"
    )
    return {
        "idx": ordinal,
        "U_label": ulabel,
        "U": template["U"],
        "Ntot_target": ntot,
        "target_density": template["target_density"],
        "beta": template["beta"],
        "T": template["T"],
        "mu_L8_reference": template["mu_L8_reference"],
        "L8_reference_Ntot": template["L8_reference_Ntot"],
        "L8_reference_density": template["L8_reference_density"],
        "probe_bracket": f"{float(a['mu_probe']):.12f}:{float(b['mu_probe']):.12f}",
        "mu_bracket_low": f"{float(a['mu_probe']):.12f}",
        "density_bracket_low": f"{float(a['density']):.12f}",
        "mu_bracket_high": f"{float(b['mu_probe']):.12f}",
        "density_bracket_high": f"{float(b['density']):.12f}",
        "mu_fitted": f"{mu_fit:.12f}",
        "mu_final": f"{mu:.12f}",
        "mu_label": mlab,
        "confirmation_density": f"{float(confirmation['density']):.12f}",
        "confirmation_density_err": f"{float(confirmation['density_err']):.12f}",
        "confirmation_N": f"{float(confirmation['N_mean']):.12f}",
        "confirmation_N_err": f"{float(confirmation['N_err']):.12f}",
        "density_tolerance": "0.03",
        "tuning_status": "confirmed_within_abs_N_0p03",
        "source_manifests": template["source_manifests"],
        "Lx": 6, "Ly": 6, "dtau": "0.1",
        "ntherm": 5000, "nmeasurements": 50000, "nbins": 100, "nupdates": 3,
        "expected_ranks": 32, "account": "ccsd", "out_parent": parent,
        "sid": sid, "seed": seed, "target_key": template["target_key"],
        "job_tag": f"gceL6{ulabel}N{ntot}b{beta_label(beta)}",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", nargs="+", type=Path)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--manifest-dir", type=Path, required=True)
    ap.add_argument("--write-next", action="store_true")
    ap.add_argument("--max-half-width", type=float, default=1.28)
    args = ap.parse_args()

    inspected = [inspect(r) for r in read_tsvs(args.manifest)]
    args.outdir.mkdir(parents=True, exist_ok=True)
    task_fields = PROBE_FIELDS + [
        "family", "complete", "checkpoint_count", "density", "density_err", "N_mean", "N_err",
        "sign", "sign_err", "complete_dir", "incomplete_dir", "_manifest",
    ]
    write_tsv(args.outdir / "mu_probe_task_status.tsv", inspected, task_fields)

    by: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in inspected:
        by[str(row["target_key"])].append(row)

    summaries: list[dict[str, object]] = []
    next_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    productions: dict[str, list[dict[str, object]]] = defaultdict(list)
    for key, rows in sorted(by.items()):
        template = rows[0]
        target = float(template["target_density"])
        target_N = int(template["Ntot_target"])
        complete = [r for r in rows if bool(r["complete"])]
        all_complete = len(complete) == len(rows)
        bracket = choose_bracket(complete, target)
        confirmations = sorted(
            [r for r in complete if str(r["probe_role"]).startswith("confirmation")],
            key=lambda r: str(r["probe_role"]),
        )
        accepted = [r for r in confirmations if abs(float(r["N_mean"]) - target_N) <= 0.03]
        status = "waiting_for_planned_probes"
        mu_fit = math.nan
        if accepted and bracket is not None:
            confirmation = min(accepted, key=lambda r: abs(float(r["N_mean"]) - target_N))
            status = "confirmed_within_abs_N_0p03"
            prod = make_production(template, confirmation, bracket, len(productions[str(template["family"])]))
            productions[str(template["family"])].append(prod)
            mu_fit = float(bracket[2])
        elif all_complete and bracket is not None:
            mu_fit = float(bracket[2])
            existing_mu = [float(r["mu_probe"]) for r in rows]
            # A missed confirmation is itself a useful point; generate the newly fitted value.
            if any(abs(mu_fit - x) < 5e-7 for x in existing_mu):
                status = "flat_or_duplicate_confirmation_no_safe_new_point"
            else:
                round_no = 1 + len(confirmations)
                role = f"confirmation_r{round_no}"
                next_rows[str(template["family"])].append(
                    make_probe(template, mu_fit, role, "secant_confirmation", len(next_rows[str(template["family"])]))
                )
                status = "confirmation_required"
        elif all_complete:
            offsets = [abs(float(r["probe_offset"])) for r in rows]
            width = max(offsets) if offsets else 0.02
            next_width = 0.04 if width < 0.04 - 1e-12 else width * 2
            if next_width > args.max_half_width + 1e-12:
                status = "unbracketed_max_width_reached"
            else:
                mu8 = float(template["mu_L8_reference"])
                for sign, label in [(-1, "minus"), (1, "plus")]:
                    role = f"extension_{label}_{str(next_width).replace('.', 'p')}"
                    next_rows[str(template["family"])].append(
                        make_probe(template, mu8 + sign * next_width, role, f"symmetric_extension_pm{next_width:g}", len(next_rows[str(template["family"])]))
                    )
                status = f"needs_symmetric_extension_pm{next_width:g}"

        summaries.append({
            "target_key": key, "family": template["family"], "U_label": template["U_label"],
            "U": template["U"], "Ntot_target": target_N, "target_density": target,
            "beta": template["beta"], "T": template["T"],
            "mu_L8_reference": template["mu_L8_reference"],
            "L8_reference_Ntot": template["L8_reference_Ntot"],
            "planned_probes": len(rows), "complete_probes": len(complete),
            "bracketed": int(bracket is not None), "mu_fitted": mu_fit,
            "confirmations_complete": len(confirmations), "status": status,
        })

    summary_fields = list(summaries[0].keys()) if summaries else []
    if summaries:
        write_tsv(args.outdir / "mu_target_status.tsv", summaries, summary_fields)
    print("TASKS", len(inspected), "complete", sum(bool(r["complete"]) for r in inspected))
    print("TARGETS", len(summaries), dict(Counter(str(r["status"]) for r in summaries)))

    if args.write_next:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        for fam in ("attractive", "spinHS"):
            if next_rows[fam]:
                path = args.manifest_dir / f"gce_mu_probe_L6_{fam}_followup_{stamp}.tsv"
                write_tsv(path, next_rows[fam], PROBE_FIELDS)
                print(f"wrote follow-up {path} ({len(next_rows[fam])} rows)")
            if productions[fam]:
                path = args.manifest_dir / f"gce_prod_L6_{fam}_confirmed.tsv"
                write_tsv(path, productions[fam], PROD_FIELDS)
                print(f"wrote production {path} ({len(productions[fam])} rows)")


if __name__ == "__main__":
    main()
