#!/usr/bin/env python3.11
"""Generate the deterministic L=6 thermometry manifests."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

from exact_u0_l6 import BETAS, L6_NTOTALS, exact_gce_mu


HERE = Path(__file__).resolve().parent
MANIFESTS = HERE / "manifests"
REFERENCE_SOURCES = HERE / "status_source" / "l8_reference_sources"
REMOTE_PROJECT = Path("/home/9pm/nUHubbard")

U_VALUES = (-5.0, -3.0, 0.0, 3.0, 5.0)
POSITIVE_BETAS = tuple(b for b in BETAS if b != 20.0)
L8_N_MAP = {12: 22, 18: 32, 26: 46, 32: 56}


def fmt_label(value: float, digits: int = 6) -> str:
    return ("m" if value < 0 else "p") + f"{abs(value):.{digits}f}".replace(".", "p")


def beta_label(beta: float) -> str:
    return f"b{beta:.1f}".replace(".", "p")


def u_label(u: float) -> str:
    if u == 0:
        return "U0"
    return ("Um" if u < 0 else "Up") + str(int(abs(u)))


def write_tsv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty manifest {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_l8_interacting_reference() -> dict[tuple[float, int, float], dict]:
    grouped: dict[tuple[float, int, float], list[dict]] = defaultdict(list)
    for path in sorted(REFERENCE_SOURCES.glob("*gce_prod*.tsv")):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if not {"U", "Ntot_target", "beta", "mu_final"}.issubset(row):
                    continue
                key = (float(row["U"]), int(row["Ntot_target"]), float(row["beta"]))
                grouped[key].append({
                    "mu": float(row["mu_final"]),
                    "source_manifest": path.name,
                    "source_out_parent": row.get("out_parent", ""),
                })
    result = {}
    for key, rows in grouped.items():
        mus = {round(r["mu"], 12) for r in rows}
        if len(mus) != 1:
            raise ValueError(f"conflicting L8 production mu values for {key}: {sorted(mus)}")
        result[key] = {
            "mu": rows[0]["mu"],
            "source_manifests": ";".join(sorted({r["source_manifest"] for r in rows})),
            "source_out_parents": ";".join(sorted({r["source_out_parent"] for r in rows if r["source_out_parent"]})),
        }
    expected = {
        (u, n, b)
        for u in (-5.0, -3.0, 3.0, 5.0)
        for n in (22, 32, 46, 56)
        for b in (BETAS if u < 0 else POSITIVE_BETAS)
    }
    missing = sorted(expected - result.keys())
    extra = sorted(result.keys() - expected)
    if missing or extra:
        raise ValueError(f"L8 reference coverage error: missing={missing[:10]} extra={extra[:10]}")
    if len(result) != 152:
        raise ValueError(f"expected 152 interacting references, found {len(result)}")
    return result


def build_reference_rows(interacting: dict[tuple[float, int, float], dict]) -> list[dict]:
    rows = []
    idx = 0
    for u in U_VALUES:
        betas = BETAS if u <= 0 else POSITIVE_BETAS
        for l6_n in L6_NTOTALS:
            l8_n = L8_N_MAP[l6_n]
            for beta in betas:
                if u == 0:
                    mu = exact_gce_mu(8, 8, beta, l8_n)
                    manifests = "exact_U0_L8_PBC_recomputed_20260719"
                    out_parents = ""
                else:
                    ref = interacting[(u, l8_n, beta)]
                    mu = ref["mu"]
                    manifests = ref["source_manifests"]
                    out_parents = ref["source_out_parents"]
                rows.append({
                    "idx": idx,
                    "U_label": u_label(u),
                    "U": f"{u:.1f}",
                    "beta": f"{beta:.1f}",
                    "T": f"{1.0 / beta:.12f}",
                    "L6_Ntot_target": l6_n,
                    "L6_target_density": f"{l6_n / 36.0:.12f}",
                    "L8_reference_Ntot": l8_n,
                    "L8_reference_density": f"{l8_n / 64.0:.12f}",
                    "mu_L8_reference": f"{mu:.12f}",
                    "source_manifests": manifests,
                    "source_out_parents": out_parents,
                    "reference_status": "final_production" if u else "exact_recomputed",
                })
                idx += 1
    if len(rows) != 192:
        raise ValueError(f"expected 192 reference rows, found {len(rows)}")
    return rows


def build_condition_rows(reference_rows: list[dict]) -> list[dict]:
    rows = []
    for idx, ref in enumerate(reference_rows):
        u, beta, ntotal = float(ref["U"]), float(ref["beta"]), int(ref["L6_Ntot_target"])
        rows.append({
            "idx": idx,
            "U_label": ref["U_label"],
            "U": ref["U"],
            "beta": ref["beta"],
            "T": ref["T"],
            "Ntot": ntotal,
            "Nup": ntotal // 2,
            "Ndn": ntotal // 2,
            "density": ref["L6_target_density"],
            "L8_reference_Ntot": ref["L8_reference_Ntot"],
            "L8_reference_density": ref["L8_reference_density"],
            "mu_L8_reference": ref["mu_L8_reference"],
            "ce_method": "exact" if u == 0 else "QMC",
            "gce_method": "exact" if u == 0 else "QMC_mu_tuned_from_L8",
            "stage": "exact_u0" if u == 0 else "interacting",
        })
    return rows


def ce_row(ref: dict, stage: str, idx: int, warmups: int, measurements: int, ranks: int) -> dict:
    u, beta = float(ref["U"]), float(ref["beta"])
    ntotal = int(ref["L6_Ntot_target"])
    seed = 1907190000 + idx + {"attractive": 0, "positive_lowbeta": 1000, "positive_pilot": 2000}[stage]
    bl = beta_label(beta)
    ul = ref["U_label"]
    suffix = f"r{ranks}_w{warmups}_m{measurements}_seed{seed}"
    outdir = REMOTE_PROJECT / "runs" / "ce_eqtime_L6_thermometry_20260719" / stage / ul / (
        f"Ntot{ntotal:03d}_n{ntotal/36.0:.6f}_{bl}_T{1/beta:.6f}_Nup{ntotal//2:03d}_Ndn{ntotal//2:03d}_{suffix}"
    )
    return {
        "idx": idx,
        "stage": stage,
        "U_label": ul,
        "U": f"{u:.1f}",
        "beta": f"{beta:.1f}",
        "target_T": f"{1/beta:.12f}",
        "actual_T": f"{1/beta:.12f}",
        "Ltau": int(round(beta / 0.1)),
        "Ntot": ntotal,
        "Nup": ntotal // 2,
        "Ndn": ntotal // 2,
        "density": f"{ntotal/36.0:.12f}",
        "account": "ccsd",
        "job_tag": f"ceL6{ul}N{ntotal}{bl}{'Pilot' if stage == 'positive_pilot' else ''}",
        "outdir": str(outdir),
        "seed": seed,
        "nwarmups": warmups,
        "max_batches": measurements,
        "batch_nsamples": 1,
        "measure_interval": 3,
        "cluster_size": 36,
        "Lx": 6,
        "Ly": 6,
        "dtau": "0.1",
        "expected_ranks": ranks,
        "phase_reweighted": str(u > 0).lower(),
        "force_symmetry": "false" if u > 0 else "driver_default",
    }


def build_ce_manifests(reference_rows: list[dict]) -> dict[str, list[dict]]:
    attractive_r32, attractive_r64, positive_low, positive_pilot = [], [], [], []
    for ref in reference_rows:
        u, beta = float(ref["U"]), float(ref["beta"])
        if u < 0 and beta < 20:
            attractive_r32.append(ce_row(ref, "attractive", len(attractive_r32), 5000, 50000, 32))
        elif u < 0 and beta == 20:
            attractive_r64.append(ce_row(ref, "attractive", len(attractive_r64), 5000, 30000, 64))
        elif u > 0 and beta <= 4:
            positive_low.append(ce_row(ref, "positive_lowbeta", len(positive_low), 5000, 50000, 32))
        elif u > 0 and beta in (5.0, 6.7, 10.0):
            positive_pilot.append(ce_row(ref, "positive_pilot", len(positive_pilot), 2000, 10000, 32))
    expected = (len(attractive_r32), len(attractive_r64), len(positive_low), len(positive_pilot))
    if expected != (72, 8, 48, 24):
        raise ValueError(f"unexpected CE manifest sizes {expected}")
    return {
        "ce_L6_attractive_beta_le10_r32_m50000.tsv": attractive_r32,
        "ce_L6_attractive_beta20_r64_m30000.tsv": attractive_r64,
        "ce_L6_positive_beta_le4_r32_m50000.tsv": positive_low,
        "ce_L6_positive_pilot_beta5_6p7_10_r32_m10000.tsv": positive_pilot,
    }


def probe_row(ref: dict, idx: int, offset: float, family: str) -> dict:
    u, beta = float(ref["U"]), float(ref["beta"])
    ntotal = int(ref["L6_Ntot_target"])
    mu8 = float(ref["mu_L8_reference"])
    mu = mu8 + offset
    role = "center" if offset == 0 else "minus_0p02" if offset < 0 else "plus_0p02"
    ul, bl, ml = ref["U_label"], beta_label(beta), fmt_label(mu)
    sid = (916000000 if family == "attractive" else 917000000) + idx
    seed = (1907193000 if family == "attractive" else 1907194000) + idx
    target_key = f"{ul}_L6_N{ntotal:03d}_{bl}"
    out_parent = REMOTE_PROJECT / "runs" / "gce_mu_L6_L8seed_20260719" / family / ul / (
        f"Ntot{ntotal:03d}_n{ntotal/36.0:.6f}_{bl}_mu{ml}_{role}"
    )
    return {
        "idx": idx,
        "U_label": ul,
        "U": f"{u:.1f}",
        "Ntot_target": ntotal,
        "target_density": f"{ntotal/36.0:.12f}",
        "beta": f"{beta:.1f}",
        "T": f"{1/beta:.12f}",
        "mu_L8_reference": f"{mu8:.12f}",
        "L8_reference_Ntot": ref["L8_reference_Ntot"],
        "L8_reference_density": ref["L8_reference_density"],
        "mu_probe": f"{mu:.12f}",
        "mu_label": ml,
        "probe_offset": f"{offset:.2f}",
        "probe_role": role,
        "Lx": 6,
        "Ly": 6,
        "dtau": "0.1",
        "ntherm": 2000,
        "nmeasurements": 10000,
        "nbins": 20,
        "nupdates": 3,
        "account": "ccsd",
        "out_parent": str(out_parent),
        "sid": sid,
        "seed": seed,
        "target_key": target_key,
        "tuning_status": "initial_L8_seed_pm0p02",
        "source_manifests": ref["source_manifests"],
        "job_tag": f"muL6{ul}N{ntotal}{bl}{role}",
    }


def build_probe_manifests(reference_rows: list[dict]) -> dict[str, list[dict]]:
    attractive, positive = [], []
    for ref in reference_rows:
        u = float(ref["U"])
        if u == 0:
            continue
        target = attractive if u < 0 else positive
        family = "attractive" if u < 0 else "spinHS"
        for offset in (-0.02, 0.0, 0.02):
            target.append(probe_row(ref, len(target), offset, family))
    if (len(attractive), len(positive)) != (240, 216):
        raise ValueError(f"unexpected GCE probe sizes {(len(attractive), len(positive))}")
    return {
        "gce_mu_probe_L6_attractive_L8seed_pm0p02.tsv": attractive,
        "gce_mu_probe_L6_positive_spinHS_L8seed_pm0p02.tsv": positive,
    }


def build_smoke_manifests() -> dict[str, list[dict]]:
    ce_rows = []
    for idx, u in enumerate((-3.0, 3.0)):
        ul = u_label(u)
        ce_rows.append({
            "idx": idx, "stage": "smoke", "U_label": ul, "U": f"{u:.1f}",
            "beta": "0.2", "target_T": "5.000000000000", "actual_T": "5.000000000000",
            "Ltau": 2, "Ntot": 2, "Nup": 1, "Ndn": 1, "density": "0.500000000000",
            "account": "ccsd", "job_tag": f"smokeCeL6{ul}",
            "outdir": str(REMOTE_PROJECT / "runs" / "ce_gce_eqtime_L6_thermometry_20260719_smoke" / "ce" / ul),
            "seed": 1_900_000_000 + idx, "nwarmups": 1, "max_batches": 2,
            "batch_nsamples": 1, "measure_interval": 1, "cluster_size": 4,
            "Lx": 2, "Ly": 2, "dtau": "0.1", "expected_ranks": 2,
            "phase_reweighted": str(u > 0).lower(),
            "force_symmetry": "false" if u > 0 else "driver_default",
        })

    probe_rows = []
    for idx, u in enumerate((-3.0, 3.0)):
        ul = u_label(u); family = "attractive" if u < 0 else "spinHS"
        probe_rows.append({
            "idx": idx, "U_label": ul, "U": f"{u:.1f}", "Ntot_target": 2,
            "target_density": "0.500000000000", "beta": "0.2", "T": "5.000000000000",
            "mu_L8_reference": "0.000000000000", "L8_reference_Ntot": 32,
            "L8_reference_density": "0.500000000000", "mu_probe": "0.000000000000",
            "mu_label": "p0p000000", "probe_offset": "0.00", "probe_role": "smoke",
            "Lx": 2, "Ly": 2, "dtau": "0.1", "ntherm": 1, "nmeasurements": 2,
            "nbins": 1, "nupdates": 1, "account": "ccsd",
            "out_parent": str(REMOTE_PROJECT / "runs" / "ce_gce_eqtime_L6_thermometry_20260719_smoke" / "gce" / family),
            "sid": 918_000_000 + idx, "seed": 1_910_000_000 + idx,
            "target_key": f"smoke_{ul}", "tuning_status": "smoke_only",
            "source_manifests": "smoke_only", "job_tag": f"smokeMuL6{ul}",
        })
    return {
        "smoke_ce_L6_two_rank.tsv": ce_rows,
        "smoke_gce_mu_L6_two_rank.tsv": probe_rows,
    }


def main() -> None:
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    interacting = load_l8_interacting_reference()
    refs = build_reference_rows(interacting)
    write_tsv(MANIFESTS / "L8_mu_reference_for_L6.tsv", refs)
    write_tsv(MANIFESTS / "L6_full_condition_grid.tsv", build_condition_rows(refs))
    for name, rows in build_ce_manifests(refs).items():
        write_tsv(MANIFESTS / name, rows)
    for name, rows in build_probe_manifests(refs).items():
        write_tsv(MANIFESTS / name, rows)
    for name, rows in build_smoke_manifests().items():
        write_tsv(MANIFESTS / name, rows)
    print("generated L=6 workflow manifests")
    print("  reference conditions: 192")
    print("  CE immediate QMC: 72 attractive r32 + 8 attractive r64 + 48 positive low-beta + 24 pilots")
    print("  GCE initial probes: 240 attractive + 216 positive spin-HS")


if __name__ == "__main__":
    main()
