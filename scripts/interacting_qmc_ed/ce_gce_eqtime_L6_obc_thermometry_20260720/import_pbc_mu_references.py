#!/usr/bin/env python3
"""Freeze confirmed L=6 PBC chemical potentials and create OBC probes."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


HERE = Path(__file__).resolve().parent
MANIFESTS = HERE / "manifests"
DEFAULT_PBC = Path("/home/9pm/nUHubbard/scripts/interacting_qmc_ed/ce_gce_eqtime_L6_thermometry_20260719/manifests")
REMOTE_RUNS = Path("/home/9pm/nUHubbard_obc_runs")
WORKFLOW_TAG = "20260720"


def read(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def stable_int(label: str, base: int) -> int:
    return base + int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "big") % 80_000_000


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def beta_label(beta: float) -> str:
    return f"b{beta:.1f}".replace(".", "p")


def mu_label(mu: float) -> str:
    return ("m" if mu < 0 else "p") + f"{abs(mu):.6f}".replace(".", "p")


def key(u: float | str, n: int | str, beta: float | str) -> tuple[float, int, float]:
    return round(float(u), 10), int(n), round(float(beta), 10)


def load_targets(path: Path) -> dict[tuple[float, int, float], dict[str, str]]:
    rows = read(path)
    result = {key(r["U"], r["Ntot_target"], r["beta"]): r for r in rows}
    if len(rows) != 152 or len(result) != 152:
        raise ValueError(f"expected 152 unique OBC targets, got rows={len(rows)} unique={len(result)}")
    return result


def accepted_pbc_rows(paths: list[Path]) -> list[dict[str, str]]:
    accepted: list[dict[str, str]] = []
    for path in paths:
        if not path.is_file():
            continue
        sha = file_sha256(path)
        for row in read(path):
            if row.get("tuning_status") != "confirmed_within_abs_N_0p03":
                continue
            accepted.append({**row, "pbc_reference_manifest": str(path), "pbc_reference_manifest_sha256": sha})
    return accepted


def reference_row(target: dict[str, str], pbc: dict[str, str]) -> dict[str, object]:
    return {
        "target_key": target["target_key"], "family": target["family"],
        "U_label": target["U_label"], "U": target["U"],
        "Ntot_target": target["Ntot_target"], "target_density": target["target_density"],
        "beta": target["beta"], "T": target["T"],
        "mu_L6_PBC_reference": pbc["mu_final"],
        "mu_L8_reference": pbc.get("mu_L8_reference", target["mu_L8_reference"]),
        "L8_reference_Ntot": pbc.get("L8_reference_Ntot", target["L8_reference_Ntot"]),
        "L8_reference_density": pbc.get("L8_reference_density", target["L8_reference_density"]),
        "pbc_reference_manifest": pbc["pbc_reference_manifest"],
        "pbc_reference_manifest_sha256": pbc["pbc_reference_manifest_sha256"],
        "pbc_reference_row_idx": pbc.get("idx", ""),
        "pbc_tuning_status": pbc["tuning_status"],
        "pbc_confirmation_N": pbc.get("confirmation_N", ""),
        "pbc_confirmation_N_err": pbc.get("confirmation_N_err", ""),
        "pbc_probe_bracket": pbc.get("probe_bracket", ""),
        "pbc_source_manifests": pbc.get("source_manifests", ""),
        "import_status": "frozen_confirmed_L6_PBC_mu",
        **{k: target[k] for k in target if k in {
            "boundary", "project_commit", "smoqydqmc_version", "smoqydqmc_commit",
            "canensafqmc_base_commit", "canensafqmc_current_patch_sha256",
            "canensafqmc_obc_patch_sha256", "site_count", "nn_bond_count",
            "nnn_bond_count", "kinetic_normalization", "double_occupancy_normalization",
            "nn_normalization", "nnn_normalization",
        }},
    }


def probe_rows(reference: dict[str, object]) -> list[dict[str, object]]:
    u, n, beta = float(reference["U"]), int(reference["Ntot_target"]), float(reference["beta"])
    center = float(reference["mu_L6_PBC_reference"])
    family = str(reference["family"])
    out: list[dict[str, object]] = []
    for offset, role in ((-0.02, "minus_0p02"), (0.0, "center"), (0.02, "plus_0p02")):
        mu = center + offset
        label = mu_label(mu)
        identity = f"L6OBC:GCEprobe:{family}:{u}:{n}:{beta}:{mu:.12f}"
        root = REMOTE_RUNS / f"gce_mu_L6_obc_thermometry_{WORKFLOW_TAG}" / family / str(reference["U_label"]) / (
            f"Ntot{n:03d}_n{n/36:.6f}_{beta_label(beta)}_mu{label}_{role}"
        )
        out.append({
            "idx": 0, "target_key": reference["target_key"], "family": family,
            "U_label": reference["U_label"], "U": reference["U"],
            "Ntot_target": n, "target_density": reference["target_density"],
            "beta": reference["beta"], "T": reference["T"],
            "mu_L6_PBC_reference": reference["mu_L6_PBC_reference"],
            "mu_L8_reference": reference["mu_L8_reference"],
            "L8_reference_Ntot": reference["L8_reference_Ntot"],
            "L8_reference_density": reference["L8_reference_density"],
            "mu_probe": f"{mu:.12f}", "mu_label": label, "probe_offset": f"{offset:.12f}",
            "probe_role": role, "probe_generation": "initial_pbc_seed_pm0p02",
            "Lx": 6, "Ly": 6, "dtau": "0.1", "ntherm": 2000,
            "nmeasurements": 10000, "nbins": 20, "nupdates": 3,
            "account": "ccsd", "partition": "burst", "qos": "default",
            "out_parent": str(root), "sid": stable_int(identity + ":sid", 920_720_000),
            "seed": stable_int(identity + ":seed", 2_020_720_000),
            "tuning_status": "initial_OBC_probe_from_confirmed_L6_PBC_mu",
            "pbc_reference_manifest": reference["pbc_reference_manifest"],
            "pbc_reference_manifest_sha256": reference["pbc_reference_manifest_sha256"],
            "job_tag": f"muL6O{reference['U_label']}N{n}{beta_label(beta)}{role}",
            **{k: reference[k] for k in reference if k in {
                "boundary", "project_commit", "smoqydqmc_version", "smoqydqmc_commit",
                "canensafqmc_base_commit", "canensafqmc_current_patch_sha256",
                "canensafqmc_obc_patch_sha256", "site_count", "nn_bond_count",
                "nnn_bond_count", "kinetic_normalization", "double_occupancy_normalization",
                "nn_normalization", "nnn_normalization",
            }},
        })
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=MANIFESTS / "gce_L6_obc_interacting_targets.tsv")
    parser.add_argument("--pbc-attractive", type=Path, default=DEFAULT_PBC / "gce_prod_L6_attractive_confirmed.tsv")
    parser.add_argument("--pbc-spin", type=Path, default=DEFAULT_PBC / "gce_prod_L6_spinHS_confirmed.tsv")
    parser.add_argument("--manifest-dir", type=Path, default=MANIFESTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = load_targets(args.targets)
    pbc_rows = accepted_pbc_rows([args.pbc_attractive, args.pbc_spin])
    existing_path = args.manifest_dir / "L6_PBC_mu_reference_for_OBC.tsv"
    frozen = {key(r["U"], r["Ntot_target"], r["beta"]): r for r in read(existing_path)}
    imported = 0
    for pbc in pbc_rows:
        k = key(pbc["U"], pbc["Ntot_target"], pbc["beta"])
        if k not in targets:
            raise ValueError(f"confirmed PBC row does not map to an OBC target: {k}")
        candidate = reference_row(targets[k], pbc)
        if k in frozen:
            if abs(float(frozen[k]["mu_L6_PBC_reference"]) - float(candidate["mu_L6_PBC_reference"])) > 5e-13:
                raise ValueError(f"frozen PBC mu changed for {k}; refusing silent root drift")
            continue
        frozen[k] = candidate
        imported += 1
    references = [frozen[k] for k in sorted(frozen)]
    for idx, row in enumerate(references):
        row["idx"] = idx
    write(existing_path, references)
    probes = [probe for reference in references for probe in probe_rows(reference)]
    families = {
        "attractive": [r for r in probes if r["family"] == "attractive"],
        "spinHS": [r for r in probes if r["family"] == "spinHS"],
    }
    for family, rows in families.items():
        for idx, row in enumerate(rows):
            row["idx"] = idx
        write(args.manifest_dir / f"gce_mu_probe_L6_obc_{family}_from_PBC.tsv", rows)
    print(f"confirmed PBC references available={len(references)}/152 newly_imported={imported}")
    print(f"OBC initial probes attractive={len(families['attractive'])} spinHS={len(families['spinHS'])}")


if __name__ == "__main__":
    main()
