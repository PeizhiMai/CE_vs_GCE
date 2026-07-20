#!/usr/bin/env python3
"""Generate deterministic manifests for L=6 OBC CE/GCE thermometry."""

from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
from pathlib import Path

from exact_u0_l6_obc import BETAS, L6_NTOTALS, SMOQY_COMMIT, SMOQY_VERSION


HERE = Path(__file__).resolve().parent
MANIFESTS = HERE / "manifests"
PBC_WORKFLOW = HERE.parent / "ce_gce_eqtime_L6_thermometry_20260719"
PBC_L8_REFERENCE = PBC_WORKFLOW / "manifests" / "L8_mu_reference_for_L6.tsv"
REMOTE_CODE = Path("/home/9pm/nUHubbard_obc_dev")
REMOTE_RUNS = Path("/home/9pm/nUHubbard_obc_runs")
WORKFLOW_TAG = "20260720"
U_VALUES = (-5.0, -3.0, 0.0, 3.0, 5.0)
POSITIVE_BETAS = tuple(beta for beta in BETAS if beta != 20.0)
CANENS_BASE_COMMIT = "21b4f6815d0b836973064ff8401fb2ba9c23b802"
CANENS_CURRENT_PATCH = "bf35357dce7b29e5d14bae12539aad4b1c8ebc05726a619facced251c076d1c9"
CANENS_OBC_PATCH = "8e65be6f9f9bb010d444b45847b77cf56835b937cec54f62d794b25daf77a4c0"


def git_head() -> str:
    root = HERE.parents[2]
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty manifest {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def u_label(u: float) -> str:
    if u == 0:
        return "U0"
    return ("Um" if u < 0 else "Up") + str(int(abs(u)))


def beta_label(beta: float) -> str:
    return f"b{beta:.1f}".replace(".", "p")


def stable_int(label: str, base: int) -> int:
    return base + int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "big") % 80_000_000


def provenance(commit: str, *, lx: int = 6, ly: int = 6) -> dict[str, object]:
    return {
        "boundary": "open",
        "project_commit": commit,
        "smoqydqmc_version": SMOQY_VERSION,
        "smoqydqmc_commit": SMOQY_COMMIT,
        "canensafqmc_base_commit": CANENS_BASE_COMMIT,
        "canensafqmc_current_patch_sha256": CANENS_CURRENT_PATCH,
        "canensafqmc_obc_patch_sha256": CANENS_OBC_PATCH,
        "site_count": lx * ly,
        "nn_bond_count": ly * (lx - 1) + lx * (ly - 1),
        "nnn_bond_count": 2 * (lx - 1) * (ly - 1),
        "kinetic_normalization": "physical hopping matrix / site_count",
        "double_occupancy_normalization": "sites / site_count",
        "nn_normalization": "existing undirected physical NN bonds / nn_bond_count",
        "nnn_normalization": "existing undirected physical NNN bonds / nnn_bond_count",
    }


def load_l8_reference(path: Path) -> dict[tuple[float, int, float], dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing inherited L=8 reference table: {path}")
    rows: dict[tuple[float, int, float], dict[str, str]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            key = (float(row["U"]), int(row["L6_Ntot_target"]), float(row["beta"]))
            if key in rows:
                raise ValueError(f"duplicate L8 reference key {key}")
            rows[key] = row
    if len(rows) != 192:
        raise ValueError(f"expected 192 inherited L8 reference rows, found {len(rows)}")
    return rows


def full_grid(reference: dict[tuple[float, int, float], dict[str, str]], commit: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for u in U_VALUES:
        betas = BETAS if u <= 0 else POSITIVE_BETAS
        for ntotal in L6_NTOTALS:
            for beta in betas:
                ref = reference[(u, ntotal, beta)]
                rows.append({
                    "idx": len(rows), "target_key": f"{u_label(u)}_L6OBC_N{ntotal:03d}_{beta_label(beta)}",
                    "U_label": u_label(u), "U": f"{u:.1f}", "beta": f"{beta:.1f}",
                    "T": f"{1.0 / beta:.12f}", "Ntot": ntotal, "Nup": ntotal // 2,
                    "Ndn": ntotal // 2, "target_density": f"{ntotal / 36.0:.12f}",
                    "L8_reference_Ntot": ref["L8_reference_Ntot"],
                    "L8_reference_density": ref["L8_reference_density"],
                    "mu_L8_reference": ref["mu_L8_reference"],
                    "l8_source_manifests": ref.get("source_manifests", ""),
                    "ce_method": "exact" if u == 0 else "QMC",
                    "gce_method": "exact" if u == 0 else "QMC_mu_tuned_for_OBC_from_L6_PBC",
                    "stage": "exact_u0" if u == 0 else "interacting",
                    **provenance(commit),
                })
    if len(rows) != 192:
        raise ValueError(f"expected 192 physical conditions, got {len(rows)}")
    return rows


def ce_row(condition: dict[str, object], stage: str, idx: int, warmups: int, measurements: int, ranks: int) -> dict[str, object]:
    u, beta, ntotal = float(condition["U"]), float(condition["beta"]), int(condition["Ntot"])
    seed = stable_int(f"L6OBC:CE:{stage}:{u}:{ntotal}:{beta}", 2_020_720_000)
    root = REMOTE_RUNS / f"ce_eqtime_L6_obc_thermometry_{WORKFLOW_TAG}" / stage / u_label(u) / (
        f"Ntot{ntotal:03d}_n{ntotal/36:.6f}_{beta_label(beta)}_T{1/beta:.6f}_"
        f"Nup{ntotal//2:03d}_Ndn{ntotal//2:03d}_r{ranks}_w{warmups}_m{measurements}_seed{seed}"
    )
    return {
        "idx": idx, "target_key": condition["target_key"], "stage": stage,
        "U_label": u_label(u), "U": f"{u:.1f}", "beta": f"{beta:.1f}",
        "target_T": f"{1/beta:.12f}", "actual_T": f"{1/beta:.12f}",
        "Ltau": int(round(beta / 0.1)), "Ntot": ntotal, "Nup": ntotal // 2,
        "Ndn": ntotal // 2, "density": f"{ntotal/36:.12f}", "account": "ccsd",
        "partition": "burst", "qos": "default", "job_tag": f"ceL6O{u_label(u)}N{ntotal}{beta_label(beta)}",
        "outdir": str(root), "seed": seed, "nwarmups": warmups,
        "max_batches": measurements, "batch_nsamples": 1, "measure_interval": 3,
        "cluster_size": 36, "Lx": 6, "Ly": 6, "dtau": "0.1",
        "expected_ranks": ranks, "phase_reweighted": str(u > 0).lower(),
        "force_symmetry": "false" if u > 0 else "driver_default",
        **provenance(str(condition["project_commit"])),
    }


def ce_manifests(grid: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    attr32: list[dict[str, object]] = []
    attr64: list[dict[str, object]] = []
    poslow: list[dict[str, object]] = []
    pilots: list[dict[str, object]] = []
    for condition in grid:
        u, beta = float(condition["U"]), float(condition["beta"])
        if u < 0 and beta < 20:
            attr32.append(ce_row(condition, "attractive", len(attr32), 5000, 50000, 32))
        elif u < 0 and beta == 20:
            attr64.append(ce_row(condition, "attractive_beta20", len(attr64), 5000, 30000, 64))
        elif u > 0 and beta <= 4:
            poslow.append(ce_row(condition, "positive_lowbeta", len(poslow), 5000, 50000, 32))
        elif u > 0:
            pilots.append(ce_row(condition, "positive_pilot", len(pilots), 2000, 10000, 32))
    counts = len(attr32), len(attr64), len(poslow), len(pilots)
    if counts != (72, 8, 48, 24):
        raise ValueError(f"bad CE manifest counts {counts}")
    return {
        "ce_L6_obc_attractive_beta_le10_r32_m50000.tsv": attr32,
        "ce_L6_obc_attractive_beta20_r64_m30000.tsv": attr64,
        "ce_L6_obc_positive_beta_le4_r32_m50000.tsv": poslow,
        "ce_L6_obc_positive_pilot_beta5_6p7_10_r32_m10000.tsv": pilots,
    }


def interacting_targets(grid: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for condition in grid:
        if float(condition["U"]) == 0:
            continue
        rows.append({
            "idx": len(rows), "target_key": condition["target_key"],
            "family": "attractive" if float(condition["U"]) < 0 else "spinHS",
            "U_label": condition["U_label"], "U": condition["U"], "Ntot_target": condition["Ntot"],
            "target_density": condition["target_density"], "beta": condition["beta"], "T": condition["T"],
            "mu_L8_reference": condition["mu_L8_reference"],
            "L8_reference_Ntot": condition["L8_reference_Ntot"],
            "L8_reference_density": condition["L8_reference_density"],
            "pbc_mu_import_status": "waiting_for_confirmed_L6_PBC_mu",
            "mu_L6_PBC_reference": "", "pbc_reference_manifest": "",
            **provenance(str(condition["project_commit"])),
        })
    if len(rows) != 152:
        raise ValueError(f"expected 152 interacting GCE targets, got {len(rows)}")
    return rows


def smoke_manifests(commit: str) -> dict[str, list[dict[str, object]]]:
    ce: list[dict[str, object]] = []
    gce: list[dict[str, object]] = []
    for idx, u in enumerate((-3.0, 3.0)):
        ul = u_label(u)
        ce.append({
            "idx": idx, "target_key": f"smoke_ce_{ul}", "stage": "smoke", "U_label": ul,
            "U": f"{u:.1f}", "beta": "0.2", "target_T": "5.0", "actual_T": "5.0",
            "Ltau": 2, "Ntot": 2, "Nup": 1, "Ndn": 1, "density": "0.5",
            "account": "ccsd", "partition": "burst", "qos": "default",
            "job_tag": f"smokeCeL6O{ul}",
            "outdir": str(REMOTE_RUNS / "ce_gce_eqtime_L6_obc_thermometry_20260720_smoke" / "ce" / ul),
            "seed": 2_020_720_100 + idx, "nwarmups": 1, "max_batches": 4,
            "batch_nsamples": 1, "measure_interval": 1, "cluster_size": 4,
            "Lx": 2, "Ly": 2, "dtau": "0.1", "expected_ranks": 2,
            "phase_reweighted": str(u > 0).lower(), "force_symmetry": "false" if u > 0 else "driver_default",
            **provenance(commit, lx=2, ly=2),
        })
        family = "attractive" if u < 0 else "spinHS"
        gce.append({
            "idx": idx, "target_key": f"smoke_gce_{ul}", "family": family,
            "U_label": ul, "U": f"{u:.1f}", "Ntot_target": 2,
            "target_density": "0.5", "beta": "0.2", "T": "5.0",
            "mu_L8_reference": "0.0", "mu_L6_PBC_reference": "0.0",
            "L8_reference_Ntot": 2, "L8_reference_density": "0.5",
            "mu_probe": "0.0", "mu_label": "p0p000000", "probe_offset": "0.0",
            "probe_role": "smoke", "Lx": 2, "Ly": 2, "dtau": "0.1",
            "ntherm": 1, "nmeasurements": 4, "nbins": 1, "nupdates": 1,
            "account": "ccsd", "partition": "burst", "qos": "default",
            "out_parent": str(REMOTE_RUNS / "ce_gce_eqtime_L6_obc_thermometry_20260720_smoke" / "gce" / family),
            "sid": 920_720_100 + idx, "seed": 2_020_720_200 + idx,
            "tuning_status": "smoke_only", "pbc_reference_manifest": "smoke_only",
            "job_tag": f"smokeMuL6O{ul}", **provenance(commit, lx=2, ly=2),
        })
    return {"smoke_ce_L6_obc_two_rank.tsv": ce, "smoke_gce_L6_obc_two_rank.tsv": gce}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--l8-reference", type=Path, default=PBC_L8_REFERENCE)
    parser.add_argument("--manifest-dir", type=Path, default=MANIFESTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    commit = git_head()
    reference = load_l8_reference(args.l8_reference)
    grid = full_grid(reference, commit)
    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(args.manifest_dir / "L6_obc_full_condition_grid.tsv", grid)
    write_tsv(args.manifest_dir / "gce_L6_obc_interacting_targets.tsv", interacting_targets(grid))
    for name, rows in ce_manifests(grid).items():
        write_tsv(args.manifest_dir / name, rows)
    for name, rows in smoke_manifests(commit).items():
        write_tsv(args.manifest_dir / name, rows)
    print(f"generated L=6 OBC manifests at project commit {commit}")
    print("grid=192; CE=72 attr r32 + 8 attr r64 + 48 positive low-beta + 24 pilots")
    print("GCE targets=152 waiting for confirmed L=6 PBC mu imports")


if __name__ == "__main__":
    main()
