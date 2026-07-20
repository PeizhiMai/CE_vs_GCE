#!/usr/bin/env python3
"""Generate the deterministic 2x2/3x3 OBC CE/GCE validation matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path


OBSERVABLES = (
    "kinetic_per_site",
    "double_occupancy_per_site",
    "nn_spin_s_s",
    "nn_connected_charge",
)
MATRIX = {
    2: {"sectors": ((1, 1), (2, 2)), "betas": (2.0, 5.0)},
    3: {"sectors": ((2, 2), (4, 4)), "betas": (5.0,)},
}
SMOQY_VERSION = "2.0.12"
SMOQY_OBC_COMMIT = "c5f0c81bc98029bae585e0cb283428e293553999"


def dependency_provenance() -> dict[str, str]:
    project = Path(__file__).resolve().parents[3]

    def commit(relative: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(project / relative), "rev-parse", "HEAD"],
            text=True,
        ).strip()

    def digest(relative: str) -> str:
        return hashlib.sha256((project / relative).read_bytes()).hexdigest()

    provenance = {
        "canensafqmc_base_commit": commit("external/CanEnsAFQMC"),
        "canensafqmc_current_response_patch_sha256": digest(
            "patches/CanEnsAFQMC-current-response.patch"
        ),
        "canensafqmc_obc_patch_sha256": digest("patches/CanEnsAFQMC-obc.patch"),
        "smoqydqmc_version": SMOQY_VERSION,
        "smoqydqmc_commit": commit("external/SmoQyDQMC"),
    }
    if provenance["smoqydqmc_commit"] != SMOQY_OBC_COMMIT:
        raise RuntimeError(
            "validation manifests require pinned SmoQyDQMC OBC commit "
            f"{SMOQY_OBC_COMMIT}, found {provenance['smoqydqmc_commit']}"
        )
    return provenance


def token(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty manifest {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for column in row:
            if column not in columns:
                columns.append(column)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=columns, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def reference_key(row: dict[str, str], ensemble: str) -> tuple[object, ...]:
    base = (
        int(row["L"]),
        float(row["U"]),
        float(row["beta"]),
        int(float(row["target_N"])),
    )
    if ensemble == "CE":
        return base + (int(row["nup"]), int(row["ndn"]))
    return base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ed-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--run-root",
        default="/home/9pm/nUHubbard_obc_runs/obc_ce_gce_validation_20260720",
    )
    parser.add_argument("--expected-ranks", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=2000)
    parser.add_argument("--measurements-per-rank", type=int, default=10000)
    parser.add_argument("--measurement-interval", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=2026072000)
    parser.add_argument(
        "--run-tag",
        default="",
        help="Optional filesystem-safe suffix for a fresh corrective validation generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_tag and not args.run_tag.replace("_", "").isalnum():
        raise ValueError("run-tag may contain only letters, digits, and underscores")
    provenance = dependency_provenance()
    if args.measurements_per_rank % 100:
        raise ValueError("measurements-per-rank must be divisible by 100")
    ce_refs = {
        reference_key(row, "CE"): row
        for row in read_tsv(args.ed_dir / "ce_ed_reference.tsv")
        if row.get("accepted", "").lower() == "true"
    }
    gce_refs = {
        reference_key(row, "GCE"): row
        for row in read_tsv(args.ed_dir / "gce_ed_reference.tsv")
        if row.get("accepted", "").lower() == "true"
    }
    ce_reference_path = args.ed_dir / "ce_ed_reference.tsv"
    gce_reference_path = args.ed_dir / "gce_ed_reference.tsv"
    ce_reference_sha256 = hashlib.sha256(ce_reference_path.read_bytes()).hexdigest()
    gce_reference_sha256 = hashlib.sha256(gce_reference_path.read_bytes()).hexdigest()

    ce_rows: list[dict[str, object]] = []
    gce_rows: list[dict[str, object]] = []
    condition_index = 0
    for L, specification in MATRIX.items():
        for U in (-3.0, 3.0):
            for beta in specification["betas"]:
                for nup, ndn in specification["sectors"]:
                    target_n = nup + ndn
                    ce_key = (L, U, beta, target_n, nup, ndn)
                    gce_key = (L, U, beta, target_n)
                    if ce_key not in ce_refs or gce_key not in gce_refs:
                        raise KeyError(
                            f"missing accepted ED reference CE={ce_key in ce_refs} "
                            f"GCE={gce_key in gce_refs} for L={L}, U={U}, beta={beta}, N={target_n}"
                        )
                    for dtau in (0.20, 0.10, 0.05):
                        ltau = round(beta / dtau)
                        if abs(ltau * dtau - beta) > 1e-12:
                            raise ValueError("validation beta is not commensurate with dtau")
                        for seed_index in (0, 1):
                            seed = args.base_seed + 1000 * condition_index + seed_index
                            stem = (
                                f"L{L}_U{token(U)}_b{token(beta)}_N{target_n}_"
                                f"dt{token(dtau)}_seed{seed_index}"
                            )
                            if args.run_tag:
                                stem = f"{stem}_{args.run_tag}"
                            common: dict[str, object] = {
                                "condition_index": condition_index,
                                "L": L,
                                "boundary": "open",
                                "t": 1.0,
                                "tprime": 0.0,
                                "U": U,
                                "beta": beta,
                                "temperature": 1.0 / beta,
                                "dtau": dtau,
                                "Ltau": ltau,
                                "nup": nup,
                                "ndn": ndn,
                                "target_N": target_n,
                                "seed_index": seed_index,
                                "seed": seed,
                                "expected_ranks": args.expected_ranks,
                                "warmups": args.warmups,
                                "measurements_per_rank": args.measurements_per_rank,
                                "measurement_interval": args.measurement_interval,
                                "account": "ccsd",
                                "partition": "burst",
                                "qos": "default",
                                "site_count": L * L,
                                "nn_bond_count": 2 * L * (L - 1),
                                "nnn_bond_count": 2 * (L - 1) ** 2,
                                "kinetic_normalization": "sum over physical hopping matrix / site_count",
                                "double_occupancy_normalization": "sum over sites / site_count",
                                "nn_normalization": "sum over existing undirected NN bonds / nn_bond_count",
                                "nnn_normalization": "sum over existing undirected NNN bonds / nnn_bond_count",
                                **provenance,
                            }
                            ce_reference = ce_refs[ce_key]
                            ce_row = {
                                "idx": len(ce_rows),
                                "run_id": f"ce_obc_{stem}",
                                "ensemble": "CE",
                                **common,
                                "phase_reweighted": str(U > 0).lower(),
                                "force_symmetry": str(U <= 0).lower(),
                                "ce_same_spin_estimator": "CanEnsAFQMC canonical two-body RDM rho2 per physical bond; never Wick-contract the projected one-body RDM",
                                "outdir": f"{args.run_root}/ce/ce_obc_{stem}",
                                "ed_reference_file": str(ce_reference_path),
                                "ed_reference_sha256": ce_reference_sha256,
                                "ed_tail_weight_bound": ce_reference["tail_weight_bound"],
                                "ed_dimension": ce_reference["dimension"],
                                "ed_states_kept": ce_reference["states_kept"],
                                "ed_method": ce_reference["ed_method"],
                                "ed_max_eigen_residual": ce_reference["max_eigen_residual"],
                            }
                            for observable in OBSERVABLES:
                                ce_row[f"ed_{observable}"] = ce_reference[observable]
                            ce_rows.append(ce_row)

                            gce_reference = gce_refs[gce_key]
                            gce_row = {
                                "idx": len(gce_rows),
                                "run_id": f"gce_obc_{stem}",
                                "ensemble": "GCE",
                                **common,
                                "family": "attractive" if U < 0 else "spin_hs",
                                "mu_ph_symmetric": gce_reference["mu_ph_symmetric"],
                                "ed_achieved_N": gce_reference["achieved_N"],
                                "density_tolerance": 0.01,
                                "sid": 820000 + len(gce_rows),
                                "out_parent": f"{args.run_root}/gce/gce_obc_{stem}",
                                "ed_reference_file": str(gce_reference_path),
                                "ed_reference_sha256": gce_reference_sha256,
                                "ed_tail_weight_bound": gce_reference["tail_weight_bound"],
                                "ed_states_kept": gce_reference["states_kept"],
                                "ed_full_dimension": gce_reference["full_dimension"],
                                "ed_max_eigen_residual": gce_reference["max_eigen_residual"],
                            }
                            for observable in OBSERVABLES:
                                gce_row[f"ed_{observable}"] = gce_reference[observable]
                            gce_rows.append(gce_row)
                    condition_index += 1

    if len(ce_rows) != 72 or len(gce_rows) != 72:
        raise AssertionError(f"unexpected matrix size CE={len(ce_rows)} GCE={len(gce_rows)}")
    if len({row["outdir"] for row in ce_rows}) != len(ce_rows):
        raise AssertionError("duplicate CE run root")
    if len({row["out_parent"] for row in gce_rows}) != len(gce_rows):
        raise AssertionError("duplicate GCE run root")

    write_tsv(args.outdir / "ce_validation_manifest.tsv", ce_rows)
    write_tsv(args.outdir / "gce_validation_manifest.tsv", gce_rows)
    summary = {
        "schema_version": 1,
        "boundary": "open",
        "ce_rows": len(ce_rows),
        "gce_rows": len(gce_rows),
        "physical_conditions_per_ensemble": condition_index,
        "dtau_values": [0.2, 0.1, 0.05],
        "seed_replicates": 2,
        "expected_ranks": args.expected_ranks,
        "warmups": args.warmups,
        "measurements_per_rank": args.measurements_per_rank,
        "run_root": args.run_root,
        "run_tag": args.run_tag,
        "ed_reference_dir": str(args.ed_dir),
        "ce_ed_reference_sha256": ce_reference_sha256,
        "gce_ed_reference_sha256": gce_reference_sha256,
    }
    (args.outdir / "manifest_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
