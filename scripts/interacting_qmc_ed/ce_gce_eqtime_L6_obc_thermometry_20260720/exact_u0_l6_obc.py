#!/usr/bin/env python3
"""Exact U=0 CE/GCE equal-time observables for a finite square OBC lattice.

The canonical trace uses elementary-symmetric-polynomial occupation moments in
the one-body eigenbasis.  It never enumerates the C(36, N) Slater determinants.
Real-space site and bond estimators are then reconstructed from the one- and
two-orbital occupation moments using the same x-fastest OBC geometry as the
CE/GCE production drivers.
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


BETAS = (2.0, 2.2, 2.5, 2.9, 3.3, 4.0, 5.0, 6.7, 10.0, 20.0)
L6_NTOTALS = (12, 18, 26, 32)
L = 6
NSITES = 36
NN_BONDS = 60
NNN_BONDS = 50
SMOQY_VERSION = "2.0.12"
SMOQY_COMMIT = "c5f0c81bc98029bae585e0cb283428e293553999"
CANENS_BASE_COMMIT = "21b4f6815d0b836973064ff8401fb2ba9c23b802"
CANENS_CURRENT_PATCH = "bf35357dce7b29e5d14bae12539aad4b1c8ebc05726a619facced251c076d1c9"
CANENS_OBC_PATCH = "8e65be6f9f9bb010d444b45847b77cf56835b937cec54f62d794b25daf77a4c0"


@dataclass(frozen=True)
class Geometry:
    lx: int
    ly: int
    hopping: np.ndarray
    nn_bonds: tuple[tuple[int, int], ...]
    nnn_bonds: tuple[tuple[int, int], ...]

    @property
    def nsites(self) -> int:
        return self.lx * self.ly


def site(x: int, y: int, lx: int) -> int:
    return x + y * lx


def build_geometry(lx: int, ly: int, *, t: float = 1.0, tprime: float = 0.0) -> Geometry:
    """Build x-fastest square OBC hopping and existing physical bond lists."""
    nn: list[tuple[int, int]] = []
    nnn: list[tuple[int, int]] = []
    for y in range(ly):
        for x in range(lx):
            i = site(x, y, lx)
            if x + 1 < lx:
                nn.append((i, site(x + 1, y, lx)))
            if y + 1 < ly:
                nn.append((i, site(x, y + 1, lx)))
            if x + 1 < lx and y + 1 < ly:
                nnn.append((i, site(x + 1, y + 1, lx)))
            if x - 1 >= 0 and y + 1 < ly:
                nnn.append((i, site(x - 1, y + 1, lx)))
    h = np.zeros((lx * ly, lx * ly), dtype=float)
    for i, j in nn:
        h[i, j] = h[j, i] = -float(t)
    for i, j in nnn:
        h[i, j] = h[j, i] = -float(tprime)
    geometry = Geometry(lx, ly, h, tuple(nn), tuple(nnn))
    validate_geometry(geometry)
    return geometry


def validate_geometry(geometry: Geometry) -> None:
    if not np.array_equal(geometry.hopping, geometry.hopping.T):
        raise ValueError("OBC hopping matrix is not exactly Hermitian")
    expected_nn = geometry.ly * (geometry.lx - 1) + geometry.lx * (geometry.ly - 1)
    expected_nnn = 2 * (geometry.lx - 1) * (geometry.ly - 1)
    if len(geometry.nn_bonds) != expected_nn or len(geometry.nnn_bonds) != expected_nnn:
        raise ValueError(
            f"bad OBC bond counts: NN={len(geometry.nn_bonds)}/{expected_nn}, "
            f"NNN={len(geometry.nnn_bonds)}/{expected_nnn}"
        )
    for bonds in (geometry.nn_bonds, geometry.nnn_bonds):
        if len(set(bonds)) != len(bonds):
            raise ValueError("duplicate OBC physical bond")
        for i, j in bonds:
            xi, yi = i % geometry.lx, i // geometry.lx
            xj, yj = j % geometry.lx, j // geometry.lx
            if abs(xi - xj) > 1 or abs(yi - yj) > 1:
                raise ValueError(f"periodic wrap edge found: {(i, j)}")


def _poly_coeff(weights: np.ndarray, degree: int, excluded: frozenset[int]) -> np.longdouble:
    if degree < 0:
        return np.longdouble(0.0)
    dp = np.zeros(degree + 1, dtype=np.longdouble)
    dp[0] = 1.0
    used = 0
    for idx, weight in enumerate(weights):
        if idx in excluded:
            continue
        used += 1
        for n in range(min(degree, used), 0, -1):
            dp[n] += weight * dp[n - 1]
    return dp[degree]


def canonical_orbital_moments(eigenvalues: np.ndarray, beta: float, npart: int) -> tuple[np.ndarray, np.ndarray]:
    """Return <n_a> and <n_a n_b> for one spin at fixed particle number."""
    norb = len(eigenvalues)
    if not 0 <= npart <= norb:
        raise ValueError(f"invalid N={npart} for {norb} orbitals")
    shifted = np.asarray(eigenvalues - np.min(eigenvalues), dtype=np.longdouble)
    weights = np.exp(-np.longdouble(beta) * shifted)
    partition = _poly_coeff(weights, npart, frozenset())
    if not np.isfinite(partition) or partition <= 0:
        raise ArithmeticError(f"invalid canonical partition coefficient {partition}")
    first = np.zeros(norb, dtype=np.longdouble)
    second = np.zeros((norb, norb), dtype=np.longdouble)
    if npart:
        for a in range(norb):
            first[a] = weights[a] * _poly_coeff(weights, npart - 1, frozenset((a,))) / partition
            second[a, a] = first[a]
    if npart >= 2:
        for a in range(norb):
            for b in range(a + 1, norb):
                value = (
                    weights[a]
                    * weights[b]
                    * _poly_coeff(weights, npart - 2, frozenset((a, b)))
                    / partition
                )
                second[a, b] = second[b, a] = value
    first64 = np.asarray(first, dtype=float)
    second64 = np.asarray(second, dtype=float)
    if abs(float(np.sum(first64)) - npart) > 5e-10:
        raise ArithmeticError("canonical orbital occupation sum failed")
    return first64, second64


def canonical_site_and_pair_moments(
    eigenvectors: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    bonds: tuple[tuple[int, int], ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Return site means and same-spin <n_i n_j> on each physical bond."""
    probabilities = np.abs(eigenvectors) ** 2
    site_means = probabilities @ first
    pairs = np.empty(len(bonds), dtype=float)
    for k, (i, j) in enumerate(bonds):
        direct = np.outer(probabilities[i], probabilities[j])
        overlap = eigenvectors[i] * np.conjugate(eigenvectors[j])
        exchange = np.outer(overlap, np.conjugate(overlap))
        pairs[k] = float(np.real(np.sum(second * (direct - exchange))))
    return site_means, pairs


def bond_summary(
    bonds: tuple[tuple[int, int], ...],
    mean_up: np.ndarray,
    mean_dn: np.ndarray,
    same_up: np.ndarray,
    same_dn: np.ndarray,
) -> tuple[float, float, float]:
    raw_charge: list[float] = []
    spin: list[float] = []
    connected: list[float] = []
    for k, (i, j) in enumerate(bonds):
        cross = mean_up[i] * mean_dn[j] + mean_dn[i] * mean_up[j]
        same = same_up[k] + same_dn[k]
        raw = same + cross
        raw_charge.append(raw)
        spin.append(same - cross)
        connected.append(raw - (mean_up[i] + mean_dn[i]) * (mean_up[j] + mean_dn[j]))
    return float(np.mean(spin)), float(np.mean(raw_charge)), float(np.mean(connected))


def canonical_summary(geometry: Geometry, beta: float, nup: int, ndn: int) -> dict[str, object]:
    eps, orbitals = np.linalg.eigh(geometry.hopping)
    first_up, second_up = canonical_orbital_moments(eps, beta, nup)
    first_dn, second_dn = canonical_orbital_moments(eps, beta, ndn)
    mean_up, nn_up = canonical_site_and_pair_moments(orbitals, first_up, second_up, geometry.nn_bonds)
    mean_dn, nn_dn = canonical_site_and_pair_moments(orbitals, first_dn, second_dn, geometry.nn_bonds)
    _, nnn_up = canonical_site_and_pair_moments(orbitals, first_up, second_up, geometry.nnn_bonds)
    _, nnn_dn = canonical_site_and_pair_moments(orbitals, first_dn, second_dn, geometry.nnn_bonds)
    nn_spin, nn_raw, nn_connected = bond_summary(geometry.nn_bonds, mean_up, mean_dn, nn_up, nn_dn)
    nnn_spin, nnn_raw, nnn_connected = bond_summary(
        geometry.nnn_bonds, mean_up, mean_dn, nnn_up, nnn_dn
    )
    site_density = mean_up + mean_dn
    kinetic = float(np.dot(first_up + first_dn, eps) / geometry.nsites)
    docc = float(np.mean(mean_up * mean_dn))
    density = float(np.mean(site_density))
    return {
        "N_mean": float(np.sum(site_density)),
        "density": density,
        "site_density": site_density,
        "kinetic": kinetic,
        "interaction": 0.0,
        "total": kinetic,
        "double_occupancy": docc,
        "local_moment": density - 2.0 * docc,
        "nn_spin": nn_spin,
        "nn_charge_raw": nn_raw,
        "nn_charge_connected": nn_connected,
        "nnn_spin": nnn_spin,
        "nnn_charge_raw": nnn_raw,
        "nnn_charge_connected": nnn_connected,
    }


def grand_canonical_summary(geometry: Geometry, beta: float, mu: float) -> dict[str, object]:
    eps, orbitals = np.linalg.eigh(geometry.hopping)
    occupations = 1.0 / (1.0 + np.exp(np.clip(beta * (eps - mu), -700.0, 700.0)))
    rho = (orbitals * occupations[None, :]) @ np.conjugate(orbitals.T)
    one_spin = np.real(np.diag(rho))
    site_density = 2.0 * one_spin

    def pairs(bonds: tuple[tuple[int, int], ...]) -> tuple[float, float, float]:
        spin: list[float] = []
        raw: list[float] = []
        connected: list[float] = []
        for i, j in bonds:
            exchange = abs(rho[i, j]) ** 2
            ni, nj = one_spin[i], one_spin[j]
            spin.append(-2.0 * exchange)
            raw.append(4.0 * ni * nj - 2.0 * exchange)
            connected.append(-2.0 * exchange)
        return float(np.mean(spin)), float(np.mean(raw)), float(np.mean(connected))

    nn_spin, nn_raw, nn_connected = pairs(geometry.nn_bonds)
    nnn_spin, nnn_raw, nnn_connected = pairs(geometry.nnn_bonds)
    density = float(np.mean(site_density))
    kinetic = float(2.0 * np.real(np.trace(geometry.hopping @ rho)) / geometry.nsites)
    docc = float(np.mean(one_spin**2))
    return {
        "N_mean": float(np.sum(site_density)),
        "density": density,
        "site_density": site_density,
        "kinetic": kinetic,
        "interaction": 0.0,
        "total": kinetic,
        "double_occupancy": docc,
        "local_moment": density - 2.0 * docc,
        "nn_spin": nn_spin,
        "nn_charge_raw": nn_raw,
        "nn_charge_connected": nn_connected,
        "nnn_spin": nnn_spin,
        "nnn_charge_raw": nnn_raw,
        "nnn_charge_connected": nnn_connected,
    }


def exact_gce_mu(geometry: Geometry, beta: float, target_n: float) -> float:
    eps = np.linalg.eigvalsh(geometry.hopping)

    def number(mu: float) -> float:
        occupations = 1.0 / (1.0 + np.exp(np.clip(beta * (eps - mu), -700.0, 700.0)))
        return float(2.0 * np.sum(occupations))

    lo = float(np.min(eps) - 8.0 - 30.0 / beta)
    hi = float(np.max(eps) + 8.0 + 30.0 / beta)
    for _ in range(260):
        mid = 0.5 * (lo + hi)
        if number(mid) < target_n:
            lo = mid
        else:
            hi = mid
    mu = 0.5 * (lo + hi)
    if abs(number(mu) - target_n) > 2e-10:
        raise ArithmeticError("exact OBC GCE chemical-potential solve failed")
    return mu


def project_commit() -> str:
    root = Path(__file__).resolve().parents[3]
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def beta_label(beta: float) -> str:
    return f"b{beta:.1f}".replace(".", "p")


def read_pbc_reference(path: Path | None) -> dict[tuple[int, float], dict[str, str]]:
    if path is None or not path.is_file():
        return {}
    result: dict[tuple[int, float], dict[str, str]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if float(row.get("U", "nan")) != 0.0:
                continue
            n = int(row.get("L6_Ntot_target", row.get("Ntot", 0)))
            beta = float(row["beta"])
            result[(n, beta)] = row
    return result


def write_tsv(path: Path, rows: Iterable[dict[str, object]], fields: list[str] | None = None) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"refusing to write empty table {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = fields or list(materialized[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)


def write_case_tables(root: Path, row: dict[str, object], summary: dict[str, object]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    common = {
        "boundary": "open",
        "beta": row["beta"],
        "temperature": row["T"],
        "nsamples": 0,
        "nranks": 0,
        "average_phase_abs": 1.0,
        "smoqydqmc_version": SMOQY_VERSION,
        "smoqydqmc_commit": SMOQY_COMMIT,
    }
    primary = (
        ("kinetic_per_site", "kinetic", "equal_time_kinetic_per_site_exact.tsv"),
        ("double_occupancy_per_site", "double_occupancy", "equal_time_double_occupancy_per_site_exact.tsv"),
        ("nn_spin_s_s", "nn_spin", "equal_time_nn_spin_exact.tsv"),
        ("nn_connected_charge", "nn_charge_connected", "equal_time_nn_connected_charge_exact.tsv"),
    )
    for observable, key, filename in primary:
        write_tsv(
            root / filename,
            [{**common, "observable": observable, "value": summary[key], "stderr": 0.0}],
        )
    write_tsv(
        root / "equal_time_observables_obc_exact.tsv",
        [{
            "boundary": "open", "beta": row["beta"], "temperature": row["T"],
            "density": summary["density"], "achieved_N": summary["N_mean"],
            "kinetic_per_site": summary["kinetic"], "kinetic_stderr": 0.0,
            "interaction_per_site": 0.0, "total_per_site": summary["total"],
            "double_occupancy_per_site": summary["double_occupancy"],
            "double_occupancy_stderr": 0.0, "local_moment": summary["local_moment"],
            "nsamples": 0,
        }],
    )
    write_tsv(
        root / "equal_time_bond_observables_exact.tsv",
        [
            {
                "shell": "NN", "bond_count": len(row["geometry"].nn_bonds),
                "normalization": "existing undirected physical bonds",
                "charge_corr_raw": summary["nn_charge_raw"],
                "charge_corr_connected": summary["nn_charge_connected"],
                "spin_corr_s_s": summary["nn_spin"], "stderr": 0.0,
            },
            {
                "shell": "NNN", "bond_count": len(row["geometry"].nnn_bonds),
                "normalization": "existing undirected physical bonds",
                "charge_corr_raw": summary["nnn_charge_raw"],
                "charge_corr_connected": summary["nnn_charge_connected"],
                "spin_corr_s_s": summary["nnn_spin"], "stderr": 0.0,
            },
        ],
    )
    write_tsv(
        root / "equal_time_site_density_exact.tsv",
        [
            {"site": idx, "x": idx % L, "y": idx // L, "density": value}
            for idx, value in enumerate(np.asarray(summary["site_density"], dtype=float))
        ],
    )
    (root / "exact_complete.txt").write_text("strict exact OBC U=0 completion\n")


def build_snapshot(outdir: Path, pbc_reference_path: Path | None = None) -> list[dict[str, object]]:
    geometry = build_geometry(L, L)
    if (geometry.nsites, len(geometry.nn_bonds), len(geometry.nnn_bonds)) != (
        NSITES,
        NN_BONDS,
        NNN_BONDS,
    ):
        raise ValueError("L=6 OBC geometry count guard failed")
    pbc_reference = read_pbc_reference(pbc_reference_path)
    commit = project_commit()
    rows: list[dict[str, object]] = []
    for ntotal in L6_NTOTALS:
        for beta in BETAS:
            for ensemble in ("CE", "GCE"):
                if ensemble == "CE":
                    mu = math.nan
                    summary = canonical_summary(geometry, beta, ntotal // 2, ntotal // 2)
                    method = "one_body_eigensystem_plus_elementary_symmetric_fixed_N_trace"
                else:
                    mu = exact_gce_mu(geometry, beta, ntotal)
                    summary = grand_canonical_summary(geometry, beta, mu)
                    method = "one_body_fermi_trace_with_exact_mu_bisection"
                if abs(float(summary["N_mean"]) - ntotal) > 2e-9:
                    raise ArithmeticError(
                        f"exact {ensemble} N mismatch N={summary['N_mean']} target={ntotal}"
                    )
                ref = pbc_reference.get((ntotal, beta), {})
                row: dict[str, object] = {
                    "idx": len(rows), "boundary": "open", "Lx": L, "Ly": L,
                    "site_count": geometry.nsites, "nn_bond_count": len(geometry.nn_bonds),
                    "nnn_bond_count": len(geometry.nnn_bonds), "ensemble": ensemble,
                    "U_label": "U0", "U": 0.0, "beta": beta, "T": 1.0 / beta,
                    "Ntot": ntotal, "N_mean": summary["N_mean"], "density": summary["density"],
                    "mu": mu, "mu_L6_PBC_reference": ref.get("mu_L6_PBC_reference", ref.get("mu_L8_reference", "")),
                    "mu_L8_reference": ref.get("mu_L8_reference", ""),
                    "kinetic": summary["kinetic"], "kinetic_err": 0.0,
                    "interaction": 0.0, "interaction_err": 0.0,
                    "total": summary["total"], "total_err": 0.0,
                    "double_occupancy": summary["double_occupancy"], "double_occupancy_err": 0.0,
                    "local_moment": summary["local_moment"], "local_moment_err": 0.0,
                    "nn_spin": summary["nn_spin"], "nn_spin_err": 0.0,
                    "nn_charge_connected": summary["nn_charge_connected"],
                    "nn_charge_connected_err": 0.0,
                    "nnn_spin": summary["nnn_spin"], "nnn_spin_err": 0.0,
                    "nnn_charge_connected": summary["nnn_charge_connected"],
                    "nnn_charge_connected_err": 0.0,
                    "average_phase": 1.0, "average_phase_err": 0.0,
                    "exact_method": method, "project_commit": commit,
                    "smoqydqmc_version": SMOQY_VERSION, "smoqydqmc_commit": SMOQY_COMMIT,
                    "canensafqmc_base_commit": CANENS_BASE_COMMIT,
                    "canensafqmc_current_patch_sha256": CANENS_CURRENT_PATCH,
                    "canensafqmc_obc_patch_sha256": CANENS_OBC_PATCH,
                    "kinetic_normalization": "sum over physical hopping matrix / site_count",
                    "double_occupancy_normalization": "sum over sites / site_count",
                    "nn_normalization": "existing undirected physical bonds / nn_bond_count",
                    "nnn_normalization": "existing undirected physical bonds / nnn_bond_count",
                    "final": 1, "sign_limited": 0,
                }
                case_root = outdir / "cases" / f"N{ntotal:03d}" / beta_label(beta) / ensemble.lower()
                row["source"] = str(case_root)
                write_case_tables(case_root, {**row, "geometry": geometry}, summary)
                rows.append(row)
    if len(rows) != 80:
        raise ValueError(f"expected 80 exact ensemble rows, got {len(rows)}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path(__file__).resolve().parent / "status_source" / "exact_u0_obc"
    parser.add_argument("--outdir", type=Path, default=default)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--pbc-reference", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot = args.snapshot or args.outdir / "exact_u0_L6_obc_snapshot.tsv"
    rows = build_snapshot(args.outdir, args.pbc_reference)
    write_tsv(snapshot, rows)
    print(f"wrote {len(rows)} exact OBC ensemble rows to {snapshot}")
    print(f"geometry site/NN/NNN={NSITES}/{NN_BONDS}/{NNN_BONDS}")


if __name__ == "__main__":
    main()
