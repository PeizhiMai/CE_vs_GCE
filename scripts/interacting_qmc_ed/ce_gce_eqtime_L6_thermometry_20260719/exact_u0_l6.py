#!/usr/bin/env python3.11
"""Exact U=0 equal-time CE/GCE observables for finite square PBC lattices.

The canonical calculation fixes N_up=N_down and evaluates orbital-occupation
probabilities with elementary-symmetric-polynomial dynamic programming.  This
retains the finite-size canonical occupation covariances needed for density
correlations; replacing them by a Fermi function would give the wrong CE
answer.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import numpy as np


BETAS = (2.0, 2.2, 2.5, 2.9, 3.3, 4.0, 5.0, 6.7, 10.0, 20.0)
L6_NTOTALS = (12, 18, 26, 32)


def spectrum(lx: int, ly: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mx, my = np.meshgrid(np.arange(lx), np.arange(ly), indexing="ij")
    kx = 2.0 * np.pi * mx.reshape(-1) / lx
    ky = 2.0 * np.pi * my.reshape(-1) / ly
    eps = -2.0 * (np.cos(kx) + np.cos(ky))
    return eps.astype(float), kx.astype(float), ky.astype(float)


def _poly_coeff(weights: np.ndarray, degree: int, excluded: frozenset[int]) -> np.longdouble:
    if degree < 0:
        return np.longdouble(0.0)
    dp = np.zeros(degree + 1, dtype=np.longdouble)
    dp[0] = 1.0
    used = 0
    for i, w in enumerate(weights):
        if i in excluded:
            continue
        used += 1
        hi = min(degree, used)
        for n in range(hi, 0, -1):
            dp[n] += w * dp[n - 1]
    return dp[degree]


def canonical_orbital_moments(eps: np.ndarray, beta: float, npart: int) -> tuple[np.ndarray, np.ndarray]:
    """Return <n_k> and <n_k n_q> for one spin at fixed particle number."""
    n_orb = len(eps)
    if not 0 <= npart <= n_orb:
        raise ValueError(f"invalid canonical particle number {npart} for {n_orb} orbitals")
    shifted = np.asarray(eps - np.min(eps), dtype=np.longdouble)
    weights = np.exp(-np.longdouble(beta) * shifted)
    z = _poly_coeff(weights, npart, frozenset())
    if not np.isfinite(z) or z <= 0:
        raise ArithmeticError(f"invalid canonical partition coefficient Z={z}")
    p = np.zeros(n_orb, dtype=np.longdouble)
    p2 = np.zeros((n_orb, n_orb), dtype=np.longdouble)
    if npart:
        for k in range(n_orb):
            p[k] = weights[k] * _poly_coeff(weights, npart - 1, frozenset((k,))) / z
            p2[k, k] = p[k]
    if npart >= 2:
        for k in range(n_orb):
            for q in range(k + 1, n_orb):
                value = weights[k] * weights[q] * _poly_coeff(
                    weights, npart - 2, frozenset((k, q))
                ) / z
                p2[k, q] = p2[q, k] = value
    p = np.asarray(p, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    if abs(float(np.sum(p)) - npart) > 5e-10:
        raise ArithmeticError("canonical occupation sum failed")
    return p, p2


def exact_gce_mu(lx: int, ly: int, beta: float, ntotal: float) -> float:
    eps, _, _ = spectrum(lx, ly)

    def number(mu: float) -> float:
        x = np.clip(beta * (eps - mu), -700.0, 700.0)
        return float(2.0 * np.sum(1.0 / (1.0 + np.exp(x))))

    lo = float(np.min(eps) - 20.0 / beta - 4.0)
    hi = float(np.max(eps) + 20.0 / beta + 4.0)
    for _ in range(240):
        mid = 0.5 * (lo + hi)
        if number(mid) < ntotal:
            lo = mid
        else:
            hi = mid
    mu = 0.5 * (lo + hi)
    if abs(number(mu) - ntotal) > 2e-10:
        raise ArithmeticError("GCE chemical-potential solve failed")
    return mu


def gce_orbital_moments(eps: np.ndarray, beta: float, mu: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.clip(beta * (eps - mu), -700.0, 700.0)
    p = 1.0 / (1.0 + np.exp(x))
    p2 = np.outer(p, p)
    np.fill_diagonal(p2, p)
    return p, p2


def _minimal_component(x: int, length: int) -> int:
    x %= length
    return min(x, length - x)


def _correlation_arrays(
    lx: int,
    ly: int,
    kx: np.ndarray,
    ky: np.ndarray,
    p: np.ndarray,
    p2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    volume = lx * ly
    nspin = float(np.sum(p) / volume)
    mean_nspin_sq_per_state = float(np.sum(p2) / (volume * volume))
    raw = np.empty((lx, ly), dtype=float)
    connected = np.empty_like(raw)
    spin_ss = np.empty_like(raw)
    spin_szz = np.empty_like(raw)
    phase_diff = (kx[:, None] - kx[None, :], ky[:, None] - ky[None, :])
    for dx in range(lx):
        for dy in range(ly):
            if dx == 0 and dy == 0:
                same_spin_raw = nspin
            else:
                phase = np.exp(1j * (phase_diff[0] * dx + phase_diff[1] * dy))
                g2 = float(np.real(np.sum(p2 * phase)) / (volume * volume))
                same_spin_raw = mean_nspin_sq_per_state - g2
            charge_raw = 2.0 * same_spin_raw + 2.0 * nspin * nspin
            charge_connected = charge_raw - (2.0 * nspin) ** 2
            raw[dx, dy] = charge_raw
            connected[dx, dy] = charge_connected
            spin_ss[dx, dy] = 2.0 * same_spin_raw - 2.0 * nspin * nspin
            spin_szz[dx, dy] = 0.25 * spin_ss[dx, dy]
    return raw, connected, spin_ss, spin_szz


def compute_case(lx: int, ly: int, beta: float, ntotal: int, ensemble: str) -> dict:
    if ntotal % 2:
        raise ValueError("balanced U=0 cases require even Ntotal")
    volume = lx * ly
    eps, kx, ky = spectrum(lx, ly)
    ensemble = ensemble.upper()
    if ensemble == "CE":
        mu = math.nan
        p, p2 = canonical_orbital_moments(eps, beta, ntotal // 2)
    elif ensemble == "GCE":
        mu = exact_gce_mu(lx, ly, beta, ntotal)
        p, p2 = gce_orbital_moments(eps, beta, mu)
    else:
        raise ValueError(f"unknown ensemble {ensemble}")
    nspin = float(np.sum(p) / volume)
    kinetic = float(2.0 * np.sum(eps * p) / volume)
    kx_per_site = float(2.0 * np.sum((-2.0 * np.cos(kx)) * p) / volume)
    double_occ = nspin * nspin
    local_moment = 2.0 * nspin - 2.0 * double_occ
    raw, conn, spin, spin_z = _correlation_arrays(lx, ly, kx, ky, p, p2)
    nn_vectors = ((1, 0), (lx - 1, 0), (0, 1), (0, ly - 1))
    nnn_vectors = ((1, 1), (1, ly - 1), (lx - 1, 1), (lx - 1, ly - 1))
    avg = lambda arr, vecs: float(np.mean([arr[x, y] for x, y in vecs]))
    return {
        "Lx": lx,
        "Ly": ly,
        "volume": volume,
        "ensemble": ensemble,
        "beta": float(beta),
        "T": 1.0 / float(beta),
        "Ntot": int(ntotal),
        "N_mean": 2.0 * float(np.sum(p)),
        "density": 2.0 * nspin,
        "mu": mu,
        "kinetic": kinetic,
        "interaction": 0.0,
        "total": kinetic,
        "double_occupancy": double_occ,
        "local_moment_z": local_moment,
        "Kx_per_site": kx_per_site,
        "diamagnetic_minus_Kx_per_site": -kx_per_site,
        "charge_raw": raw,
        "charge_connected": conn,
        "spin_s_s": spin,
        "spin_SzSz": spin_z,
        "nn_charge_raw": avg(raw, nn_vectors),
        "nn_charge_connected": avg(conn, nn_vectors),
        "nn_spin": avg(spin, nn_vectors),
        "nn_spin_SzSz": avg(spin_z, nn_vectors),
        "nnn_charge_raw": avg(raw, nnn_vectors),
        "nnn_charge_connected": avg(conn, nnn_vectors),
        "nnn_spin": avg(spin, nnn_vectors),
        "nnn_spin_SzSz": avg(spin_z, nnn_vectors),
    }


def _write_tsv(path: Path, fields: Iterable[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _vectors_for_r2(lx: int, ly: int, r2: int) -> list[tuple[int, int]]:
    out = []
    for dx in range(lx):
        for dy in range(ly):
            if dx == 0 and dy == 0:
                continue
            if _minimal_component(dx, lx) ** 2 + _minimal_component(dy, ly) ** 2 == r2:
                out.append((dx, dy))
    return out


def write_case_tables(case: dict, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    common = {
        "nsamples": 0,
        "batches": 0,
        "nranks": 0,
        "phase_reweighted": False,
        "phase_sum": 1.0,
        "average_phase": 1.0,
    }
    obs = {
        "beta": case["beta"],
        "temperature": case["T"],
        "nup": case["Ntot"] // 2 if case["ensemble"] == "CE" else case["N_mean"] / 2,
        "ndn": case["Ntot"] // 2 if case["ensemble"] == "CE" else case["N_mean"] / 2,
        "ntotal": case["Ntot"] if case["ensemble"] == "CE" else case["N_mean"],
        "density": case["density"],
        "kinetic_per_site": case["kinetic"],
        "kinetic_stderr": 0.0,
        "interaction_per_site": 0.0,
        "interaction_stderr": 0.0,
        "total_per_site": case["total"],
        "total_stderr": 0.0,
        "double_occupancy_per_site": case["double_occupancy"],
        "double_occupancy_stderr": 0.0,
        "local_moment_z": case["local_moment_z"],
        "local_moment_z_stderr": 0.0,
        "Kx_per_site": case["Kx_per_site"],
        "Kx_stderr": 0.0,
        "diamagnetic_minus_Kx_per_site": case["diamagnetic_minus_Kx_per_site"],
        "diamagnetic_minus_Kx_stderr": 0.0,
        "time_slices": int(round(case["beta"] / 0.1)),
        **common,
    }
    _write_tsv(outdir / "equal_time_observables_exact.tsv", obs.keys(), [obs])

    corr_fields = [
        "dx", "dy", "charge_corr_raw", "charge_corr_raw_stderr",
        "charge_corr_connected", "charge_corr_connected_stderr",
        "spin_corr_s_s", "spin_corr_s_s_stderr", "spin_corr_SzSz",
        "spin_corr_SzSz_stderr", *common.keys(),
    ]
    corr_rows = []
    for dx in range(case["Lx"] // 2 + 1):
        for dy in range(case["Ly"] // 2 + 1):
            corr_rows.append({
                "dx": dx,
                "dy": dy,
                "charge_corr_raw": case["charge_raw"][dx, dy],
                "charge_corr_raw_stderr": 0.0,
                "charge_corr_connected": case["charge_connected"][dx, dy],
                "charge_corr_connected_stderr": 0.0,
                "spin_corr_s_s": case["spin_s_s"][dx, dy],
                "spin_corr_s_s_stderr": 0.0,
                "spin_corr_SzSz": case["spin_SzSz"][dx, dy],
                "spin_corr_SzSz_stderr": 0.0,
                **common,
            })
    _write_tsv(outdir / "equal_time_charge_spin_wedge_exact.tsv", corr_fields, corr_rows)

    sf_fields = [
        "mx", "my", "qx", "qy", "charge_structure_raw", "charge_structure_raw_stderr",
        "charge_structure_connected", "charge_structure_connected_stderr", "spin_structure_s_s",
        "spin_structure_s_s_stderr", "spin_structure_SzSz", "spin_structure_SzSz_stderr",
        *common.keys(),
    ]
    sf_rows = []
    for mx in range(case["Lx"] // 2 + 1):
        for my in range(min(mx, case["Ly"] // 2) + 1):
            qx, qy = 2.0 * np.pi * mx / case["Lx"], 2.0 * np.pi * my / case["Ly"]
            phase = np.empty((case["Lx"], case["Ly"]), dtype=complex)
            for dx in range(case["Lx"]):
                for dy in range(case["Ly"]):
                    phase[dx, dy] = np.exp(1j * (qx * dx + qy * dy))
            transform = lambda arr: float(np.real(np.sum(arr * phase)))
            sf_rows.append({
                "mx": mx,
                "my": my,
                "qx": qx,
                "qy": qy,
                "charge_structure_raw": transform(case["charge_raw"]),
                "charge_structure_raw_stderr": 0.0,
                "charge_structure_connected": transform(case["charge_connected"]),
                "charge_structure_connected_stderr": 0.0,
                "spin_structure_s_s": transform(case["spin_s_s"]),
                "spin_structure_s_s_stderr": 0.0,
                "spin_structure_SzSz": transform(case["spin_SzSz"]),
                "spin_structure_SzSz_stderr": 0.0,
                **common,
            })
    _write_tsv(outdir / "equal_time_structure_factors_exact.tsv", sf_fields, sf_rows)

    shell_fields = [
        "shell", "r2", "vectors", "charge_corr_raw", "charge_corr_raw_stderr",
        "charge_corr_connected", "charge_corr_connected_stderr", "spin_corr_s_s",
        "spin_corr_s_s_stderr", "spin_corr_SzSz", "spin_corr_SzSz_stderr", *common.keys(),
    ]
    r2_values = sorted({
        _minimal_component(dx, case["Lx"]) ** 2 + _minimal_component(dy, case["Ly"]) ** 2
        for dx in range(case["Lx"]) for dy in range(case["Ly"]) if (dx, dy) != (0, 0)
    })[:4]
    shell_rows = []
    for shell, r2 in enumerate(r2_values, start=1):
        vectors = _vectors_for_r2(case["Lx"], case["Ly"], r2)
        avg = lambda arr: float(np.mean([arr[x, y] for x, y in vectors]))
        shell_rows.append({
            "shell": shell,
            "r2": r2,
            "vectors": ";".join(f"({x},{y})" for x, y in vectors),
            "charge_corr_raw": avg(case["charge_raw"]),
            "charge_corr_raw_stderr": 0.0,
            "charge_corr_connected": avg(case["charge_connected"]),
            "charge_corr_connected_stderr": 0.0,
            "spin_corr_s_s": avg(case["spin_s_s"]),
            "spin_corr_s_s_stderr": 0.0,
            "spin_corr_SzSz": avg(case["spin_SzSz"]),
            "spin_corr_SzSz_stderr": 0.0,
            **common,
        })
    _write_tsv(outdir / "equal_time_neighbor_shells_exact.tsv", shell_fields, shell_rows)
    (outdir / "exact_complete.txt").write_text("strict_final=true\n")


def generate_l6(out_root: Path, snapshot_path: Path) -> None:
    rows = []
    for ntotal in L6_NTOTALS:
        for beta in BETAS:
            for ensemble in ("CE", "GCE"):
                case = compute_case(6, 6, beta, ntotal, ensemble)
                beta_label = str(beta).replace(".", "p")
                case_dir = out_root / ensemble / f"Ntot{ntotal:03d}_n{ntotal/36:.6f}_b{beta_label}"
                write_case_tables(case, case_dir)
                rows.append({
                    "L": 6,
                    "U": 0.0,
                    "ensemble": ensemble,
                    "Ntot": ntotal,
                    "target_density": ntotal / 36.0,
                    "beta": beta,
                    "T": 1.0 / beta,
                    "N_mean": case["N_mean"],
                    "density": case["density"],
                    "kinetic": case["kinetic"],
                    "kinetic_err": 0.0,
                    "double_occupancy": case["double_occupancy"],
                    "double_occupancy_err": 0.0,
                    "nn_spin": case["nn_spin"],
                    "nn_spin_err": 0.0,
                    "nn_charge_connected": case["nn_charge_connected"],
                    "nn_charge_connected_err": 0.0,
                    "nnn_spin": case["nnn_spin"],
                    "nnn_spin_err": 0.0,
                    "nnn_charge_connected": case["nnn_charge_connected"],
                    "nnn_charge_connected_err": 0.0,
                    "mu": case["mu"],
                    "final": 1,
                    "source": str(case_dir),
                    "workflow": "exact_U0_L6_PBC_20260719",
                })
    _write_tsv(snapshot_path, rows[0].keys(), rows)


def validate_against_l8(reference_snapshot: Path, tolerance: float = 2e-10) -> list[str]:
    rows = list(csv.DictReader(reference_snapshot.open(), delimiter="\t"))
    failures = []
    for r in rows:
        if float(r["U"]) != 0.0:
            continue
        ntotal, beta, ensemble = int(r["Ntot"]), float(r["beta"]), r["ensemble"]
        case = compute_case(8, 8, beta, ntotal, ensemble)
        for source_key, case_key in (
            ("kinetic", "kinetic"),
            ("double_occupancy", "double_occupancy"),
            ("nn_spin", "nn_spin"),
            ("nn_charge_connected", "nn_charge_connected"),
        ):
            expected = float(r[source_key])
            actual = float(case[case_key])
            if abs(expected - actual) > tolerance:
                failures.append(
                    f"{ensemble} N={ntotal} beta={beta} {source_key}: expected={expected} actual={actual}"
                )
        if ensemble == "GCE" and r.get("mu", "") not in ("", "nan", "NaN"):
            expected_mu = float(r["mu"])
            if abs(expected_mu - float(case["mu"])) > 2e-6:
                failures.append(
                    f"GCE N={ntotal} beta={beta} mu: expected={expected_mu} actual={case['mu']}"
                )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--validate-l8", type=Path)
    args = parser.parse_args()
    if args.validate_l8:
        failures = validate_against_l8(args.validate_l8)
        if failures:
            raise SystemExit("\n".join(failures[:30]))
        print("L8 exact-U0 regression: PASS")
    if args.out_root or args.snapshot:
        if not args.out_root or not args.snapshot:
            raise SystemExit("--out-root and --snapshot are required together")
        generate_l6(args.out_root, args.snapshot)
        print(f"wrote exact L6 U=0 outputs under {args.out_root}")
        print(f"wrote {args.snapshot}")


if __name__ == "__main__":
    main()
