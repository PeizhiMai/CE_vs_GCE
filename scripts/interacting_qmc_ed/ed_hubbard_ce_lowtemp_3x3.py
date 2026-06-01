#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import coo_matrix, csr_matrix, diags, eye, kron
from scipy.sparse.linalg import eigsh


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Low-temperature exact diagonalization benchmark for the 3x3 Hubbard model in a fixed (Nup,Ndn) sector."
    )
    p.add_argument("--lx", type=int, default=3)
    p.add_argument("--ly", type=int, default=3)
    p.add_argument("--nup", type=int, default=4)
    p.add_argument("--ndn", type=int, default=4)
    p.add_argument("--t", type=float, default=1.0)
    p.add_argument("--u", type=float, default=-5.0)
    p.add_argument("--beta", type=float, default=10.0)
    p.add_argument(
        "--energy-window",
        type=float,
        default=3.0,
        help="Retain eigenstates with E-E0 <= energy-window once enough states have been computed.",
    )
    p.add_argument(
        "--initial-k",
        type=int,
        default=96,
        help="Initial number of low-energy eigenpairs to request from eigsh.",
    )
    p.add_argument(
        "--max-k",
        type=int,
        default=768,
        help="Maximum number of low-energy eigenpairs to request before stopping.",
    )
    p.add_argument("--outdir", type=Path, required=True)
    return p.parse_args()


def gen_basis(nsites: int, nparticles: int) -> list[int]:
    out: list[int] = []
    for comb in combinations(range(nsites), nparticles):
        state = 0
        for i in comb:
            state |= 1 << i
        out.append(state)
    return out


def occ(state: int, site: int) -> int:
    return (state >> site) & 1


def cdagc(state: int, i: int, j: int):
    if not occ(state, j) or occ(state, i):
        return None
    mask_j = 1 << j
    st = state & ~mask_j
    sign1 = -1 if ((state & (mask_j - 1)).bit_count() % 2) else 1
    mask_i = 1 << i
    sign2 = -1 if ((st & (mask_i - 1)).bit_count() % 2) else 1
    return sign1 * sign2, st | mask_i


def site(x: int, y: int, lx: int) -> int:
    return x + y * lx


def hopping(lx: int, ly: int, t: float = 1.0) -> csr_matrix:
    n = lx * ly
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for y in range(ly):
        for x in range(lx):
            i = site(x, y, lx)
            for j in (site((x + 1) % lx, y, lx), site(x, (y + 1) % ly, lx)):
                rows += [j, i]
                cols += [i, j]
                vals += [-t, -t]
    return coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()


def x_terms(lx: int, ly: int, t: float, qx: float = 0.0, qy: float = 0.0, current: bool = True):
    terms: list[tuple[int, int, complex]] = []
    for y in range(ly):
        for x in range(lx):
            src = site(x, y, lx)
            dst = site((x + 1) % lx, y, lx)
            phase = np.exp(1j * (qx * x + qy * y))
            if current:
                terms += [(dst, src, 1j * t * phase), (src, dst, -1j * t * phase)]
            else:
                terms += [(dst, src, -t + 0j), (src, dst, -t + 0j)]
    return terms


def bilinear_from_terms(n: int, basis: list[int], terms: list[tuple[int, int, complex]]) -> csr_matrix:
    index = {state: idx for idx, state in enumerate(basis)}
    rows: list[int] = []
    cols: list[int] = []
    vals: list[complex] = []
    for col, state in enumerate(basis):
        for i, j, amp in terms:
            out = cdagc(state, i, j)
            if out is None:
                continue
            sign, new_state = out
            rows.append(index[new_state])
            cols.append(col)
            vals.append(amp * sign)
    return coo_matrix((vals, (rows, cols)), shape=(len(basis), len(basis))).tocsr()


def one_spin_hamiltonian(tmat: csr_matrix, basis: list[int]) -> csr_matrix:
    coo = tmat.tocoo()
    terms = [(int(i), int(j), float(a)) for i, j, a in zip(coo.row, coo.col, coo.data)]
    return bilinear_from_terms(tmat.shape[0], basis, terms)


def delta_r_list(lx: int, ly: int) -> list[tuple[int, int]]:
    return [(dx, dy) for dx in range(lx // 2 + 1) for dy in range(ly // 2 + 1)]


def index_ipdr(lx: int, ly: int, dx: int, dy: int) -> list[int]:
    out: list[int] = []
    for y in range(ly):
        for x in range(lx):
            out.append(site((x + dx) % lx, (y + dy) % ly, lx))
    return out


def build_sector(args: argparse.Namespace):
    n = args.lx * args.ly
    up_basis = gen_basis(n, args.nup)
    dn_basis = gen_basis(n, args.ndn)
    du = len(up_basis)
    dd = len(dn_basis)
    tmat = hopping(args.lx, args.ly, args.t)
    hup = one_spin_hamiltonian(tmat, up_basis)
    hdn = one_spin_hamiltonian(tmat, dn_basis)
    h_kin = kron(eye(dd, format="csr"), hup) + kron(hdn, eye(du, format="csr"))

    docc = np.empty(du * dd, dtype=float)
    density = np.empty((du * dd, n), dtype=np.int8)
    idx = 0
    for dn in dn_basis:
        occ_dn = np.array([occ(dn, i) for i in range(n)], dtype=np.int8)
        for up in up_basis:
            occ_up = np.array([occ(up, i) for i in range(n)], dtype=np.int8)
            density[idx, :] = occ_up + occ_dn
            docc[idx] = (up & dn).bit_count()
            idx += 1

    h_full = h_kin + diags(args.u * docc)
    return h_full.tocsr(), h_kin.tocsr(), up_basis, dn_basis, docc, density


def low_energy_subspace(h_full: csr_matrix, window: float, initial_k: int, max_k: int):
    dim = h_full.shape[0]
    if dim <= 4096:
        dense = h_full.toarray()
        evals, vecs = eigh(dense, check_finite=False)
        mask = evals <= evals[0] + window
        return evals[mask], vecs[:, mask], dim, evals[0], True

    k = min(initial_k, dim - 2)
    e0 = None
    while True:
        evals, vecs = eigsh(h_full, k=k, which="SA", tol=1e-10)
        order = np.argsort(evals)
        evals = evals[order]
        vecs = vecs[:, order]
        e0 = float(evals[0])
        if float(evals[-1] - e0) >= window or k >= min(max_k, dim - 2):
            mask = evals <= e0 + window
            return evals[mask], vecs[:, mask], k, e0, False
        next_k = min(dim - 2, max(k * 2, k + 32), max_k)
        if next_k == k:
            mask = evals <= e0 + window
            return evals[mask], vecs[:, mask], k, e0, False
        k = next_k


def charge_spin_diagonals(
    density: np.ndarray, up_basis: list[int], dn_basis: list[int], lx: int, ly: int
) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray]:
    n = lx * ly
    deltas = delta_r_list(lx, ly)
    charge = np.zeros((density.shape[0], len(deltas)), dtype=float)
    spin = np.zeros((density.shape[0], len(deltas)), dtype=float)

    spin_density = np.empty_like(density, dtype=np.int8)
    idx = 0
    for dn in dn_basis:
        occ_dn = np.array([occ(dn, i) for i in range(n)], dtype=np.int8)
        for up in up_basis:
            occ_up = np.array([occ(up, i) for i in range(n)], dtype=np.int8)
            spin_density[idx, :] = occ_up - occ_dn
            idx += 1

    for j, (dx, dy) in enumerate(deltas):
        ipdr = index_ipdr(lx, ly, dx, dy)
        charge[:, j] = np.sum(density[:, ipdr] * density, axis=1) / n
        spin[:, j] = np.sum(spin_density[:, ipdr] * spin_density, axis=1) / n

    return deltas, charge, spin


def pair_operator(dx: int, dy: int, lx: int, ly: int, up_basis: list[int], dn_basis: list[int], docc: np.ndarray) -> csr_matrix:
    n = lx * ly
    du = len(up_basis)
    dd = len(dn_basis)
    if dx == 0 and dy == 0:
        return diags(2.0 * docc / n).tocsr()

    up_index = {state: idx for idx, state in enumerate(up_basis)}
    dn_index = {state: idx for idx, state in enumerate(dn_basis)}
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    weight = 1.0 / n

    for dn_idx, dn_state in enumerate(dn_basis):
        for up_idx, up_state in enumerate(up_basis):
            col = up_idx + du * dn_idx
            for y in range(ly):
                for x in range(lx):
                    i = site(x, y, lx)
                    ip = site((x + dx) % lx, (y + dy) % ly, lx)

                    out_up = cdagc(up_state, ip, i)
                    out_dn = cdagc(dn_state, ip, i)
                    if out_up is not None and out_dn is not None:
                        s_up, new_up = out_up
                        s_dn, new_dn = out_dn
                        row = up_index[new_up] + du * dn_index[new_dn]
                        rows.append(row)
                        cols.append(col)
                        vals.append(weight * s_up * s_dn)

                    out_up = cdagc(up_state, i, ip)
                    out_dn = cdagc(dn_state, i, ip)
                    if out_up is not None and out_dn is not None:
                        s_up, new_up = out_up
                        s_dn, new_dn = out_dn
                        row = up_index[new_up] + du * dn_index[new_dn]
                        rows.append(row)
                        cols.append(col)
                        vals.append(weight * s_up * s_dn)

    return coo_matrix((vals, (rows, cols)), shape=(du * dd, du * dd)).tocsr()


def static_response(evals: np.ndarray, Avec: np.ndarray, beta: float, e0: float) -> float:
    total = 0.0
    n = len(evals)
    for m in range(n):
        em = evals[m]
        for nidx in range(n):
            en = evals[nidx]
            denom = em - en
            if abs(denom) < 1e-10:
                kernel = beta * np.exp(-beta * (em - e0))
            else:
                kernel = (np.exp(-beta * (en - e0)) - np.exp(-beta * (em - e0))) / denom
            total += abs(Avec[m, nidx]) ** 2 * kernel
    return float(np.real(total))


def expectation_diag(diag_vals: np.ndarray, vecs: np.ndarray) -> np.ndarray:
    weights = np.abs(vecs) ** 2
    return weights.T @ diag_vals


def expectation_sparse(op: csr_matrix, vecs: np.ndarray) -> np.ndarray:
    out = np.empty(vecs.shape[1], dtype=float)
    for i in range(vecs.shape[1]):
        v = vecs[:, i]
        out[i] = float(np.real(np.vdot(v, op @ v)))
    return out


def thermal_average(values: np.ndarray, evals: np.ndarray, beta: float):
    e0 = float(evals[0])
    w = np.exp(-beta * (evals - e0))
    z = np.sum(w)
    return float(np.dot(w, values) / z), float(z), e0


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    n = args.lx * args.ly
    h_full, h_kin, up_basis, dn_basis, docc, density = build_sector(args)
    evals, vecs, requested_k, e0, used_dense = low_energy_subspace(
        h_full, args.energy_window, args.initial_k, args.max_k
    )

    deltas, charge_diag, spin_diag = charge_spin_diagonals(density, up_basis, dn_basis, args.lx, args.ly)
    kinetic_eig = expectation_sparse(h_kin, vecs)
    docc_eig = expectation_diag(docc, vecs)
    charge_eig = np.column_stack([expectation_diag(charge_diag[:, j], vecs) for j in range(len(deltas))])
    spin_eig = np.column_stack([expectation_diag(spin_diag[:, j], vecs) for j in range(len(deltas))])

    pair_eig_cols = []
    for dx, dy in deltas:
        pair_op = pair_operator(dx, dy, args.lx, args.ly, up_basis, dn_basis, docc)
        pair_eig_cols.append(expectation_sparse(pair_op, vecs))
    pair_eig = np.column_stack(pair_eig_cols)

    du = len(up_basis)
    dd = len(dn_basis)
    Iu = eye(du, format="csr")
    Id = eye(dd, format="csr")
    qxmin = 2 * math.pi / args.lx
    qymin = 2 * math.pi / args.ly
    jL = kron(Id, bilinear_from_terms(n, up_basis, x_terms(args.lx, args.ly, args.t, qx=qxmin, current=True))) \
        + kron(bilinear_from_terms(n, dn_basis, x_terms(args.lx, args.ly, args.t, qx=qxmin, current=True)), Iu)
    jT = kron(Id, bilinear_from_terms(n, up_basis, x_terms(args.lx, args.ly, args.t, qy=qymin, current=True))) \
        + kron(bilinear_from_terms(n, dn_basis, x_terms(args.lx, args.ly, args.t, qy=qymin, current=True)), Iu)
    kx = kron(Id, bilinear_from_terms(n, up_basis, x_terms(args.lx, args.ly, args.t, current=False))) \
        + kron(bilinear_from_terms(n, dn_basis, x_terms(args.lx, args.ly, args.t, current=False)), Iu)

    JL = vecs.conj().T @ jL.toarray() @ vecs
    JT = vecs.conj().T @ jT.toarray() @ vecs
    KX = vecs.conj().T @ kx.toarray() @ vecs

    total_avg, shifted_z, e0_used = thermal_average(evals, evals, args.beta)
    kinetic_avg, _, _ = thermal_average(kinetic_eig, evals, args.beta)
    docc_avg, _, _ = thermal_average(docc_eig, evals, args.beta)

    charge_avg = []
    spin_avg = []
    pair_avg = []
    for j in range(len(deltas)):
        val, _, _ = thermal_average(charge_eig[:, j], evals, args.beta)
        charge_avg.append(val)
        val, _, _ = thermal_average(spin_eig[:, j], evals, args.beta)
        spin_avg.append(val)
        val, _, _ = thermal_average(pair_eig[:, j], evals, args.beta)
        pair_avg.append(val)

    potential_avg = total_avg - kinetic_avg
    lambda_l = static_response(evals, JL, args.beta, e0_used) / (shifted_z * n)
    lambda_t = static_response(evals, JT, args.beta, e0_used) / (shifted_z * n)
    kx_avg, _, _ = thermal_average(np.real(np.diag(KX)), evals, args.beta)
    kx_per_site = kx_avg / n
    rho_s_current = 0.25 * (lambda_l - lambda_t)
    rho_s_diamagnetic = 0.25 * (-kx_per_site - lambda_t)
    temperature = 1.0 / args.beta
    bkt_universal_jump = 2.0 * temperature / math.pi
    bkt_residual_current = rho_s_current - bkt_universal_jump
    bkt_residual_diamagnetic = rho_s_diamagnetic - bkt_universal_jump
    dim = h_full.shape[0]
    retained = len(evals)
    tail_estimate = max(dim - retained, 0) * math.exp(-args.beta * max(args.energy_window, float(evals[-1] - evals[0])))

    with open(args.outdir / "summary.tsv", "w", newline="") as fh:
        cols = [
            "beta",
            "lx",
            "ly",
            "nup",
            "ndn",
            "t",
            "u",
            "hilbert_dim",
            "retained_eigenstates",
            "requested_k",
            "used_dense_diagonalization",
            "energy_window",
            "ground_energy",
            "max_retained_energy",
            "tail_weight_estimate",
            "total_energy",
            "energy_per_site",
            "energy_per_particle",
            "kinetic_energy",
            "kinetic_per_site",
            "potential_energy",
            "potential_per_site",
            "double_occupancy",
            "double_occupancy_per_site",
            "density",
            "lambda_longitudinal_qmin0",
            "lambda_transverse_0qmin",
            "rho_s_current",
            "Kx_per_site",
            "diamagnetic_minus_Kx_per_site",
            "rho_s_diamagnetic",
            "temperature",
            "bkt_universal_jump_2T_over_pi",
            "bkt_residual_current",
            "bkt_residual_diamagnetic",
        ]
        wr = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
        wr.writeheader()
        wr.writerow(
            {
                "beta": args.beta,
                "lx": args.lx,
                "ly": args.ly,
                "nup": args.nup,
                "ndn": args.ndn,
                "t": args.t,
                "u": args.u,
                "hilbert_dim": dim,
                "retained_eigenstates": retained,
                "requested_k": requested_k,
                "used_dense_diagonalization": str(used_dense).lower(),
                "energy_window": args.energy_window,
                "ground_energy": e0_used,
                "max_retained_energy": float(evals[-1]),
                "tail_weight_estimate": tail_estimate,
                "total_energy": total_avg,
                "energy_per_site": total_avg / n,
                "energy_per_particle": total_avg / (args.nup + args.ndn),
                "kinetic_energy": kinetic_avg,
                "kinetic_per_site": kinetic_avg / n,
                "potential_energy": potential_avg,
                "potential_per_site": potential_avg / n,
                "double_occupancy": docc_avg,
                "double_occupancy_per_site": docc_avg / n,
                "density": (args.nup + args.ndn) / n,
                "lambda_longitudinal_qmin0": lambda_l,
                "lambda_transverse_0qmin": lambda_t,
                "rho_s_current": rho_s_current,
                "Kx_per_site": kx_per_site,
                "diamagnetic_minus_Kx_per_site": -kx_per_site,
                "rho_s_diamagnetic": rho_s_diamagnetic,
                "temperature": temperature,
                "bkt_universal_jump_2T_over_pi": bkt_universal_jump,
                "bkt_residual_current": bkt_residual_current,
                "bkt_residual_diamagnetic": bkt_residual_diamagnetic,
            }
        )

    with open(args.outdir / "bkt_observables_ed.tsv", "w", newline="") as fh:
        cols = [
            "beta",
            "temperature",
            "bkt_universal_jump_2T_over_pi",
            "lambda_longitudinal_qmin0",
            "lambda_transverse_0qmin",
            "Kx_per_site",
            "diamagnetic_minus_Kx_per_site",
            "rho_s_current",
            "rho_s_diamagnetic",
            "bkt_residual_current",
            "bkt_residual_diamagnetic",
            "source",
        ]
        wr = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
        wr.writeheader()
        wr.writerow(
            {
                "beta": args.beta,
                "temperature": temperature,
                "bkt_universal_jump_2T_over_pi": bkt_universal_jump,
                "lambda_longitudinal_qmin0": lambda_l,
                "lambda_transverse_0qmin": lambda_t,
                "Kx_per_site": kx_per_site,
                "diamagnetic_minus_Kx_per_site": -kx_per_site,
                "rho_s_current": rho_s_current,
                "rho_s_diamagnetic": rho_s_diamagnetic,
                "bkt_residual_current": bkt_residual_current,
                "bkt_residual_diamagnetic": bkt_residual_diamagnetic,
                "source": "low_temperature_ed_subspace",
            }
        )

    with open(args.outdir / "correlations.tsv", "w", newline="") as fh:
        cols = ["dx", "dy", "charge_corr", "spin_z_corr", "pair_corr"]
        wr = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
        wr.writeheader()
        for (dx, dy), charge_val, spin_val, pair_val in zip(deltas, charge_avg, spin_avg, pair_avg):
            wr.writerow(
                {
                    "dx": dx,
                    "dy": dy,
                    "charge_corr": charge_val,
                    "spin_z_corr": spin_val,
                    "pair_corr": pair_val,
                }
            )

    with open(args.outdir / "retained_spectrum.tsv", "w", newline="") as fh:
        wr = csv.writer(fh, delimiter="\t")
        wr.writerow(["state_index", "eigenvalue", "boltzmann_weight_shifted"])
        shifted_weights = np.exp(-args.beta * (evals - evals[0]))
        for i, (ev, w) in enumerate(zip(evals, shifted_weights), start=1):
            wr.writerow([i, f"{ev:.12f}", f"{w:.12e}"])

    print("Computed canonical low-temperature ED benchmark")
    print(f"  lattice = {args.lx}x{args.ly}")
    print(f"  Nup = {args.nup}, Ndn = {args.ndn}")
    print(f"  U = {args.u}, t = {args.t}, beta = {args.beta}")
    print(f"  Hilbert dimension = {dim}")
    print(f"  Retained eigenstates = {retained}")
    print(f"  E/site = {total_avg / n:.12f}")
    print(f"  docc/site = {docc_avg / n:.12f}")
    print(f"  Lambda_L = {lambda_l:.12f}, Lambda_T = {lambda_t:.12f}")
    print(f"  rho_s = {rho_s_current:.12f}, rho_s_dia = {rho_s_diamagnetic:.12f}")
    print(f"  output directory = {args.outdir}")


if __name__ == "__main__":
    main()
