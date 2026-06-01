#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags, eye, kron
from scipy.sparse.linalg import eigsh


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Canonical ED Matsubara Green benchmark for 3x3 Hubbard.")
    p.add_argument("--lx", type=int, default=3)
    p.add_argument("--ly", type=int, default=3)
    p.add_argument("--nup", type=int, default=4)
    p.add_argument("--ndn", type=int, default=4)
    p.add_argument("--t", type=float, default=1.0)
    p.add_argument("--u", type=float, default=-5.0)
    p.add_argument("--beta", type=float, default=10.0)
    p.add_argument("--dtau", type=float, default=0.1)
    p.add_argument("--nfreq", type=int, default=10)
    p.add_argument("--energy-window", type=float, default=4.0)
    p.add_argument("--initial-k", type=int, default=128)
    p.add_argument("--max-k", type=int, default=1024)
    p.add_argument(
        "--full-spectrum",
        action="store_true",
        help="Use all eigenstates in the connected canonical sectors, via 2D translation momentum blocks.",
    )
    p.add_argument(
        "--measure-bkt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write bkt_observables_ed.tsv with current response and superfluid stiffness when available.",
    )
    p.add_argument(
        "--bkt-only",
        action="store_true",
        help="Only compute bkt_observables_ed.tsv in the fixed (Nup,Ndn) sector; skip one-particle Green functions.",
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
    if i == j:
        return (1, state) if occ(state, j) else None
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


def coords(i: int, lx: int) -> tuple[int, int]:
    return i % lx, i // lx


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


def spinful_bilinear_from_terms(
    nsites: int,
    up_basis: list[int],
    dn_basis: list[int],
    terms: list[tuple[int, int, complex]],
) -> csr_matrix:
    iup = eye(len(up_basis), format="csr")
    idn = eye(len(dn_basis), format="csr")
    op_up = bilinear_from_terms(nsites, up_basis, terms)
    op_dn = bilinear_from_terms(nsites, dn_basis, terms)
    return (kron(idn, op_up, format="csr") + kron(op_dn, iup, format="csr")).tocsr()


def build_sector(lx: int, ly: int, nup: int, ndn: int, t: float, u: float):
    n = lx * ly
    up_basis = gen_basis(n, nup)
    dn_basis = gen_basis(n, ndn)
    du = len(up_basis)
    dd = len(dn_basis)
    tmat = hopping(lx, ly, t)
    hup = one_spin_hamiltonian(tmat, up_basis)
    hdn = one_spin_hamiltonian(tmat, dn_basis)
    h_kin = kron(eye(dd, format="csr"), hup) + kron(hdn, eye(du, format="csr"))

    docc = np.empty(du * dd, dtype=float)
    idx = 0
    for dn in dn_basis:
        for up in up_basis:
            docc[idx] = (up & dn).bit_count()
            idx += 1
    h_full = h_kin + diags(u * docc)
    return h_full.tocsr(), up_basis, dn_basis


def low_energy_subspace(h_full: csr_matrix, window: float, initial_k: int, max_k: int):
    dim = h_full.shape[0]
    k = min(initial_k, dim - 2)
    while True:
        evals, vecs = eigsh(h_full, k=k, which="SA", tol=1e-10)
        order = np.argsort(evals)
        evals = evals[order]
        vecs = vecs[:, order]
        e0 = float(evals[0])
        if float(evals[-1] - e0) >= window or k >= min(max_k, dim - 2):
            mask = evals <= e0 + window
            return evals[mask], vecs[:, mask], k, e0
        next_k = min(dim - 2, max(k * 2, k + 32), max_k)
        if next_k == k:
            mask = evals <= e0 + window
            return evals[mask], vecs[:, mask], k, e0
        k = next_k


def annihilation_up_matrix(site_idx: int, up_basis_n: list[int], up_basis_np1: list[int]) -> csr_matrix:
    idx_n = {state: i for i, state in enumerate(up_basis_n)}
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    mask = 1 << site_idx
    for col, state in enumerate(up_basis_np1):
        if not occ(state, site_idx):
            continue
        new_state = state & ~mask
        sign = -1 if ((state & (mask - 1)).bit_count() % 2) else 1
        rows.append(idx_n[new_state])
        cols.append(col)
        vals.append(sign)
    return coo_matrix((vals, (rows, cols)), shape=(len(up_basis_n), len(up_basis_np1))).tocsr()


def dense_one_spin_hamiltonian(tmat: csr_matrix, basis: list[int]) -> np.ndarray:
    coo = tmat.tocoo()
    terms = [(int(i), int(j), float(a)) for i, j, a in zip(coo.row, coo.col, coo.data)]
    return bilinear_from_terms(tmat.shape[0], basis, terms).toarray()


def realspace_average_from_gij(gij: np.ndarray, lx: int, ly: int) -> np.ndarray:
    """Return Gr[dx,dy] = V^-1 sum_j G[j+r,j]."""
    nsites = lx * ly
    gr = np.zeros((lx, ly), dtype=np.complex128)
    for dy in range(ly):
        for dx in range(lx):
            acc = 0.0j
            for y in range(ly):
                for x in range(lx):
                    j = site(x, y, lx)
                    i = site((x + dx) % lx, (y + dy) % ly, lx)
                    acc += gij[i, j]
            gr[dx, dy] = acc / nsites
    return gr


def momentum_from_realspace(gr: np.ndarray) -> np.ndarray:
    """Return Gk[nx,ny] = sum_r exp(-i k.r) Gr[r]."""
    lx, ly = gr.shape
    gk = np.zeros((lx, ly), dtype=np.complex128)
    for ny in range(ly):
        ky = 2.0 * math.pi * ny / ly
        for nx in range(lx):
            kx = 2.0 * math.pi * nx / lx
            acc = 0.0j
            for dy in range(ly):
                for dx in range(lx):
                    acc += np.exp(-1j * (kx * dx + ky * dy)) * gr[dx, dy]
            gk[nx, ny] = acc
    return gk


def addition_space_tau_from_site_mats(
    lx: int,
    ly: int,
    beta: float,
    tau_vals: np.ndarray,
    evals_n: np.ndarray,
    evals_np1: np.ndarray,
    z_n: np.ndarray,
    znorm: float,
    c_mats: list[np.ndarray],
    gij_tau0: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build addition-branch Gr(tau), Gk(tau) from site annihilation matrices.

    c_mats[i][m,n] = <m,N| c_i |n,N+1>.
    """
    nsites = lx * ly
    gr_tau = np.zeros((len(tau_vals), lx, ly), dtype=np.complex128)
    gk_tau = np.zeros((len(tau_vals), lx, ly), dtype=np.complex128)
    start = 0
    if gij_tau0 is not None:
        gr_tau[0] = realspace_average_from_gij(gij_tau0, lx, ly)
        gk_tau[0] = momentum_from_realspace(gr_tau[0])
        start = 1
    for l in range(start, len(tau_vals)):
        tau = float(tau_vals[l])
        weights = np.exp(-(beta - tau) * evals_n)[:, None] * np.exp(-tau * evals_np1)[None, :] / znorm
        gij = np.zeros((nsites, nsites), dtype=np.complex128)
        for i in range(nsites):
            mi = c_mats[i]
            for j in range(nsites):
                gij[i, j] = np.sum(weights * mi * np.conj(c_mats[j]))
        gr_tau[l] = realspace_average_from_gij(gij, lx, ly)
        gk_tau[l] = momentum_from_realspace(gr_tau[l])
    return gr_tau, gk_tau


def write_space_tau_tables(outdir: Path, tau_vals: np.ndarray, gr_tau: np.ndarray, gk_tau: np.ndarray, suffix: str) -> None:
    lx, ly = gr_tau.shape[1:]
    with open(outdir / f"greens_r_tau_add_{suffix}.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["slice", "tau", "dx", "dy", "Greal", "Gimag"])
        for l, tau in enumerate(tau_vals):
            for dy in range(ly):
                for dx in range(lx):
                    val = gr_tau[l, dx, dy]
                    w.writerow([l, tau, dx, dy, val.real, val.imag])

    with open(outdir / f"greens_k_tau_add_{suffix}.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["slice", "tau", "nx", "ny", "kx", "ky", "Greal", "Gimag"])
        for l, tau in enumerate(tau_vals):
            for ny in range(ly):
                ky = 2.0 * math.pi * ny / ly
                for nx in range(lx):
                    kx = 2.0 * math.pi * nx / lx
                    val = gk_tau[l, nx, ny]
                    w.writerow([l, tau, nx, ny, kx, ky, val.real, val.imag])


def translate_bits(bits: int, dx: int, dy: int, lx: int, ly: int) -> int:
    out = 0
    for y in range(ly):
        for x in range(lx):
            i = site(x, y, lx)
            if (bits >> i) & 1:
                j = site((x + dx) % lx, (y + dy) % ly, lx)
                out |= 1 << j
    return out


def translate_bits_signed(bits: int, dx: int, dy: int, lx: int, ly: int) -> tuple[int, int]:
    mapped: list[int] = []
    for i in range(lx * ly):
        if (bits >> i) & 1:
            x, y = coords(i, lx)
            mapped.append(site((x + dx) % lx, (y + dy) % ly, lx))
    inversions = 0
    for a in range(len(mapped)):
        for b in range(a + 1, len(mapped)):
            if mapped[a] > mapped[b]:
                inversions += 1
    out = 0
    for i in mapped:
        out |= 1 << i
    return out, (-1 if inversions % 2 else 1)


def translate_pair(state: tuple[int, int], dx: int, dy: int, lx: int, ly: int) -> tuple[int, int]:
    up, dn = state
    return translate_bits(up, dx, dy, lx, ly), translate_bits(dn, dx, dy, lx, ly)


def translate_pair_signed(state: tuple[int, int], dx: int, dy: int, lx: int, ly: int) -> tuple[tuple[int, int], int]:
    up, dn = state
    up_t, sign_up = translate_bits_signed(up, dx, dy, lx, ly)
    dn_t, sign_dn = translate_bits_signed(dn, dx, dy, lx, ly)
    return (up_t, dn_t), sign_up * sign_dn


def combined_spin_basis(up_basis: list[int], dn_basis: list[int]) -> list[tuple[int, int]]:
    # Same ordering as build_sector(): down index outside, up index inside.
    return [(up, dn) for dn in dn_basis for up in up_basis]


def build_translation_projectors(
    lx: int,
    ly: int,
    up_basis: list[int],
    dn_basis: list[int],
) -> dict[tuple[int, int], csr_matrix]:
    states = combined_spin_basis(up_basis, dn_basis)
    index = {state: i for i, state in enumerate(states)}
    nstates = len(states)
    translations = [(dx, dy) for dy in range(ly) for dx in range(lx)]

    rows: dict[tuple[int, int], list[int]] = {(nx, ny): [] for ny in range(ly) for nx in range(lx)}
    cols: dict[tuple[int, int], list[int]] = {(nx, ny): [] for ny in range(ly) for nx in range(lx)}
    vals: dict[tuple[int, int], list[complex]] = {(nx, ny): [] for ny in range(ly) for nx in range(lx)}
    ncols: dict[tuple[int, int], int] = {(nx, ny): 0 for ny in range(ly) for nx in range(lx)}

    visited: set[tuple[int, int]] = set()
    for state in states:
        if state in visited:
            continue
        orbit = {translate_pair(state, dx, dy, lx, ly) for dx, dy in translations}
        visited.update(orbit)

        for ny in range(ly):
            ky = 2.0 * math.pi * ny / ly
            for nx in range(lx):
                kx = 2.0 * math.pi * nx / lx
                coeffs: dict[int, complex] = {}
                for dx, dy in translations:
                    translated, translate_sign = translate_pair_signed(state, dx, dy, lx, ly)
                    phase = np.exp(-1j * (kx * dx + ky * dy))
                    row = index[translated]
                    coeffs[row] = coeffs.get(row, 0.0j) + translate_sign * phase
                norm = math.sqrt(float(sum(abs(c) ** 2 for c in coeffs.values())))
                if norm < 1e-10:
                    continue
                key = (nx, ny)
                col = ncols[key]
                ncols[key] += 1
                for row, coeff in coeffs.items():
                    rows[key].append(row)
                    cols[key].append(col)
                    vals[key].append(coeff / norm)

    projectors: dict[tuple[int, int], csr_matrix] = {}
    total_dim = 0
    for key in sorted(ncols):
        q = coo_matrix((vals[key], (rows[key], cols[key])), shape=(nstates, ncols[key]), dtype=np.complex128).tocsr()
        projectors[key] = q
        total_dim += q.shape[1]
    if total_dim != nstates:
        raise RuntimeError(f"translation projectors are incomplete: block dims sum to {total_dim}, expected {nstates}")
    return projectors


def diagonalize_translation_blocks(
    h: csr_matrix,
    projectors: dict[tuple[int, int], csr_matrix],
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    blocks: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for key, q in sorted(projectors.items()):
        hk = (q.conj().T @ (h @ q)).toarray()
        hk = 0.5 * (hk + hk.conj().T)
        evals, vecs = np.linalg.eigh(hk)
        blocks[key] = (evals, vecs)
    return blocks


def shifted_partition_from_blocks(
    blocks: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
    beta: float,
    e0: float,
) -> float:
    return float(sum(np.sum(np.exp(-beta * (evals - e0))) for evals, _ in blocks.values()))


def static_response_kernel_sum(
    evals_left: np.ndarray,
    evals_right: np.ndarray,
    spectral: np.ndarray,
    beta: float,
    e0: float,
) -> float:
    em = evals_left[:, None]
    en = evals_right[None, :]
    denom = em - en
    wm = np.exp(-beta * (em - e0))
    wn = np.exp(-beta * (en - e0))
    kernel = np.empty_like(denom, dtype=np.float64)
    mask = np.abs(denom) < 1.0e-10
    kernel[mask] = np.broadcast_to(beta * wm, denom.shape)[mask]
    numerator = np.broadcast_to(wn, denom.shape) - np.broadcast_to(wm, denom.shape)
    kernel[~mask] = numerator[~mask] / denom[~mask]
    return float(np.sum(spectral * kernel).real)


def static_response_blocks(
    op: csr_matrix,
    projectors: dict[tuple[int, int], csr_matrix],
    blocks: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
    beta: float,
    e0: float,
) -> float:
    total = 0.0
    for key_right, q_right in projectors.items():
        op_q_right = op @ q_right
        evals_right, vecs_right = blocks[key_right]
        for key_left, q_left in projectors.items():
            block_sparse = q_left.conj().T @ op_q_right
            if block_sparse.nnz == 0:
                continue
            block = block_sparse.toarray()
            if np.linalg.norm(block) < 1.0e-12:
                continue
            evals_left, vecs_left = blocks[key_left]
            mat = vecs_left.conj().T @ block @ vecs_right
            spectral = np.abs(mat) ** 2
            if float(spectral.sum()) < 1.0e-18:
                continue
            total += static_response_kernel_sum(evals_left, evals_right, spectral, beta, e0)
    return total


def thermal_average_operator_blocks(
    op: csr_matrix,
    projectors: dict[tuple[int, int], csr_matrix],
    blocks: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
    beta: float,
    e0: float,
) -> float:
    total = 0.0
    for key, q in projectors.items():
        evals, vecs = blocks[key]
        block = (q.conj().T @ (op @ q)).toarray()
        mat = vecs.conj().T @ block @ vecs
        diag = np.real(np.diag(mat))
        weights = np.exp(-beta * (evals - e0))
        total += float(np.dot(weights, diag))
    return total


def bkt_observables_from_blocks(
    lx: int,
    ly: int,
    t: float,
    beta: float,
    up_basis: list[int],
    dn_basis: list[int],
    projectors: dict[tuple[int, int], csr_matrix],
    blocks: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
) -> dict[str, float]:
    nsites = lx * ly
    all_e = np.concatenate([evals for evals, _ in blocks.values()])
    e0 = float(all_e.min())
    z_shifted = shifted_partition_from_blocks(blocks, beta, e0)
    qxmin = 2.0 * math.pi / lx
    qymin = 2.0 * math.pi / ly

    j_l = spinful_bilinear_from_terms(nsites, up_basis, dn_basis, x_terms(lx, ly, t, qx=qxmin, qy=0.0, current=True))
    j_t = spinful_bilinear_from_terms(nsites, up_basis, dn_basis, x_terms(lx, ly, t, qx=0.0, qy=qymin, current=True))
    kx = spinful_bilinear_from_terms(nsites, up_basis, dn_basis, x_terms(lx, ly, t, current=False))

    lambda_l = static_response_blocks(j_l, projectors, blocks, beta, e0) / (z_shifted * nsites)
    lambda_t = static_response_blocks(j_t, projectors, blocks, beta, e0) / (z_shifted * nsites)
    kx_per_site = thermal_average_operator_blocks(kx, projectors, blocks, beta, e0) / (z_shifted * nsites)
    rho_s_current = 0.25 * (lambda_l - lambda_t)
    rho_s_diamagnetic = 0.25 * (-kx_per_site - lambda_t)
    temperature = 1.0 / beta
    jump = 2.0 * temperature / math.pi
    return {
        "beta": beta,
        "temperature": temperature,
        "bkt_universal_jump_2T_over_pi": jump,
        "lambda_longitudinal_qmin0": lambda_l,
        "lambda_transverse_0qmin": lambda_t,
        "Kx_per_site": kx_per_site,
        "diamagnetic_minus_Kx_per_site": -kx_per_site,
        "rho_s_current": rho_s_current,
        "rho_s_diamagnetic": rho_s_diamagnetic,
        "bkt_residual_current": rho_s_current - jump,
        "bkt_residual_diamagnetic": rho_s_diamagnetic - jump,
        "qmin": qxmin,
        "znorm_shifted": z_shifted,
    }


def write_bkt_observables_ed(outdir: Path, bkt: dict[str, float], source: str) -> None:
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
        "qmin",
        "znorm_shifted",
        "source",
    ]
    with open(outdir / "bkt_observables_ed.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        row = {c: bkt.get(c, "") for c in cols}
        row["source"] = source
        w.writerow(row)


def bkt_observables_from_dense_subspace(
    lx: int,
    ly: int,
    t: float,
    beta: float,
    up_basis: list[int],
    dn_basis: list[int],
    evals: np.ndarray,
    vecs: np.ndarray,
) -> dict[str, float]:
    nsites = lx * ly
    e0 = float(evals[0])
    z_shifted = float(np.sum(np.exp(-beta * (evals - e0))))
    qxmin = 2.0 * math.pi / lx
    qymin = 2.0 * math.pi / ly
    j_l = spinful_bilinear_from_terms(nsites, up_basis, dn_basis, x_terms(lx, ly, t, qx=qxmin, qy=0.0, current=True))
    j_t = spinful_bilinear_from_terms(nsites, up_basis, dn_basis, x_terms(lx, ly, t, qx=0.0, qy=qymin, current=True))
    kx = spinful_bilinear_from_terms(nsites, up_basis, dn_basis, x_terms(lx, ly, t, current=False))
    jl_eig = vecs.conj().T @ (j_l @ vecs)
    jt_eig = vecs.conj().T @ (j_t @ vecs)
    kx_eig = vecs.conj().T @ (kx @ vecs)
    lambda_l = static_response_kernel_sum(evals, evals, np.abs(jl_eig) ** 2, beta, e0) / (z_shifted * nsites)
    lambda_t = static_response_kernel_sum(evals, evals, np.abs(jt_eig) ** 2, beta, e0) / (z_shifted * nsites)
    weights = np.exp(-beta * (evals - e0))
    kx_per_site = float(np.dot(weights, np.real(np.diag(kx_eig)))) / (z_shifted * nsites)
    rho_s_current = 0.25 * (lambda_l - lambda_t)
    rho_s_diamagnetic = 0.25 * (-kx_per_site - lambda_t)
    temperature = 1.0 / beta
    jump = 2.0 * temperature / math.pi
    return {
        "beta": beta,
        "temperature": temperature,
        "bkt_universal_jump_2T_over_pi": jump,
        "lambda_longitudinal_qmin0": lambda_l,
        "lambda_transverse_0qmin": lambda_t,
        "Kx_per_site": kx_per_site,
        "diamagnetic_minus_Kx_per_site": -kx_per_site,
        "rho_s_current": rho_s_current,
        "rho_s_diamagnetic": rho_s_diamagnetic,
        "bkt_residual_current": rho_s_current - jump,
        "bkt_residual_diamagnetic": rho_s_diamagnetic - jump,
        "qmin": qxmin,
        "znorm_shifted": z_shifted,
    }


def annihilation_up_k_matrix(
    lx: int,
    ly: int,
    kx: float,
    ky: float,
    up_basis_n: list[int],
    up_basis_np1: list[int],
    dn_dim: int,
) -> csr_matrix:
    nsites = lx * ly
    dim_n = len(up_basis_n) * dn_dim
    dim_np1 = len(up_basis_np1) * dn_dim
    id_dn = eye(dn_dim, format="csr")
    out = csr_matrix((dim_n, dim_np1), dtype=np.complex128)
    for y in range(ly):
        for x in range(lx):
            i = site(x, y, lx)
            phase = np.exp(-1j * (kx * x + ky * y)) / math.sqrt(nsites)
            cup = annihilation_up_matrix(i, up_basis_n, up_basis_np1)
            out = out + phase * kron(id_dn, cup, format="csr")
    return out.tocsr()


def realspace_from_momentum(gk_tau: np.ndarray) -> np.ndarray:
    ntau, lx, ly = gk_tau.shape
    gr_tau = np.zeros_like(gk_tau, dtype=np.complex128)
    volume = lx * ly
    for l in range(ntau):
        for dy in range(ly):
            for dx in range(lx):
                acc = 0.0j
                for ny in range(ly):
                    ky = 2.0 * math.pi * ny / ly
                    for nx in range(lx):
                        kx = 2.0 * math.pi * nx / lx
                        acc += np.exp(1j * (kx * dx + ky * dy)) * gk_tau[l, nx, ny]
                gr_tau[l, dx, dy] = acc / volume
    return gr_tau


def full_spectrum_momentum_reference(
    lx: int,
    ly: int,
    nup: int,
    ndn: int,
    t: float,
    u: float,
    beta: float,
    dtau: float,
    measure_bkt: bool = True,
):
    h_n, up_basis_n, dn_basis = build_sector(lx, ly, nup, ndn, t, u)
    h_np1, up_basis_np1, dn_basis_np1 = build_sector(lx, ly, nup + 1, ndn, t, u)
    if dn_basis != dn_basis_np1:
        raise RuntimeError("down-spin bases differ between addition sectors")

    projectors_n = build_translation_projectors(lx, ly, up_basis_n, dn_basis)
    projectors_np1 = build_translation_projectors(lx, ly, up_basis_np1, dn_basis_np1)
    blocks_n = diagonalize_translation_blocks(h_n, projectors_n)
    blocks_np1 = diagonalize_translation_blocks(h_np1, projectors_np1)
    bkt = bkt_observables_from_blocks(lx, ly, t, beta, up_basis_n, dn_basis, projectors_n, blocks_n) if measure_bkt else None

    all_e_n = np.concatenate([evals for evals, _ in blocks_n.values()])
    e0_n = float(all_e_n.min())
    z_shifted = np.exp(-beta * (all_e_n - e0_n))
    znorm_shifted = float(np.sum(z_shifted))

    ntau = round(beta / dtau)
    tau_vals = dtau * np.arange(ntau, dtype=float)
    gk_tau = np.zeros((ntau, lx, ly), dtype=np.complex128)

    dn_dim = len(dn_basis)
    for ny in range(ly):
        ky = 2.0 * math.pi * ny / ly
        for nx in range(lx):
            kx = 2.0 * math.pi * nx / lx
            c_k = annihilation_up_k_matrix(lx, ly, kx, ky, up_basis_n, up_basis_np1, dn_dim)
            g_tau = np.zeros(ntau, dtype=np.float64)

            for key_np1, q_np1 in projectors_np1.items():
                c_q_np1 = c_k @ q_np1
                evals_np1, vecs_np1 = blocks_np1[key_np1]
                for key_n, q_n in projectors_n.items():
                    block_op_sparse = q_n.conj().T @ c_q_np1
                    if block_op_sparse.nnz == 0:
                        continue
                    block_op = block_op_sparse.toarray()
                    if np.linalg.norm(block_op) < 1e-10:
                        continue
                    evals_n, vecs_n = blocks_n[key_n]
                    m = vecs_n.conj().T @ block_op @ vecs_np1
                    spectral = np.abs(m) ** 2
                    if float(spectral.sum()) < 1e-18:
                        continue
                    for l, tau in enumerate(tau_vals):
                        wn = np.exp(-(beta - tau) * (evals_n - e0_n))
                        wnp1 = np.exp(-tau * (evals_np1 - e0_n))
                        g_tau[l] += float(wn @ (spectral @ wnp1))

            gk_tau[:, nx, ny] = g_tau / znorm_shifted

    gr_tau = realspace_from_momentum(gk_tau)
    meta = {
        "method": "full_spectrum_translation_blocks",
        "sector_n_dim": int(h_n.shape[0]),
        "sector_np1_dim": int(h_np1.shape[0]),
        "block_dims_n": {f"{k[0]},{k[1]}": int(q.shape[1]) for k, q in projectors_n.items()},
        "block_dims_np1": {f"{k[0]},{k[1]}": int(q.shape[1]) for k, q in projectors_np1.items()},
        "ground_energy_n": e0_n,
        "ground_energy_np1": float(np.concatenate([evals for evals, _ in blocks_np1.values()]).min()),
        "znorm_shifted": znorm_shifted,
    }
    return tau_vals, gr_tau, gk_tau, meta, bkt


def one_spin_equal_time_addition_gij(nsites: int, basis_n: list[int], evals_n: np.ndarray, vecs_n: np.ndarray, z_n: np.ndarray, znorm: float) -> np.ndarray:
    rho = np.zeros((nsites, nsites), dtype=np.complex128)
    for i in range(nsites):
        for j in range(nsites):
            op = bilinear_from_terms(nsites, basis_n, [(j, i, 1.0)]).toarray()
            mat = vecs_n.conj().T @ op @ vecs_n
            rho[i, j] = np.sum(z_n * np.diag(mat)) / znorm
    return np.eye(nsites, dtype=np.complex128) - rho


def spinful_equal_time_addition_gij(
    nsites: int,
    up_basis_n: list[int],
    dn_dim: int,
    evals_n: np.ndarray,
    vecs_n: np.ndarray,
    z_n: np.ndarray,
    znorm: float,
) -> np.ndarray:
    rho = np.zeros((nsites, nsites), dtype=np.complex128)
    id_dn = eye(dn_dim, format="csr")
    for i in range(nsites):
        for j in range(nsites):
            op_up = bilinear_from_terms(nsites, up_basis_n, [(j, i, 1.0)])
            op = kron(id_dn, op_up, format="csr")
            op_vecs = op @ vecs_n
            diag = np.einsum("ij,ij->j", vecs_n.conj(), op_vecs)
            rho[i, j] = np.sum(z_n * diag) / znorm
    return np.eye(nsites, dtype=np.complex128) - rho


def free_factorized_reference(
    lx: int,
    ly: int,
    nup: int,
    t: float,
    beta: float,
    dtau: float,
    nfreq: int,
):
    nsites = lx * ly
    tmat = hopping(lx, ly, t)
    up_basis_n = gen_basis(nsites, nup)
    up_basis_np1 = gen_basis(nsites, nup + 1)
    up_basis_nm1 = gen_basis(nsites, nup - 1)

    h_n = dense_one_spin_hamiltonian(tmat, up_basis_n)
    h_np1 = dense_one_spin_hamiltonian(tmat, up_basis_np1)
    h_nm1 = dense_one_spin_hamiltonian(tmat, up_basis_nm1)

    evals_n, vecs_n = np.linalg.eigh(h_n)
    evals_np1, vecs_np1 = np.linalg.eigh(h_np1)
    evals_nm1, vecs_nm1 = np.linalg.eigh(h_nm1)

    z_n = np.exp(-beta * evals_n)
    znorm = float(np.sum(z_n))
    gij_tau0 = one_spin_equal_time_addition_gij(nsites, up_basis_n, evals_n, vecs_n, z_n, znorm)

    c_mats = []
    a_mats = []
    for i in range(nsites):
        cup = annihilation_up_matrix(i, up_basis_n, up_basis_np1).toarray()
        c_mats.append(vecs_n.conj().T @ cup @ vecs_np1)
        aup = annihilation_up_matrix(i, up_basis_nm1, up_basis_n).toarray()
        a_mats.append(vecs_nm1.conj().T @ aup @ vecs_n)

    spectral_add = np.zeros((len(evals_n), len(evals_np1)), dtype=np.float64)
    for m in c_mats:
        spectral_add += np.abs(m) ** 2
    spectral_add /= nsites

    spectral_rem = np.zeros((len(evals_n), len(evals_nm1)), dtype=np.float64)
    for m in a_mats:
        spectral_rem += np.abs(m.T) ** 2
    spectral_rem /= nsites

    ntau = round(beta / dtau)
    tau_vals = dtau * np.arange(ntau, dtype=float)
    gr_tau, gk_tau = addition_space_tau_from_site_mats(
        lx, ly, beta, tau_vals, evals_n, evals_np1, z_n, znorm, c_mats, gij_tau0=gij_tau0
    )
    c_tau = gr_tau[:, 0, 0].real

    rows = []
    for n in range(nfreq):
        omega_n = (2 * n + 1) * math.pi / beta
        add = np.sum((z_n[:, None] * spectral_add) / (1j * omega_n + evals_n[:, None] - evals_np1[None, :])) / znorm
        rem = np.sum((z_n[:, None] * spectral_rem) / (1j * omega_n - evals_n[:, None] + evals_nm1[None, :])) / znorm
        g_iwn = add + rem
        rows.append((n, omega_n, g_iwn.real, g_iwn.imag))

    meta = {
        "method": "free_factorized_exact",
        "subspace_n_dim": int(len(evals_n)),
        "subspace_np1_dim": int(len(evals_np1)),
        "subspace_nm1_dim": int(len(evals_nm1)),
        "ground_energy_n": float(evals_n[0]),
        "ground_energy_np1": float(evals_np1[0]),
        "ground_energy_nm1": float(evals_nm1[0]),
    }
    return tau_vals, c_tau, rows, meta, gr_tau, gk_tau


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.bkt_only:
        h_n, up_basis_n, dn_basis = build_sector(args.lx, args.ly, args.nup, args.ndn, args.t, args.u)
        projectors_n = build_translation_projectors(args.lx, args.ly, up_basis_n, dn_basis)
        blocks_n = diagonalize_translation_blocks(h_n, projectors_n)
        bkt = bkt_observables_from_blocks(
            args.lx, args.ly, args.t, args.beta, up_basis_n, dn_basis, projectors_n, blocks_n
        )
        write_bkt_observables_ed(args.outdir, bkt, "full_spectrum_translation_blocks_bkt_only")
        all_e_n = np.concatenate([evals for evals, _ in blocks_n.values()])
        with open(args.outdir / "metadata.toml", "w", newline="") as f:
            f.write('target = "canonical ED BKT/superfluid-stiffness benchmark"\n')
            f.write(f"\n[lattice]\nlx = {args.lx}\nly = {args.ly}\n")
            f.write(f"\n[particles]\nnup = {args.nup}\nndn = {args.ndn}\n")
            f.write(f"\n[model]\nt = {args.t}\nu = {args.u}\nbeta = {args.beta}\ndtau = {args.dtau}\n")
            f.write('\n[reference]\nmethod = "full_spectrum_translation_blocks_bkt_only"\n')
            f.write(f"sector_n_dim = {h_n.shape[0]}\n")
            f.write(f"ground_energy_n = {float(all_e_n.min())}\n")
            f.write(f"znorm_shifted = {bkt['znorm_shifted']}\n")
            f.write("\n[bkt]\n")
            f.write('output_file = "bkt_observables_ed.tsv"\n')
            f.write('rho_s_current = "0.25 * (lambda_longitudinal_qmin0 - lambda_transverse_0qmin)"\n')
            f.write('rho_s_diamagnetic = "0.25 * (-Kx_per_site - lambda_transverse_0qmin)"\n')
            f.write('bkt_universal_jump_2T_over_pi = "2 / (pi * beta)"\n')
            f.write("\n[block_dims_n]\n")
            for key, q in projectors_n.items():
                f.write(f'"{key[0]},{key[1]}" = {q.shape[1]}\n')
        print(f"Computed ED BKT observables at {args.outdir / 'bkt_observables_ed.tsv'}")
        return

    if args.full_spectrum:
        tau_vals, gr_tau, gk_tau, meta, bkt = full_spectrum_momentum_reference(
            args.lx, args.ly, args.nup, args.ndn, args.t, args.u, args.beta, args.dtau,
            measure_bkt=args.measure_bkt,
        )
        write_space_tau_tables(args.outdir, tau_vals, gr_tau, gk_tau, "ed")
        if bkt is not None:
            write_bkt_observables_ed(args.outdir, bkt, "full_spectrum_translation_blocks")
        with open(args.outdir / "greens_tau0_ed.tsv", "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["slice", "tau", "Ctau_ed"])
            for l, tau in enumerate(tau_vals):
                w.writerow([l, tau, gr_tau[l, 0, 0].real])

        with open(args.outdir / "metadata.toml", "w", newline="") as f:
            f.write('target = "3x3 canonical ED full-spectrum G(r,tau)/G(k,tau) benchmark"\n')
            f.write(f"\n[lattice]\nlx = {args.lx}\nly = {args.ly}\n")
            f.write(f"\n[particles]\nnup = {args.nup}\nndn = {args.ndn}\n")
            f.write(f"\n[model]\nt = {args.t}\nu = {args.u}\nbeta = {args.beta}\ndtau = {args.dtau}\n")
            f.write(f'\n[reference]\nmethod = "{meta["method"]}"\n')
            f.write(f"sector_n_dim = {meta['sector_n_dim']}\n")
            f.write(f"sector_np1_dim = {meta['sector_np1_dim']}\n")
            f.write(f"ground_energy_n = {meta['ground_energy_n']}\n")
            f.write(f"ground_energy_np1 = {meta['ground_energy_np1']}\n")
            f.write(f"znorm_shifted = {meta['znorm_shifted']}\n")
            if bkt is not None:
                f.write("\n[bkt]\n")
                f.write('output_file = "bkt_observables_ed.tsv"\n')
                f.write('rho_s_current = "0.25 * (lambda_longitudinal_qmin0 - lambda_transverse_0qmin)"\n')
                f.write('rho_s_diamagnetic = "0.25 * (-Kx_per_site - lambda_transverse_0qmin)"\n')
                f.write('bkt_universal_jump_2T_over_pi = "2 / (pi * beta)"\n')
            f.write("\n[block_dims_n]\n")
            for key, val in meta["block_dims_n"].items():
                f.write(f'"{key}" = {val}\n')
            f.write("\n[block_dims_np1]\n")
            for key, val in meta["block_dims_np1"].items():
                f.write(f'"{key}" = {val}\n')
        return

    if abs(args.u) < 1e-12:
        tau_vals, c_tau, rows, meta, gr_tau, gk_tau = free_factorized_reference(
            args.lx, args.ly, args.nup, args.t, args.beta, args.dtau, args.nfreq
        )
        write_space_tau_tables(args.outdir, tau_vals, gr_tau, gk_tau, "ed")

        with open(args.outdir / "greens_tau0_ed.tsv", "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["slice", "tau", "Ctau_ed"])
            for l, tau in enumerate(tau_vals):
                w.writerow([l, tau, c_tau[l]])

        with open(args.outdir / "greens_iwn_ed.tsv", "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["n", "omega_n", "Greal_ed", "Gimag_ed"])
            w.writerows(rows)

        with open(args.outdir / "metadata.toml", "w", newline="") as f:
            f.write('target = "3x3 canonical ED Matsubara Green benchmark"\n')
            f.write(f"\n[lattice]\nlx = {args.lx}\nly = {args.ly}\n")
            f.write(f"\n[particles]\nnup = {args.nup}\nndn = {args.ndn}\n")
            f.write(f"\n[model]\nt = {args.t}\nu = {args.u}\nbeta = {args.beta}\ndtau = {args.dtau}\n")
            f.write(f'\n[reference]\nmethod = "{meta["method"]}"\n')
            f.write(f"subspace_n_dim = {meta['subspace_n_dim']}\n")
            f.write(f"subspace_np1_dim = {meta['subspace_np1_dim']}\n")
            f.write(f"subspace_nm1_dim = {meta['subspace_nm1_dim']}\n")
            f.write(f"ground_energy_n = {meta['ground_energy_n']}\n")
            f.write(f"ground_energy_np1 = {meta['ground_energy_np1']}\n")
            f.write(f"ground_energy_nm1 = {meta['ground_energy_nm1']}\n")
        return

    h_n, up_basis_n, dn_basis = build_sector(args.lx, args.ly, args.nup, args.ndn, args.t, args.u)
    h_np1, up_basis_np1, dn_basis_np1 = build_sector(args.lx, args.ly, args.nup + 1, args.ndn, args.t, args.u)
    h_nm1, up_basis_nm1, dn_basis_nm1 = build_sector(args.lx, args.ly, args.nup - 1, args.ndn, args.t, args.u)
    assert len(dn_basis) == len(dn_basis_np1) == len(dn_basis_nm1)

    evals_n, vecs_n, k_n, e0_n = low_energy_subspace(h_n, args.energy_window, args.initial_k, args.max_k)
    evals_np1, vecs_np1, k_np1, e0_np1 = low_energy_subspace(h_np1, args.energy_window, args.initial_k, args.max_k)
    evals_nm1, vecs_nm1, k_nm1, e0_nm1 = low_energy_subspace(h_nm1, args.energy_window, args.initial_k, args.max_k)

    β = args.beta
    z_n = np.exp(-β * evals_n)
    znorm = float(np.sum(z_n))
    gij_tau0 = spinful_equal_time_addition_gij(
        args.lx * args.ly, up_basis_n, len(dn_basis), evals_n, vecs_n, z_n, znorm
    )

    dd = len(dn_basis)
    c_mats = []
    a_mats = []
    id_dn = eye(dd, format="csr")
    for i in range(args.lx * args.ly):
        cup = annihilation_up_matrix(i, up_basis_n, up_basis_np1)
        c_tot = kron(id_dn, cup, format="csr")
        m = vecs_n.conj().T @ (c_tot @ vecs_np1)
        c_mats.append(np.asarray(m))
        a_tot = kron(id_dn, annihilation_up_matrix(i, up_basis_nm1, up_basis_n), format="csr")
        am = vecs_nm1.conj().T @ (a_tot @ vecs_n)
        a_mats.append(np.asarray(am))

    spectral = np.zeros((len(evals_n), len(evals_np1)), dtype=np.float64)
    for m in c_mats:
        spectral += np.abs(m) ** 2
    spectral /= (args.lx * args.ly)

    spectral_remove = np.zeros((len(evals_n), len(evals_nm1)), dtype=np.float64)
    for m in a_mats:
        spectral_remove += np.abs(m.T) ** 2
    spectral_remove /= (args.lx * args.ly)

    ntau = round(β / args.dtau)
    tau_vals = args.dtau * np.arange(ntau, dtype=float)
    gr_tau, gk_tau = addition_space_tau_from_site_mats(
        args.lx, args.ly, β, tau_vals, evals_n, evals_np1, z_n, znorm, c_mats, gij_tau0=gij_tau0
    )
    c_tau = gr_tau[:, 0, 0].real
    write_space_tau_tables(args.outdir, tau_vals, gr_tau, gk_tau, "ed")

    with open(args.outdir / "greens_tau0_ed.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["slice", "tau", "Ctau_ed"])
        for l, τ in enumerate(tau_vals):
            w.writerow([l, τ, c_tau[l]])

    rows = []
    for n in range(args.nfreq):
        ωn = (2 * n + 1) * math.pi / β
        add = np.sum((z_n[:, None] * spectral) / (1j * ωn + evals_n[:, None] - evals_np1[None, :])) / znorm
        rem = np.sum((z_n[:, None] * spectral_remove) / (1j * ωn - evals_n[:, None] + evals_nm1[None, :])) / znorm
        g_iwn = add + rem
        rows.append((n, ωn, g_iwn.real, g_iwn.imag))

    with open(args.outdir / "greens_iwn_ed.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["n", "omega_n", "Greal_ed", "Gimag_ed"])
        w.writerows(rows)

    bkt = None
    if args.measure_bkt:
        bkt = bkt_observables_from_dense_subspace(
            args.lx, args.ly, args.t, β, up_basis_n, dn_basis, evals_n, vecs_n
        )
        write_bkt_observables_ed(args.outdir, bkt, "low_energy_dense_subspace")

    meta = {
        "target": "3x3 canonical ED Matsubara Green benchmark",
        "lattice": {"lx": args.lx, "ly": args.ly},
        "particles": {"nup": args.nup, "ndn": args.ndn},
        "model": {"t": args.t, "u": args.u, "beta": args.beta, "dtau": args.dtau},
        "subspace_n": {"used_k": int(k_n), "retained": int(len(evals_n)), "ground_energy": float(e0_n)},
        "subspace_np1": {"used_k": int(k_np1), "retained": int(len(evals_np1)), "ground_energy": float(e0_np1)},
    }
    import tomllib  # type: ignore
    del tomllib  # quiet linters; TOML write by hand below
    with open(args.outdir / "metadata.toml", "w", newline="") as f:
        f.write(f'target = "{meta["target"]}"\n')
        f.write(f"\n[lattice]\nlx = {args.lx}\nly = {args.ly}\n")
        f.write(f"\n[particles]\nnup = {args.nup}\nndn = {args.ndn}\n")
        f.write(f"\n[model]\nt = {args.t}\nu = {args.u}\nbeta = {args.beta}\ndtau = {args.dtau}\n")
        f.write(
            f'\n[subspace_n]\nused_k = {int(k_n)}\nretained = {int(len(evals_n))}\nground_energy = {float(e0_n)}\n'
        )
        f.write(
            f'\n[subspace_np1]\nused_k = {int(k_np1)}\nretained = {int(len(evals_np1))}\nground_energy = {float(e0_np1)}\n'
        )
        f.write(
            f'\n[subspace_nm1]\nused_k = {int(k_nm1)}\nretained = {int(len(evals_nm1))}\nground_energy = {float(e0_nm1)}\n'
        )
        if bkt is not None:
            f.write("\n[bkt]\n")
            f.write('output_file = "bkt_observables_ed.tsv"\n')
            f.write('rho_s_current = "0.25 * (lambda_longitudinal_qmin0 - lambda_transverse_0qmin)"\n')
            f.write('rho_s_diamagnetic = "0.25 * (-Kx_per_site - lambda_transverse_0qmin)"\n')
            f.write('bkt_universal_jump_2T_over_pi = "2 / (pi * beta)"\n')


if __name__ == "__main__":
    main()
