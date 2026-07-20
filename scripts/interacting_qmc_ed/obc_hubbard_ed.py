#!/usr/bin/env python3
"""Exact-diagonalization oracle for square-lattice OBC CE/GCE benchmarks.

The Hamiltonian used to construct sector eigenvectors is

    H_std = -t sum_<ij>,sigma (c^dagger_i c_j + h.c.)
            + U sum_i n_{i,up} n_{i,down}.

Canonical observables are independent of the particle-hole-symmetric constant.
For grand-canonical comparisons to the SmoQyDQMC drivers we add

    U * (V/4 - N/2) - mu * N

to every fixed-(Nup,Ndn) eigenvalue, matching ``ph_sym_form=true``.  The module
uses the same x-fastest site ordering and physical NN/NNN bond normalization as
``square_lattice_geometry.jl``.

For 2x2 every sector is diagonalized completely.  For 3x3, large sectors use
the lowest eigenpairs and report an explicit conservative omitted-Boltzmann-
weight bound.  A result is never marked accepted unless this bound is below the
requested tolerance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh
from scipy.optimize import brentq
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, diags, eye, kron
from scipy.sparse.linalg import eigsh
from scipy.special import logsumexp


CACHE_VERSION = 3
PRIMARY_OBSERVABLES = (
    "kinetic_per_site",
    "double_occupancy_per_site",
    "nn_spin_s_s",
    "nn_connected_charge",
)


@dataclass(frozen=True)
class SquareGeometry:
    lx: int
    ly: int
    t: float
    tprime: float
    coordinates: tuple[tuple[int, int], ...]
    nn_bonds: tuple[tuple[int, int], ...]
    nnn_bonds: tuple[tuple[int, int], ...]
    hopping: NDArray[np.float64]

    @property
    def nsites(self) -> int:
        return self.lx * self.ly

    @property
    def digest(self) -> str:
        payload = {
            "lx": self.lx,
            "ly": self.ly,
            "t": self.t,
            "tprime": self.tprime,
            "coordinates": self.coordinates,
            "nn_bonds": self.nn_bonds,
            "nnn_bonds": self.nnn_bonds,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@dataclass
class SectorSpectrum:
    nup: int
    ndn: int
    interaction: float
    dimension: int
    eigenvalues: NDArray[np.float64]
    observables: dict[str, NDArray[np.float64]]
    site_density: NDArray[np.float64]
    complete: bool
    first_omitted_energy_lower: float
    omitted_counts: NDArray[np.int64]
    omitted_energy_lowers: NDArray[np.float64]
    max_residual: float
    method: str

    @property
    def ntotal(self) -> int:
        return self.nup + self.ndn

    @property
    def nkept(self) -> int:
        return len(self.eigenvalues)


def site_index(x: int, y: int, lx: int) -> int:
    return x + y * lx


def build_square_geometry(
    lx: int,
    ly: int,
    *,
    t: float = 1.0,
    tprime: float = 0.0,
) -> SquareGeometry:
    """Build a square lattice open in both directions."""
    if lx <= 0 or ly <= 0:
        raise ValueError("lx and ly must be positive")
    coordinates = tuple((x, y) for y in range(ly) for x in range(lx))
    nn: set[tuple[int, int]] = set()
    nnn: set[tuple[int, int]] = set()
    for y in range(ly):
        for x in range(lx):
            i = site_index(x, y, lx)
            if x + 1 < lx:
                nn.add(tuple(sorted((i, site_index(x + 1, y, lx)))))
            if y + 1 < ly:
                nn.add(tuple(sorted((i, site_index(x, y + 1, lx)))))
            if x + 1 < lx:
                if y + 1 < ly:
                    nnn.add(tuple(sorted((i, site_index(x + 1, y + 1, lx)))))
                if y - 1 >= 0:
                    nnn.add(tuple(sorted((i, site_index(x + 1, y - 1, lx)))))

    nsites = lx * ly
    hopping = np.zeros((nsites, nsites), dtype=float)
    for i, j in sorted(nn):
        hopping[i, j] = hopping[j, i] = -float(t)
    for i, j in sorted(nnn):
        hopping[i, j] += -float(tprime)
        hopping[j, i] += -float(tprime)

    geometry = SquareGeometry(
        lx=lx,
        ly=ly,
        t=float(t),
        tprime=float(tprime),
        coordinates=coordinates,
        nn_bonds=tuple(sorted(nn)),
        nnn_bonds=tuple(sorted(nnn)),
        hopping=hopping,
    )
    validate_geometry(geometry)
    return geometry


def validate_geometry(geometry: SquareGeometry) -> None:
    expected_nn = (geometry.lx - 1) * geometry.ly + geometry.lx * (geometry.ly - 1)
    expected_nnn = 2 * (geometry.lx - 1) * (geometry.ly - 1)
    if len(geometry.nn_bonds) != expected_nn:
        raise AssertionError(f"NN count {len(geometry.nn_bonds)} != {expected_nn}")
    if len(geometry.nnn_bonds) != expected_nnn:
        raise AssertionError(f"NNN count {len(geometry.nnn_bonds)} != {expected_nnn}")
    if not np.array_equal(geometry.hopping, geometry.hopping.T.conj()):
        raise AssertionError("hopping matrix is not Hermitian")
    for i, j in geometry.nn_bonds:
        xi, yi = geometry.coordinates[i]
        xj, yj = geometry.coordinates[j]
        if abs(xi - xj) + abs(yi - yj) != 1:
            raise AssertionError(f"periodic wrap appeared in OBC NN bond {(i, j)}")
    for i, j in geometry.nnn_bonds:
        xi, yi = geometry.coordinates[i]
        xj, yj = geometry.coordinates[j]
        if (abs(xi - xj), abs(yi - yj)) != (1, 1):
            raise AssertionError(f"periodic wrap appeared in OBC NNN bond {(i, j)}")


def generate_basis(nsites: int, nparticles: int) -> list[int]:
    basis: list[int] = []
    for occupied in combinations(range(nsites), nparticles):
        state = 0
        for i in occupied:
            state |= 1 << i
        basis.append(state)
    return basis


def _occupation_matrix(basis: list[int], nsites: int) -> NDArray[np.int8]:
    states = np.asarray(basis, dtype=np.int64)[:, None]
    shifts = np.arange(nsites, dtype=np.int64)[None, :]
    return ((states >> shifts) & 1).astype(np.int8)


def apply_cdag_c(state: int, i: int, j: int) -> tuple[int, int] | None:
    if not ((state >> j) & 1) or ((state >> i) & 1):
        return None
    mask_j = 1 << j
    after = state & ~mask_j
    sign_j = -1 if (state & (mask_j - 1)).bit_count() % 2 else 1
    mask_i = 1 << i
    sign_i = -1 if (after & (mask_i - 1)).bit_count() % 2 else 1
    return sign_j * sign_i, after | mask_i


def one_spin_hamiltonian(hopping: NDArray[np.float64], basis: list[int]) -> csr_matrix:
    index = {state: i for i, state in enumerate(basis)}
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    nz_i, nz_j = np.nonzero(hopping)
    terms = [(int(i), int(j), float(hopping[i, j])) for i, j in zip(nz_i, nz_j)]
    for col, state in enumerate(basis):
        for i, j, amplitude in terms:
            result = apply_cdag_c(state, i, j)
            if result is None:
                continue
            sign, new_state = result
            rows.append(index[new_state])
            cols.append(col)
            values.append(amplitude * sign)
    dim = len(basis)
    return coo_matrix((values, (rows, cols)), shape=(dim, dim)).tocsr()


def build_sector_hamiltonian(
    geometry: SquareGeometry,
    nup: int,
    ndn: int,
    interaction: float,
) -> tuple[
    csr_matrix,
    dict[str, NDArray[np.float64]],
    NDArray[np.float64],
    tuple[list[int], list[int]],
]:
    nsites = geometry.nsites
    up_basis = generate_basis(nsites, nup)
    dn_basis = generate_basis(nsites, ndn)
    hup = one_spin_hamiltonian(geometry.hopping, up_basis)
    hdn = one_spin_hamiltonian(geometry.hopping, dn_basis)
    du, dd = len(up_basis), len(dn_basis)
    kinetic = kron(eye(dd, format="csr"), hup) + kron(hdn, eye(du, format="csr"))

    occ_up_basis = _occupation_matrix(up_basis, nsites)
    occ_dn_basis = _occupation_matrix(dn_basis, nsites)
    occ_up = np.tile(occ_up_basis, (dd, 1)).astype(float)
    occ_dn = np.repeat(occ_dn_basis, du, axis=0).astype(float)
    density = occ_up + occ_dn
    spin_density = occ_up - occ_dn
    docc_total = np.sum(occ_up * occ_dn, axis=1)

    def pair_average(values: NDArray[np.float64], bonds: tuple[tuple[int, int], ...]) -> NDArray[np.float64]:
        if not bonds:
            return np.full(values.shape[0], np.nan)
        return np.mean(np.column_stack([values[:, i] * values[:, j] for i, j in bonds]), axis=1)

    diagonal = {
        "double_occupancy_total": docc_total,
        "nn_spin_s_s": pair_average(spin_density, geometry.nn_bonds),
        "nn_charge_raw": pair_average(density, geometry.nn_bonds),
        "nnn_spin_s_s": pair_average(spin_density, geometry.nnn_bonds),
        "nnn_charge_raw": pair_average(density, geometry.nnn_bonds),
    }
    hamiltonian = kinetic + diags(interaction * docc_total)
    return hamiltonian.tocsr(), diagonal, density, (up_basis, dn_basis)


def _expect_diagonal(
    diagonal: NDArray[np.float64], eigenvectors: NDArray[np.float64]
) -> NDArray[np.float64]:
    return np.asarray(np.abs(eigenvectors) ** 2).T @ diagonal


def _tail_fraction_bound(
    eigenvalues: NDArray[np.float64],
    omitted_counts: NDArray[np.int64],
    omitted_energy_lowers: NDArray[np.float64],
    beta: float,
) -> float:
    mask = omitted_counts > 0
    if not np.any(mask):
        return 0.0
    log_kept = float(logsumexp(-beta * eigenvalues))
    log_tail = float(
        logsumexp(
            np.log(omitted_counts[mask].astype(float))
            - beta * omitted_energy_lowers[mask]
        )
    )
    return float(math.exp(log_tail - np.logaddexp(log_kept, log_tail)))


def _permutation_parity(values: list[int]) -> int:
    inversions = sum(
        values[i] > values[j]
        for i in range(len(values))
        for j in range(i + 1, len(values))
    )
    return -1 if inversions % 2 else 1


def _permute_fock_state(state: int, site_permutation: NDArray[np.int64]) -> tuple[int, int]:
    occupied = [i for i in range(len(site_permutation)) if (state >> i) & 1]
    mapped = [int(site_permutation[i]) for i in occupied]
    sign = _permutation_parity(mapped)
    output = 0
    for i in mapped:
        output |= 1 << i
    return output, sign


def _c4_rotation_action(
    geometry: SquareGeometry,
    up_basis: list[int],
    dn_basis: list[int],
) -> tuple[NDArray[np.int64], NDArray[np.int8]]:
    """Return full-sector index/sign action of a 90-degree site rotation."""
    if geometry.lx != geometry.ly:
        raise ValueError("C4-resolved ED requires a square cluster")
    length = geometry.lx
    site_permutation = np.empty(geometry.nsites, dtype=np.int64)
    for y in range(length):
        for x in range(length):
            site_permutation[site_index(x, y, length)] = site_index(
                length - 1 - y, x, length
            )

    up_index = {state: i for i, state in enumerate(up_basis)}
    dn_index = {state: i for i, state in enumerate(dn_basis)}
    up_rot = [_permute_fock_state(state, site_permutation) for state in up_basis]
    dn_rot = [_permute_fock_state(state, site_permutation) for state in dn_basis]
    du, dd = len(up_basis), len(dn_basis)
    index_action = np.empty(du * dd, dtype=np.int64)
    sign_action = np.empty(du * dd, dtype=np.int8)
    for dn_i, (dn_state, dn_sign) in enumerate(dn_rot):
        rotated_dn = dn_index[dn_state]
        for up_i, (up_state, up_sign) in enumerate(up_rot):
            col = up_i + du * dn_i
            index_action[col] = up_index[up_state] + du * rotated_dn
            sign_action[col] = up_sign * dn_sign

    # Four applications must be exactly the identity, including Fock signs.
    for start in range(du * dd):
        idx = start
        sign = 1
        for _ in range(4):
            sign *= int(sign_action[idx])
            idx = int(index_action[idx])
        if idx != start or sign != 1:
            raise AssertionError("fermionic C4 action does not satisfy R^4=1")
    return index_action, sign_action


def _c4_symmetry_bases(
    dimension: int,
    index_action: NDArray[np.int64],
    sign_action: NDArray[np.int8],
) -> list[csc_matrix]:
    """Build orthonormal sparse bases for the four C4 eigenvalue sectors."""
    rows: list[list[int]] = [[] for _ in range(4)]
    cols: list[list[int]] = [[] for _ in range(4)]
    data: list[list[complex]] = [[] for _ in range(4)]
    ncolumns = [0, 0, 0, 0]
    visited: set[int] = set()

    for representative in range(dimension):
        if representative in visited:
            continue
        orbit: list[int] = []
        idx = representative
        for _ in range(4):
            orbit.append(idx)
            visited.add(idx)
            idx = int(index_action[idx])

        for momentum in range(4):
            eigenvalue = np.exp(0.5j * math.pi * momentum)
            coefficients: dict[int, complex] = {}
            idx = representative
            cumulative_sign = 1
            for power in range(4):
                coefficients[idx] = coefficients.get(idx, 0.0j) + (
                    eigenvalue ** (-power) * cumulative_sign
                )
                cumulative_sign *= int(sign_action[idx])
                idx = int(index_action[idx])
            norm = math.sqrt(sum(abs(value) ** 2 for value in coefficients.values()))
            if norm < 1e-12:
                continue
            column = ncolumns[momentum]
            ncolumns[momentum] += 1
            for row, value in coefficients.items():
                if abs(value) > 1e-13:
                    rows[momentum].append(row)
                    cols[momentum].append(column)
                    data[momentum].append(value / norm)

    bases = [
        coo_matrix(
            (data[m], (rows[m], cols[m])),
            shape=(dimension, ncolumns[m]),
            dtype=np.complex128,
        ).tocsc()
        for m in range(4)
    ]
    if sum(base.shape[1] for base in bases) != dimension:
        raise AssertionError("C4 symmetry bases do not span the full fixed-N sector")
    for base in bases:
        gram = base.conj().T @ base
        deviation = gram - eye(base.shape[1], dtype=np.complex128, format="csc")
        if deviation.nnz and np.max(np.abs(deviation.data)) > 1e-11:
            raise AssertionError("C4 symmetry basis is not orthonormal")
    return bases


def _solve_low_block(
    hamiltonian: csr_matrix | csc_matrix,
    beta: float,
    *,
    tolerance: float,
    initial_keep: int,
    maximum_keep: int,
    dense_threshold: int,
    seed: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.complex128],
    int,
    float,
    float,
    str,
]:
    block_dimension = hamiltonian.shape[0]
    if block_dimension <= dense_threshold:
        values, vectors = eigh(hamiltonian.toarray(), check_finite=False)
        residual = float(
            np.max(np.linalg.norm(hamiltonian @ vectors - vectors * values, axis=0))
        )
        return values, vectors, 0, math.inf, residual, "dense"

    keep = min(max(2, initial_keep), block_dimension - 2)
    while True:
        requested = min(keep + 1, block_dimension - 1)
        rng = np.random.default_rng(seed + keep)
        v0 = rng.standard_normal(block_dimension) + 1j * rng.standard_normal(block_dimension)
        ncv = min(block_dimension, max(2 * requested + 20, requested + 40))
        values, vectors = eigsh(
            hamiltonian,
            k=requested,
            which="SA",
            tol=1e-11,
            maxiter=max(20000, 20 * block_dimension),
            ncv=ncv,
            v0=v0,
        )
        order = np.argsort(values)
        values = np.asarray(values[order], dtype=float)
        vectors = np.asarray(vectors[:, order], dtype=np.complex128)
        residuals = np.linalg.norm(hamiltonian @ vectors - vectors * values, axis=0)
        kept_values = values[:-1]
        kept_vectors = vectors[:, :-1]
        omitted = block_dimension - len(kept_values)
        guard = float(values[-1] - max(float(residuals[-1]), 1e-11))
        tail = _tail_fraction_bound(
            kept_values,
            np.asarray([omitted], dtype=np.int64),
            np.asarray([guard], dtype=float),
            beta,
        )
        if tail <= tolerance:
            return (
                kept_values,
                kept_vectors,
                omitted,
                guard,
                float(np.max(residuals)),
                f"eigsh-k{len(kept_values)}",
            )
        next_keep = min(
            maximum_keep,
            block_dimension - 2,
            max(2 * keep, keep + 24),
        )
        if next_keep <= keep:
            return (
                kept_values,
                kept_vectors,
                omitted,
                guard,
                float(np.max(residuals)),
                f"eigsh-incomplete-k{len(kept_values)}",
            )
        keep = next_keep


def diagonalize_sector(
    geometry: SquareGeometry,
    nup: int,
    ndn: int,
    interaction: float,
    beta: float,
    *,
    boltzmann_tolerance: float = 1e-8,
    dense_threshold: int = 1800,
    initial_k: int = 64,
    max_k: int = 1024,
) -> SectorSpectrum:
    hamiltonian, diagonals, site_density_diagonal, bases = build_sector_hamiltonian(
        geometry, nup, ndn, interaction
    )
    dimension = hamiltonian.shape[0]
    if dimension <= dense_threshold:
        eigenvalues, eigenvectors = eigh(hamiltonian.toarray(), check_finite=False)
        complete = True
        first_omitted = math.inf
        omitted_counts = np.zeros(0, dtype=np.int64)
        omitted_energy_lowers = np.zeros(0, dtype=float)
        max_residual = float(
            np.max(np.linalg.norm(hamiltonian @ eigenvectors - eigenvectors * eigenvalues, axis=0))
        )
        method = "dense-full"
        docc = _expect_diagonal(diagonals["double_occupancy_total"], eigenvectors)
        observable_eigenvalues: dict[str, NDArray[np.float64]] = {
            "kinetic_per_site": (eigenvalues - interaction * docc) / geometry.nsites,
            "double_occupancy_per_site": docc / geometry.nsites,
            "nn_spin_s_s": _expect_diagonal(diagonals["nn_spin_s_s"], eigenvectors),
            "nn_charge_raw": _expect_diagonal(diagonals["nn_charge_raw"], eigenvectors),
            "nnn_spin_s_s": _expect_diagonal(diagonals["nnn_spin_s_s"], eigenvectors),
            "nnn_charge_raw": _expect_diagonal(diagonals["nnn_charge_raw"], eigenvectors),
        }
        site_density = _expect_diagonal(site_density_diagonal, eigenvectors)
    else:
        # Resolve the exact C4 point-group multiplicities before using Lanczos.
        # Reflection maps m=1 <-> m=3, so retaining both C4 sectors restores the
        # two-dimensional D4 E irrep without relying on a single Krylov vector
        # to discover degenerate partners.
        rotation_index, rotation_sign = _c4_rotation_action(geometry, *bases)
        symmetry_bases = _c4_symmetry_bases(
            dimension, rotation_index, rotation_sign
        )
        values_all: list[NDArray[np.float64]] = []
        observables_all: dict[str, list[NDArray[np.float64]]] = {
            "double_occupancy_per_site": [],
            "nn_spin_s_s": [],
            "nn_charge_raw": [],
            "nnn_spin_s_s": [],
            "nnn_charge_raw": [],
        }
        site_all: list[NDArray[np.float64]] = []
        omitted_count_list: list[int] = []
        omitted_lower_list: list[float] = []
        residual_list: list[float] = []
        method_parts: list[str] = []
        for momentum, symmetry_basis in enumerate(symmetry_bases):
            block_dimension = symmetry_basis.shape[1]
            block_hamiltonian = (
                symmetry_basis.conj().T @ hamiltonian @ symmetry_basis
            ).tocsr()
            initial_block = max(
                2, math.ceil(initial_k * block_dimension / dimension)
            )
            maximum_block = max(
                initial_block,
                math.ceil(max_k * block_dimension / dimension),
            )
            values, block_vectors, omitted, guard, residual, block_method = _solve_low_block(
                block_hamiltonian,
                beta,
                tolerance=boltzmann_tolerance,
                initial_keep=initial_block,
                maximum_keep=maximum_block,
                dense_threshold=max(64, dense_threshold // 2),
                seed=(7919 + 101 * momentum + 17 * nup + 31 * ndn),
            )
            full_vectors = symmetry_basis @ block_vectors
            docc_block = _expect_diagonal(
                diagonals["double_occupancy_total"], full_vectors
            )
            values_all.append(values)
            observables_all["double_occupancy_per_site"].append(
                docc_block / geometry.nsites
            )
            for name in (
                "nn_spin_s_s",
                "nn_charge_raw",
                "nnn_spin_s_s",
                "nnn_charge_raw",
            ):
                observables_all[name].append(
                    _expect_diagonal(diagonals[name], full_vectors)
                )
            site_all.append(_expect_diagonal(site_density_diagonal, full_vectors))
            omitted_count_list.append(omitted)
            omitted_lower_list.append(guard)
            residual_list.append(residual)
            method_parts.append(f"m{momentum}:{block_method}")

        eigenvalues_unsorted = np.concatenate(values_all)
        order = np.argsort(eigenvalues_unsorted)
        eigenvalues = eigenvalues_unsorted[order]
        docc_per_site = np.concatenate(
            observables_all["double_occupancy_per_site"]
        )[order]
        observable_eigenvalues = {
            "kinetic_per_site": (
                eigenvalues - interaction * geometry.nsites * docc_per_site
            )
            / geometry.nsites,
            "double_occupancy_per_site": docc_per_site,
        }
        for name in (
            "nn_spin_s_s",
            "nn_charge_raw",
            "nnn_spin_s_s",
            "nnn_charge_raw",
        ):
            observable_eigenvalues[name] = np.concatenate(observables_all[name])[order]
        site_density = np.vstack(site_all)[order]
        omitted_counts = np.asarray(omitted_count_list, dtype=np.int64)
        omitted_energy_lowers = np.asarray(omitted_lower_list, dtype=float)
        complete = bool(np.sum(omitted_counts) == 0)
        first_omitted = (
            math.inf
            if complete
            else float(np.min(omitted_energy_lowers[omitted_counts > 0]))
        )
        max_residual = float(max(residual_list, default=0.0))
        method = "c4-resolved[" + ",".join(method_parts) + "]"

    return SectorSpectrum(
        nup=nup,
        ndn=ndn,
        interaction=float(interaction),
        dimension=dimension,
        eigenvalues=eigenvalues,
        observables=observable_eigenvalues,
        site_density=site_density,
        complete=complete,
        first_omitted_energy_lower=first_omitted,
        omitted_counts=omitted_counts,
        omitted_energy_lowers=omitted_energy_lowers,
        max_residual=max_residual,
        method=method,
    )


def spectrum_tail_fraction(spectrum: SectorSpectrum, beta: float) -> float:
    return _tail_fraction_bound(
        spectrum.eigenvalues,
        spectrum.omitted_counts,
        spectrum.omitted_energy_lowers,
        beta,
    )


def _connected_charge(
    raw: float,
    site_density: NDArray[np.float64],
    bonds: Iterable[tuple[int, int]],
) -> float:
    bonds = tuple(bonds)
    if not bonds:
        return math.nan
    disconnected = np.mean([site_density[i] * site_density[j] for i, j in bonds])
    return float(raw - disconnected)


def canonical_summary(
    geometry: SquareGeometry,
    spectrum: SectorSpectrum,
    beta: float,
) -> dict[str, float | int | str | bool]:
    logweights = -beta * spectrum.eigenvalues
    weights = np.exp(logweights - logsumexp(logweights))
    means = {
        name: float(np.dot(weights, values))
        for name, values in spectrum.observables.items()
    }
    site_density = weights @ spectrum.site_density
    means["nn_connected_charge"] = _connected_charge(
        means["nn_charge_raw"], site_density, geometry.nn_bonds
    )
    means["nnn_connected_charge"] = _connected_charge(
        means["nnn_charge_raw"], site_density, geometry.nnn_bonds
    )
    density = (spectrum.nup + spectrum.ndn) / geometry.nsites
    means.update(
        {
            "density": density,
            "achieved_N": float(spectrum.nup + spectrum.ndn),
            "interaction_per_site": spectrum.interaction
            * means["double_occupancy_per_site"],
            "total_energy_per_site": means["kinetic_per_site"]
            + spectrum.interaction * means["double_occupancy_per_site"],
            "local_moment": density - 2 * means["double_occupancy_per_site"],
            "tail_weight_bound": spectrum_tail_fraction(spectrum, beta),
            "dimension": spectrum.dimension,
            "states_kept": spectrum.nkept,
            "ed_method": spectrum.method,
            "max_eigen_residual": spectrum.max_residual,
        }
    )
    return means


def ph_symmetric_sector_shift(
    interaction: float, chemical_potential: float, ntotal: int, nsites: int
) -> float:
    return interaction * (0.25 * nsites - 0.5 * ntotal) - chemical_potential * ntotal


def grand_canonical_summary(
    geometry: SquareGeometry,
    spectra: Mapping[tuple[int, int], SectorSpectrum],
    beta: float,
    chemical_potential: float,
) -> dict[str, float | int]:
    log_weight_blocks: list[NDArray[np.float64]] = []
    sector_keys: list[tuple[int, int]] = []
    log_tail_bounds: list[float] = []
    for key, spectrum in spectra.items():
        shift = ph_symmetric_sector_shift(
            spectrum.interaction, chemical_potential, spectrum.ntotal, geometry.nsites
        )
        log_weight_blocks.append(-beta * (spectrum.eigenvalues + shift))
        sector_keys.append(key)
        for omitted, lower in zip(
            spectrum.omitted_counts, spectrum.omitted_energy_lowers
        ):
            if omitted > 0:
                log_tail_bounds.append(
                    math.log(int(omitted)) - beta * (float(lower) + shift)
                )

    log_z_kept = float(logsumexp(np.concatenate(log_weight_blocks)))
    if log_tail_bounds:
        log_tail = float(logsumexp(np.asarray(log_tail_bounds)))
        tail_bound = float(
            math.exp(log_tail - np.logaddexp(log_z_kept, log_tail))
        )
    else:
        tail_bound = 0.0

    numerators = {name: 0.0 for name in next(iter(spectra.values())).observables}
    site_density = np.zeros(geometry.nsites, dtype=float)
    n_mean = 0.0
    n2_mean = 0.0
    for key, logweights in zip(sector_keys, log_weight_blocks):
        spectrum = spectra[key]
        weights = np.exp(logweights - log_z_kept)
        block_weight = float(np.sum(weights))
        n_mean += block_weight * spectrum.ntotal
        n2_mean += block_weight * spectrum.ntotal**2
        for name, values in spectrum.observables.items():
            numerators[name] += float(np.dot(weights, values))
        site_density += weights @ spectrum.site_density

    numerators["nn_connected_charge"] = _connected_charge(
        numerators["nn_charge_raw"], site_density, geometry.nn_bonds
    )
    numerators["nnn_connected_charge"] = _connected_charge(
        numerators["nnn_charge_raw"], site_density, geometry.nnn_bonds
    )
    density = n_mean / geometry.nsites
    interaction = next(iter(spectra.values())).interaction
    numerators.update(
        {
            "density": density,
            "achieved_N": n_mean,
            "N_variance": n2_mean - n_mean**2,
            "compressibility": beta
            * (n2_mean - n_mean**2)
            / geometry.nsites,
            "interaction_per_site": interaction
            * numerators["double_occupancy_per_site"],
            "total_energy_per_site": numerators["kinetic_per_site"]
            + interaction * numerators["double_occupancy_per_site"],
            "local_moment": density
            - 2 * numerators["double_occupancy_per_site"],
            "tail_weight_bound": tail_bound,
            "log_partition_kept": log_z_kept,
            "states_kept": sum(s.nkept for s in spectra.values()),
            "full_dimension": sum(s.dimension for s in spectra.values()),
        }
    )
    return numerators


def tune_chemical_potential(
    geometry: SquareGeometry,
    spectra: Mapping[tuple[int, int], SectorSpectrum],
    beta: float,
    target_n: float,
    *,
    tolerance: float = 0.01,
) -> tuple[float, dict[str, float | int], tuple[float, float]]:
    def objective(mu: float) -> float:
        return float(grand_canonical_summary(geometry, spectra, beta, mu)["achieved_N"]) - target_n

    radius = 1.0
    lo, hi = -radius, radius
    flo, fhi = objective(lo), objective(hi)
    while flo * fhi > 0 and radius < 128:
        radius *= 2
        lo, hi = -radius, radius
        flo, fhi = objective(lo), objective(hi)
    if flo * fhi > 0:
        raise RuntimeError(f"failed to bracket target N={target_n}: f({lo})={flo}, f({hi})={fhi}")
    mu = float(brentq(objective, lo, hi, xtol=1e-12, rtol=1e-13))
    summary = grand_canonical_summary(geometry, spectra, beta, mu)
    mismatch = abs(float(summary["achieved_N"]) - target_n)
    if mismatch > tolerance:
        raise RuntimeError(
            f"ED mu tuning missed target: N={summary['achieved_N']} target={target_n}"
        )
    return mu, summary, (lo, hi)


def _one_spin_noninteracting_configurations(
    geometry: SquareGeometry, nparticles: int, beta: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    eps, orbitals = eigh(geometry.hopping, check_finite=False)
    configs = list(combinations(range(geometry.nsites), nparticles))
    energies = np.asarray([sum(eps[list(config)]) for config in configs], dtype=float)
    logw = -beta * energies
    weights = np.exp(logw - logsumexp(logw))
    site_means = np.zeros((len(configs), geometry.nsites), dtype=float)
    nn_same = np.zeros(len(configs), dtype=float)
    nnn_same = np.zeros(len(configs), dtype=float)
    for idx, config in enumerate(configs):
        occupied = orbitals[:, list(config)]
        rho = occupied @ occupied.T.conj()
        nsite = np.real(np.diag(rho))
        site_means[idx] = nsite

        def same_spin(bonds: tuple[tuple[int, int], ...]) -> float:
            if not bonds:
                return math.nan
            return float(
                np.mean(
                    [
                        np.real(rho[i, i] * rho[j, j] - rho[i, j] * rho[j, i])
                        for i, j in bonds
                    ]
                )
            )

        nn_same[idx] = same_spin(geometry.nn_bonds)
        nnn_same[idx] = same_spin(geometry.nnn_bonds)
    return energies, weights, site_means, nn_same, nnn_same


def noninteracting_canonical_summary(
    geometry: SquareGeometry, nup: int, ndn: int, beta: float
) -> dict[str, float]:
    eup, wup, nup_configs, nn_up, nnn_up = _one_spin_noninteracting_configurations(
        geometry, nup, beta
    )
    edn, wdn, ndn_configs, nn_dn, nnn_dn = _one_spin_noninteracting_configurations(
        geometry, ndn, beta
    )
    mean_up = wup @ nup_configs
    mean_dn = wdn @ ndn_configs
    kinetic = (float(np.dot(wup, eup)) + float(np.dot(wdn, edn))) / geometry.nsites
    docc = float(np.sum(mean_up * mean_dn) / geometry.nsites)

    def cross(bonds: tuple[tuple[int, int], ...]) -> tuple[float, float]:
        updn = np.mean([mean_up[i] * mean_dn[j] for i, j in bonds])
        dnup = np.mean([mean_dn[i] * mean_up[j] for i, j in bonds])
        return float(updn), float(dnup)

    nn_updn, nn_dnup = cross(geometry.nn_bonds)
    nnn_updn, nnn_dnup = cross(geometry.nnn_bonds)
    nn_same_mean = float(np.dot(wup, nn_up) + np.dot(wdn, nn_dn))
    nnn_same_mean = float(np.dot(wup, nnn_up) + np.dot(wdn, nnn_dn))
    nn_charge = nn_same_mean + nn_updn + nn_dnup
    nnn_charge = nnn_same_mean + nnn_updn + nnn_dnup
    site_density = mean_up + mean_dn
    return {
        "kinetic_per_site": kinetic,
        "double_occupancy_per_site": docc,
        "nn_spin_s_s": nn_same_mean - nn_updn - nn_dnup,
        "nn_charge_raw": nn_charge,
        "nn_connected_charge": _connected_charge(
            nn_charge, site_density, geometry.nn_bonds
        ),
        "nnn_spin_s_s": nnn_same_mean - nnn_updn - nnn_dnup,
        "nnn_charge_raw": nnn_charge,
        "nnn_connected_charge": _connected_charge(
            nnn_charge, site_density, geometry.nnn_bonds
        ),
    }


def noninteracting_grand_canonical_summary(
    geometry: SquareGeometry,
    beta: float,
    chemical_potential: float,
) -> dict[str, float]:
    """Exact U=0 GCE observables from the one-body Fermi matrix."""
    eps, orbitals = eigh(geometry.hopping, check_finite=False)
    occupations = 1.0 / (1.0 + np.exp(beta * (eps - chemical_potential)))
    rho = (orbitals * occupations[None, :]) @ orbitals.T.conj()
    density_one_spin = np.real(np.diag(rho))
    site_density = 2.0 * density_one_spin

    def pair_values(
        bonds: tuple[tuple[int, int], ...],
    ) -> tuple[float, float, float]:
        if not bonds:
            return math.nan, math.nan, math.nan
        spin: list[float] = []
        charge_raw: list[float] = []
        charge_connected: list[float] = []
        for i, j in bonds:
            exchange = abs(rho[i, j]) ** 2
            ni, nj = density_one_spin[i], density_one_spin[j]
            spin.append(-2.0 * exchange)
            charge_raw.append(4.0 * ni * nj - 2.0 * exchange)
            charge_connected.append(-2.0 * exchange)
        return (
            float(np.mean(spin)),
            float(np.mean(charge_raw)),
            float(np.mean(charge_connected)),
        )

    nn_spin, nn_raw, nn_connected = pair_values(geometry.nn_bonds)
    nnn_spin, nnn_raw, nnn_connected = pair_values(geometry.nnn_bonds)
    density = float(np.mean(site_density))
    double_occupancy = float(np.mean(density_one_spin**2))
    kinetic_matrix = float(
        2.0 * np.real(np.trace(geometry.hopping @ rho)) / geometry.nsites
    )
    kinetic_spectrum = float(2.0 * np.dot(occupations, eps) / geometry.nsites)
    n_variance = float(2.0 * np.sum(occupations * (1.0 - occupations)))
    return {
        "kinetic_per_site": kinetic_matrix,
        "kinetic_spectrum_per_site": kinetic_spectrum,
        "double_occupancy_per_site": double_occupancy,
        "nn_spin_s_s": nn_spin,
        "nn_charge_raw": nn_raw,
        "nn_connected_charge": nn_connected,
        "nnn_spin_s_s": nnn_spin,
        "nnn_charge_raw": nnn_raw,
        "nnn_connected_charge": nnn_connected,
        "density": density,
        "achieved_N": float(np.sum(site_density)),
        "N_variance": n_variance,
        "compressibility": beta * n_variance / geometry.nsites,
        "interaction_per_site": 0.0,
        "total_energy_per_site": kinetic_matrix,
        "local_moment": density - 2.0 * double_occupancy,
        "tail_weight_bound": 0.0,
    }


def tune_noninteracting_chemical_potential(
    geometry: SquareGeometry,
    beta: float,
    target_n: float,
) -> tuple[float, dict[str, float], tuple[float, float]]:
    def objective(mu: float) -> float:
        return (
            noninteracting_grand_canonical_summary(geometry, beta, mu)[
                "achieved_N"
            ]
            - target_n
        )

    radius = 1.0
    lo, hi = -radius, radius
    flo, fhi = objective(lo), objective(hi)
    while flo * fhi > 0 and radius < 128:
        radius *= 2.0
        lo, hi = -radius, radius
        flo, fhi = objective(lo), objective(hi)
    if flo * fhi > 0:
        raise RuntimeError(f"failed to bracket U=0 target N={target_n}")
    mu = float(brentq(objective, lo, hi, xtol=1e-13, rtol=1e-14))
    return mu, noninteracting_grand_canonical_summary(geometry, beta, mu), (lo, hi)


def additive_noninteracting_spectrum(
    geometry: SquareGeometry, nup: int, ndn: int
) -> NDArray[np.float64]:
    eps = np.linalg.eigvalsh(geometry.hopping)
    eup = np.asarray([sum(eps[list(c)]) for c in combinations(range(geometry.nsites), nup)])
    edn = np.asarray([sum(eps[list(c)]) for c in combinations(range(geometry.nsites), ndn)])
    return np.sort((eup[:, None] + edn[None, :]).ravel())


def _cache_path(
    cache_dir: Path, geometry: SquareGeometry, interaction: float, nup: int, ndn: int
) -> Path:
    utag = ("p" if interaction >= 0 else "m") + f"{abs(interaction):g}".replace(".", "p")
    return cache_dir / f"v{CACHE_VERSION}_L{geometry.lx}x{geometry.ly}_U{utag}_Nu{nup}_Nd{ndn}.npz"


def save_spectrum(path: Path, geometry: SquareGeometry, spectrum: SectorSpectrum) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "cache_version": CACHE_VERSION,
        "geometry_digest": geometry.digest,
        "nup": spectrum.nup,
        "ndn": spectrum.ndn,
        "interaction": spectrum.interaction,
        "dimension": spectrum.dimension,
        "complete": spectrum.complete,
        "first_omitted_energy_lower": spectrum.first_omitted_energy_lower,
        "omitted_counts": spectrum.omitted_counts.tolist(),
        "omitted_energy_lowers": spectrum.omitted_energy_lowers.tolist(),
        "max_residual": spectrum.max_residual,
        "method": spectrum.method,
    }
    arrays: dict[str, NDArray[np.float64] | str] = {
        "metadata": json.dumps(metadata, sort_keys=True),
        "eigenvalues": spectrum.eigenvalues,
        "site_density": spectrum.site_density,
    }
    arrays.update({f"obs_{k}": v for k, v in spectrum.observables.items()})
    np.savez_compressed(path, **arrays)


def load_spectrum(path: Path, geometry: SquareGeometry) -> SectorSpectrum:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        if metadata["cache_version"] != CACHE_VERSION:
            raise ValueError("ED cache version mismatch")
        if metadata["geometry_digest"] != geometry.digest:
            raise ValueError("ED cache geometry mismatch")
        observables = {
            key.removeprefix("obs_"): np.asarray(data[key], dtype=float)
            for key in data.files
            if key.startswith("obs_")
        }
        return SectorSpectrum(
            nup=int(metadata["nup"]),
            ndn=int(metadata["ndn"]),
            interaction=float(metadata["interaction"]),
            dimension=int(metadata["dimension"]),
            eigenvalues=np.asarray(data["eigenvalues"], dtype=float),
            observables=observables,
            site_density=np.asarray(data["site_density"], dtype=float),
            complete=bool(metadata["complete"]),
            first_omitted_energy_lower=float(metadata["first_omitted_energy_lower"]),
            omitted_counts=np.asarray(metadata["omitted_counts"], dtype=np.int64),
            omitted_energy_lowers=np.asarray(
                metadata["omitted_energy_lowers"], dtype=float
            ),
            max_residual=float(metadata["max_residual"]),
            method=str(metadata["method"]),
        )


def get_spectrum(
    geometry: SquareGeometry,
    nup: int,
    ndn: int,
    interaction: float,
    beta: float,
    *,
    cache_dir: Path | None,
    boltzmann_tolerance: float,
    dense_threshold: int,
    initial_k: int,
    max_k: int,
) -> SectorSpectrum:
    # Spin interchange and the bipartite particle-hole map are exact for this
    # t'=0 OBC Hamiltonian.  Store only N<=V and Nup<=Ndn representatives.
    # This materially reduces the all-sector 3x3 GCE trace while preserving
    # eigenvalue multiplicities and every real-space estimator.
    use_particle_hole = (nup + ndn) > geometry.nsites
    if use_particle_hole:
        source_nup = geometry.nsites - nup
        source_ndn = geometry.nsites - ndn
    else:
        source_nup, source_ndn = nup, ndn
    canonical_nup, canonical_ndn = sorted((source_nup, source_ndn))
    path = (
        _cache_path(cache_dir, geometry, interaction, canonical_nup, canonical_ndn)
        if cache_dir is not None
        else None
    )
    spectrum: SectorSpectrum | None = None
    if path is not None and path.exists():
        spectrum = load_spectrum(path, geometry)
        if spectrum_tail_fraction(spectrum, beta) > boltzmann_tolerance:
            spectrum = None
    if spectrum is None:
        spectrum = diagonalize_sector(
            geometry,
            canonical_nup,
            canonical_ndn,
            interaction,
            beta,
            boltzmann_tolerance=boltzmann_tolerance,
            dense_threshold=dense_threshold,
            initial_k=initial_k,
            max_k=max_k,
        )
        if path is not None:
            save_spectrum(path, geometry, spectrum)
    source = spectrum
    if (source_nup, source_ndn) != (canonical_nup, canonical_ndn):
        source = SectorSpectrum(
            nup=source_nup,
            ndn=source_ndn,
            interaction=spectrum.interaction,
            dimension=spectrum.dimension,
            eigenvalues=spectrum.eigenvalues,
            observables=spectrum.observables,
            site_density=spectrum.site_density,
            complete=spectrum.complete,
            first_omitted_energy_lower=spectrum.first_omitted_energy_lower,
            omitted_counts=spectrum.omitted_counts,
            omitted_energy_lowers=spectrum.omitted_energy_lowers,
            max_residual=spectrum.max_residual,
            method=spectrum.method + "-spin-swap",
        )
    if not use_particle_hole:
        return source

    nsites = geometry.nsites
    ntotal_source = source.nup + source.ndn
    energy_shift = interaction * (nsites - ntotal_source)
    site_density = 2.0 - source.site_density

    def transformed_charge(
        raw: NDArray[np.float64], bonds: tuple[tuple[int, int], ...]
    ) -> NDArray[np.float64]:
        endpoint = np.mean(
            np.column_stack(
                [source.site_density[:, i] + source.site_density[:, j] for i, j in bonds]
            ),
            axis=1,
        )
        return 4.0 - 2.0 * endpoint + raw

    observables = {
        "kinetic_per_site": source.observables["kinetic_per_site"],
        "double_occupancy_per_site": (
            1.0
            - ntotal_source / nsites
            + source.observables["double_occupancy_per_site"]
        ),
        "nn_spin_s_s": source.observables["nn_spin_s_s"],
        "nn_charge_raw": transformed_charge(
            source.observables["nn_charge_raw"], geometry.nn_bonds
        ),
        "nnn_spin_s_s": source.observables["nnn_spin_s_s"],
        "nnn_charge_raw": transformed_charge(
            source.observables["nnn_charge_raw"], geometry.nnn_bonds
        ),
    }
    return SectorSpectrum(
        nup=nup,
        ndn=ndn,
        interaction=spectrum.interaction,
        dimension=spectrum.dimension,
        eigenvalues=source.eigenvalues + energy_shift,
        observables=observables,
        site_density=site_density,
        complete=spectrum.complete,
        first_omitted_energy_lower=source.first_omitted_energy_lower + energy_shift,
        omitted_counts=spectrum.omitted_counts,
        omitted_energy_lowers=source.omitted_energy_lowers + energy_shift,
        max_residual=spectrum.max_residual,
        method=source.method + "-particle-hole",
    )


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            delimiter="\t",
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--sizes", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--interactions", type=float, nargs="+", default=[-3.0, 3.0])
    parser.add_argument("--boltzmann-tolerance", type=float, default=1e-8)
    parser.add_argument("--eigen-residual-tolerance", type=float, default=1e-8)
    parser.add_argument("--dense-threshold", type=int, default=1800)
    parser.add_argument("--initial-k", type=int, default=64)
    parser.add_argument("--max-k", type=int, default=1024)
    parser.add_argument("--skip-gce", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def run_validation_suite(args: argparse.Namespace) -> dict[str, object]:
    args.outdir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or args.outdir / "spectrum_cache"
    geometry_rows: list[dict[str, object]] = []
    u0_rows: list[dict[str, object]] = []
    ce_rows: list[dict[str, object]] = []
    gce_rows: list[dict[str, object]] = []
    sector_rows: list[dict[str, object]] = []
    failures: list[str] = []

    matrix = {
        2: {"sectors": [(1, 1), (2, 2)], "betas": [2.0, 5.0]},
        3: {"sectors": [(2, 2), (4, 4)], "betas": [5.0]},
    }

    for size in args.sizes:
        if size not in matrix:
            raise ValueError(f"validation matrix is defined only for L=2,3; got {size}")
        geometry = build_square_geometry(size, size)
        geometry_rows.append(
            {
                "Lx": size,
                "Ly": size,
                "boundary": "open",
                "site_count": geometry.nsites,
                "nn_bond_count": len(geometry.nn_bonds),
                "nnn_bond_count": len(geometry.nnn_bonds),
                "hermitian": bool(np.array_equal(geometry.hopping, geometry.hopping.T)),
                "no_wrap": True,
                "site_ordering": "x-fastest: site=x+y*Lx (Python zero-based)",
            }
        )

        # U=0 spectrum and thermal-observable oracle.  At U=0, direct
        # diagonalization of the one-body OBC hopping matrix plus exact
        # enumeration of occupied-orbital combinations is the complete
        # many-body diagonalization; do not approximate this check with a
        # truncated Krylov trace.
        u0_sector = matrix[size]["sectors"][-1]
        u0_beta = float(matrix[size]["betas"][-1])
        additive = additive_noninteracting_spectrum(geometry, *u0_sector)
        u0_direct = noninteracting_canonical_summary(geometry, *u0_sector, u0_beta)
        h10, _, _, _ = build_sector_hamiltonian(geometry, 1, 0, 0.0)
        matrix_error = float(np.max(np.abs(h10.toarray() - geometry.hopping)))
        one_body_spectrum_error = float(
            np.max(
                np.abs(
                    np.linalg.eigvalsh(h10.toarray())
                    - np.linalg.eigvalsh(geometry.hopping)
                )
            )
        )
        weights = np.exp(-u0_beta * additive - logsumexp(-u0_beta * additive))
        additive_kinetic = float(np.dot(weights, additive) / geometry.nsites)
        observable_error = abs(
            additive_kinetic - float(u0_direct["kinetic_per_site"])
        )
        u0_pass = (
            matrix_error < 1e-14
            and one_body_spectrum_error < 1e-13
            and observable_error < 1e-12
            and all(np.isfinite(float(v)) for v in u0_direct.values())
        )
        u0_rows.append(
            {
                "ensemble": "CE",
                "L": size,
                "nup": u0_sector[0],
                "ndn": u0_sector[1],
                "beta": u0_beta,
                "many_body_eigenvalues_enumerated": len(additive),
                "one_body_hamiltonian_abs_error": matrix_error,
                "one_body_spectrum_abs_error": one_body_spectrum_error,
                "thermal_kinetic_abs_error": observable_error,
                "tail_weight_bound": 0.0,
                "method": "complete one-body diagonalization and exact fixed-N Slater trace",
                "passed": u0_pass,
                "target_N": sum(u0_sector),
                "achieved_N": float(sum(u0_sector)),
                "mu_ph_symmetric": "not_applicable",
                "mu_bracket_low": "not_applicable",
                "mu_bracket_high": "not_applicable",
                "density_abs_error": 0.0,
            }
        )
        if not u0_pass:
            failures.append(f"L={size} U=0 CE deterministic check")

        u0_target = sum(u0_sector)
        u0_mu, u0_gce, u0_bracket = tune_noninteracting_chemical_potential(
            geometry, u0_beta, u0_target
        )
        u0_gce_kinetic_error = abs(
            float(u0_gce["kinetic_per_site"])
            - float(u0_gce["kinetic_spectrum_per_site"])
        )
        u0_gce_density_error = abs(float(u0_gce["achieved_N"]) - u0_target)
        u0_gce_pass = (
            u0_gce_kinetic_error < 1e-13 and u0_gce_density_error < 1e-10
        )
        u0_rows.append(
            {
                "ensemble": "GCE",
                "L": size,
                "beta": u0_beta,
                "target_N": u0_target,
                "achieved_N": u0_gce["achieved_N"],
                "mu_ph_symmetric": u0_mu,
                "mu_bracket_low": u0_bracket[0],
                "mu_bracket_high": u0_bracket[1],
                "one_body_hamiltonian_abs_error": matrix_error,
                "one_body_spectrum_abs_error": one_body_spectrum_error,
                "thermal_kinetic_abs_error": u0_gce_kinetic_error,
                "density_abs_error": u0_gce_density_error,
                "tail_weight_bound": 0.0,
                "method": "complete one-body diagonalization and exact Fermi trace",
                "passed": u0_gce_pass,
            }
        )
        if not u0_gce_pass:
            failures.append(f"L={size} U=0 GCE deterministic check")

        for interaction in args.interactions:
            spectra_needed: dict[tuple[int, int], SectorSpectrum] = {}
            if not args.skip_gce:
                # The GCE trace includes every spin sector.
                for nup in range(geometry.nsites + 1):
                    for ndn in range(geometry.nsites + 1):
                        spectra_needed[(nup, ndn)] = get_spectrum(
                            geometry,
                            nup,
                            ndn,
                            interaction,
                            min(matrix[size]["betas"]),
                            cache_dir=cache_dir,
                            boltzmann_tolerance=args.boltzmann_tolerance / (geometry.nsites + 1) ** 2,
                            dense_threshold=args.dense_threshold,
                            initial_k=args.initial_k,
                            max_k=args.max_k,
                        )

            for beta in matrix[size]["betas"]:
                for nup, ndn in matrix[size]["sectors"]:
                    spectrum = spectra_needed.get((nup, ndn)) or get_spectrum(
                        geometry,
                        nup,
                        ndn,
                        interaction,
                        beta,
                        cache_dir=cache_dir,
                        boltzmann_tolerance=args.boltzmann_tolerance,
                        dense_threshold=args.dense_threshold,
                        initial_k=args.initial_k,
                        max_k=args.max_k,
                    )
                    ce = canonical_summary(geometry, spectrum, beta)
                    accepted = (
                        float(ce["tail_weight_bound"]) <= args.boltzmann_tolerance
                        and float(ce["max_eigen_residual"])
                        <= args.eigen_residual_tolerance
                    )
                    row: dict[str, object] = {
                        "ensemble": "CE",
                        "L": size,
                        "boundary": "open",
                        "U": interaction,
                        "beta": beta,
                        "temperature": 1 / beta,
                        "nup": nup,
                        "ndn": ndn,
                        "target_N": nup + ndn,
                        "eigen_residual_tolerance": args.eigen_residual_tolerance,
                        "accepted": accepted,
                    }
                    row.update(ce)
                    ce_rows.append(row)
                    sector_rows.append(
                        {
                            "L": size,
                            "U": interaction,
                            "nup": nup,
                            "ndn": ndn,
                            "dimension": spectrum.dimension,
                            "states_kept": spectrum.nkept,
                            "method": spectrum.method,
                            "first_omitted_energy_lower": spectrum.first_omitted_energy_lower,
                            "tail_weight_bound_beta": ce["tail_weight_bound"],
                            "beta": beta,
                            "max_eigen_residual": spectrum.max_residual,
                        }
                    )
                    if not accepted:
                        failures.append(
                            f"L={size} CE U={interaction} beta={beta} sector=({nup},{ndn}) "
                            "tail/residual bound"
                        )

                    if not args.skip_gce:
                        mu, gce, bracket = tune_chemical_potential(
                            geometry,
                            spectra_needed,
                            beta,
                            nup + ndn,
                            tolerance=0.01,
                        )
                        gce_max_residual = max(
                            spectrum.max_residual for spectrum in spectra_needed.values()
                        )
                        gce_accepted = (
                            float(gce["tail_weight_bound"]) <= args.boltzmann_tolerance
                            and abs(float(gce["achieved_N"]) - (nup + ndn)) <= 0.01
                            and gce_max_residual <= args.eigen_residual_tolerance
                        )
                        grow: dict[str, object] = {
                            "ensemble": "GCE",
                            "L": size,
                            "boundary": "open",
                            "U": interaction,
                            "beta": beta,
                            "temperature": 1 / beta,
                            "target_N": nup + ndn,
                            "mu_ph_symmetric": mu,
                            "mu_bracket_low": bracket[0],
                            "mu_bracket_high": bracket[1],
                            "density_tolerance": 0.01,
                            "max_eigen_residual": gce_max_residual,
                            "eigen_residual_tolerance": args.eigen_residual_tolerance,
                            "accepted": gce_accepted,
                        }
                        grow.update(gce)
                        gce_rows.append(grow)
                        if not gce_accepted:
                            failures.append(
                                f"L={size} GCE U={interaction} beta={beta} targetN={nup+ndn}"
                            )

    write_tsv(args.outdir / "geometry_checks.tsv", geometry_rows)
    write_tsv(args.outdir / "u0_checks.tsv", u0_rows)
    write_tsv(args.outdir / "ce_ed_reference.tsv", ce_rows)
    write_tsv(args.outdir / "gce_ed_reference.tsv", gce_rows)
    write_tsv(args.outdir / "sector_boltzmann_bounds.tsv", sector_rows)
    summary = {
        "schema_version": 2,
        "boundary": "open",
        "hamiltonian": "H_std=K+U*D; GCE uses U*(V/4-N/2)-mu*N sector shift",
        "boltzmann_tolerance": args.boltzmann_tolerance,
        "eigen_residual_tolerance": args.eigen_residual_tolerance,
        "geometry_checks": len(geometry_rows),
        "u0_checks": len(u0_rows),
        "ce_conditions": len(ce_rows),
        "gce_conditions": len(gce_rows),
        "failures": failures,
        "accepted": not failures,
    }
    (args.outdir / "validation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    if failures and not args.allow_incomplete:
        raise RuntimeError("ED validation gates failed: " + "; ".join(failures))
    return summary


def main() -> None:
    args = parse_args()
    summary = run_validation_suite(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
