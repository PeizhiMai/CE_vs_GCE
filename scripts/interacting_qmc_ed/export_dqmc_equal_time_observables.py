#!/usr/bin/env python3
"""Export SmoQyDQMC grand-canonical equal-time outputs in CE-style TSV files.

The CE equal-time workflow writes four compact files:

  * equal_time_observables_qmc.tsv
  * equal_time_charge_spin_wedge_qmc.tsv
  * equal_time_structure_factors_qmc.tsv
  * equal_time_neighbor_shells_qmc.tsv

This script converts a completed SmoQyDQMC DQMC data folder (or its parent
containing exactly one data folder) into the same set of files, restricted to
the equal-time checklist used for the CE/GCE comparison: local scalars,
charge/spin real-space correlations, irreducible D4-q structure factors, and
the first four neighbor shells.  It intentionally does not export pairing,
BKT/current, or unequal-time Green's functions.
"""

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--datafolder", type=Path, required=True,
                   help="SmoQyDQMC data folder, or parent containing exactly one data folder.")
    p.add_argument("--outdir", type=Path, default=None,
                   help="Directory for CE-style exports. Default: datafolder.")
    p.add_argument("--u", type=float, required=True)
    p.add_argument("--beta", type=float, required=True)
    p.add_argument("--mu", type=float, required=True)
    p.add_argument("--dtau", type=float, required=True)
    p.add_argument("--lx", type=int, required=True)
    p.add_argument("--ly", type=int, default=None)
    p.add_argument("--nsamples", type=int, default=None)
    p.add_argument("--batches", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def is_smoqy_datafolder(path: Path) -> bool:
    return (path / "global_stats_pID-0.csv").is_file() or (path / "global_stats.csv").is_file()


def resolve_datafolder(path: Path) -> Path:
    if is_smoqy_datafolder(path):
        return path
    children = [p for p in path.iterdir() if p.is_dir() and is_smoqy_datafolder(p)]
    if len(children) != 1:
        raise SystemExit(f"Expected a SmoQy data folder or one child data folder under {path}, got {children}")
    return children[0]


def first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.is_file():
            return path
    raise FileNotFoundError("None of these files exist: " + ", ".join(str(p) for p in paths))


def read_space_table(path: Path) -> List[Dict[str, str]]:
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    if not lines:
        return []
    header = lines[0].split()
    rows = []
    for ln in lines[1:]:
        vals = ln.split()
        if len(vals) != len(header):
            raise ValueError(f"Bad row length in {path}: {ln}")
        rows.append(dict(zip(header, vals)))
    return rows


def f(row: Dict[str, str], key: str) -> float:
    return float(row[key])


def i(row: Dict[str, str], key: str) -> int:
    return int(float(row[key]))


def global_map(rows: List[Dict[str, str]]) -> Dict[str, Tuple[float, float]]:
    return {r["MEASUREMENT"]: (f(r, "MEAN_REAL"), f(r, "STD")) for r in rows}


def local_select(rows: List[Dict[str, str]], measurement: str, ids: Iterable[int]) -> List[Dict[str, str]]:
    want = set(ids)
    return [r for r in rows if r["MEASUREMENT"] == measurement and r["ID_TYPE"] == "HOPPING_ID" and i(r, "ID") in want]


def quadrature(vals: Iterable[float]) -> float:
    return math.sqrt(sum(float(v) ** 2 for v in vals))


def d4_rep(mx: int, my: int, lx: int, ly: int) -> Tuple[int, int]:
    cand = []
    for a, b in [(mx, my), (my, mx), (-mx, my), (mx, -my), (-my, mx), (my, -mx), (-mx, -my), (-my, -mx)]:
        aa = a % lx
        bb = b % ly
        aa = min(aa, (-aa) % lx)
        bb = min(bb, (-bb) % ly)
        if bb > aa:
            aa, bb = bb, aa
        cand.append((aa, bb))
    return min(cand)


def q_reps(lx: int, ly: int) -> List[Tuple[int, int]]:
    reps = []
    seen = set()
    for mx in range(lx):
        for my in range(ly):
            if lx == ly:
                r = d4_rep(mx, my, lx, ly)
            else:
                r = (min(mx, (-mx) % lx), min(my, (-my) % ly))
            if r not in seen:
                seen.add(r)
                reps.append(r)
    return sorted(reps)


def wrapped_shell_vectors(shell: int, lx: int, ly: int) -> List[Tuple[int, int]]:
    if shell == 1:
        vecs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    elif shell == 2:
        vecs = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    elif shell == 3:
        vecs = [(2, 0), (-2, 0), (0, 2), (0, -2)]
    elif shell == 4:
        vecs = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]
    else:
        raise ValueError(shell)
    return sorted(set((dx % lx, dy % ly) for dx, dy in vecs))


def vector_string(vecs: Iterable[Tuple[int, int]]) -> str:
    return ";".join(f"({dx},{dy})" for dx, dy in vecs)


def read_metadata_defaults(datafolder: Path) -> Tuple[Optional[int], Optional[int]]:
    infos = sorted(datafolder.glob("simulation_info_sID-*_pID-0.toml"))
    if not infos:
        return None, None
    txt = infos[0].read_text(errors="ignore")
    def find_int(key: str) -> Optional[int]:
        m = re.search(rf"^{re.escape(key)}\s*=\s*([0-9]+)", txt, re.M)
        return int(m.group(1)) if m else None
    return find_int("N_measurements"), find_int("N_bins")


def row_by_xy(rows: List[Dict[str, str]], xkey: str, ykey: str) -> Dict[Tuple[int, int], Dict[str, str]]:
    return {(i(r, xkey), i(r, ykey)): r for r in rows}


def write_tsv(path: Path, header: List[str], rows: List[Dict[str, object]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise SystemExit(f"{path} exists; pass --overwrite to replace it")
    with path.open("w", newline="") as fp:
        wr = csv.DictWriter(fp, delimiter="\t", fieldnames=header)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in header})


def main() -> None:
    args = parse_args()
    ly = args.ly if args.ly is not None else args.lx
    nsite = args.lx * ly
    datafolder = resolve_datafolder(args.datafolder)
    outdir = args.outdir or datafolder
    outdir.mkdir(parents=True, exist_ok=True)

    ns_meta, nb_meta = read_metadata_defaults(datafolder)
    nsamples = args.nsamples if args.nsamples is not None else (ns_meta or 0)
    batches = args.batches if args.batches is not None else (nb_meta or 0)

    g = global_map(read_space_table(first_existing(datafolder / "global_stats_pID-0.csv", datafolder / "global_stats.csv")))
    loc = read_space_table(first_existing(datafolder / "local_stats_pID-0.csv", datafolder / "local_stats.csv"))
    density, density_err = g["density"]
    docc, docc_err = g["double_occ"]
    average_sign, average_sign_err = g.get("sgn", (float("nan"), float("nan")))
    compress, compress_err = g.get("compressibility", (0.0, 0.0))
    n_total = density * nsite

    hop = local_select(loc, "hopping_energy", [1, 2])
    hop_x = local_select(loc, "hopping_energy", [1])
    kinetic = sum(f(r, "MEAN_REAL") for r in hop)
    kinetic_err = quadrature(f(r, "STD") for r in hop)
    kx = sum(f(r, "MEAN_REAL") for r in hop_x)
    kx_err = quadrature(f(r, "STD") for r in hop_x)
    interaction = args.u * docc
    interaction_err = abs(args.u) * docc_err
    total = kinetic + interaction
    total_err = quadrature([kinetic_err, interaction_err])
    local_moment = density - 2.0 * docc
    local_moment_err = quadrature([density_err, 2.0 * docc_err])

    scalar_rows = [{
        "beta": args.beta,
        "temperature": 1.0 / args.beta,
        "mu": args.mu,
        "ntotal": n_total,
        "density": density,
        "density_stderr": density_err,
        "average_sign": average_sign,
        "average_sign_stderr": average_sign_err,
        "kinetic_per_site": kinetic,
        "kinetic_stderr": kinetic_err,
        "interaction_per_site": interaction,
        "interaction_stderr": interaction_err,
        "total_per_site": total,
        "total_stderr": total_err,
        "double_occupancy_per_site": docc,
        "double_occupancy_stderr": docc_err,
        "local_moment_z": local_moment,
        "local_moment_z_stderr": local_moment_err,
        "Kx_per_site": kx,
        "Kx_stderr": kx_err,
        "diamagnetic_minus_Kx_per_site": -kx,
        "diamagnetic_minus_Kx_stderr": kx_err,
        "time_slices": round(args.beta / args.dtau),
        "nsamples": nsamples,
        "batches": batches,
    }]
    write_tsv(
        outdir / "equal_time_observables_qmc.tsv",
        list(scalar_rows[0].keys()),
        scalar_rows,
        args.overwrite,
    )

    dens_pos = row_by_xy(
        read_space_table(first_existing(datafolder / "equal-time" / "density" / "density_position_equal-time_stats_pID-0.csv", datafolder / "equal-time" / "density" / "density_position_equal-time_stats.csv")),
        "R_2", "R_1",
    )
    spin_pos = row_by_xy(
        read_space_table(first_existing(datafolder / "equal-time" / "spin_z" / "spin_z_position_equal-time_stats_pID-0.csv", datafolder / "equal-time" / "spin_z" / "spin_z_position_equal-time_stats.csv")),
        "R_2", "R_1",
    )
    wedge_rows = []
    for dx, dy in sorted(dens_pos):
        dr = dens_pos[(dx, dy)]
        sr = spin_pos[(dx, dy)]
        raw = f(dr, "MEAN_REAL")
        raw_err = f(dr, "STD")
        connected = raw - density * density
        connected_err = quadrature([raw_err, 2.0 * abs(density) * density_err])
        spin_sz = f(sr, "MEAN_REAL")
        spin_sz_err = f(sr, "STD")
        wedge_rows.append({
            "dx": dx, "dy": dy,
            "charge_corr_raw": raw,
            "charge_corr_raw_stderr": raw_err,
            "charge_corr_connected": connected,
            "charge_corr_connected_stderr": connected_err,
            "spin_corr_s_s": 4.0 * spin_sz,
            "spin_corr_s_s_stderr": 4.0 * spin_sz_err,
            "spin_corr_SzSz": spin_sz,
            "spin_corr_SzSz_stderr": spin_sz_err,
            "nsamples": nsamples,
            "batches": batches,
        })
    write_tsv(outdir / "equal_time_charge_spin_wedge_qmc.tsv", list(wedge_rows[0].keys()), wedge_rows, args.overwrite)

    dens_mom = row_by_xy(
        read_space_table(first_existing(datafolder / "equal-time" / "density" / "density_momentum_equal-time_stats_pID-0.csv", datafolder / "equal-time" / "density" / "density_momentum_equal-time_stats.csv")),
        "K_2", "K_1",
    )
    spin_mom = row_by_xy(
        read_space_table(first_existing(datafolder / "equal-time" / "spin_z" / "spin_z_momentum_equal-time_stats_pID-0.csv", datafolder / "equal-time" / "spin_z" / "spin_z_momentum_equal-time_stats.csv")),
        "K_2", "K_1",
    )
    q_rows = []
    for mx, my in q_reps(args.lx, ly):
        dr = dens_mom[(mx, my)]
        sr = spin_mom[(mx, my)]
        raw = f(dr, "MEAN_REAL")
        raw_err = f(dr, "STD")
        if (mx, my) == (0, 0):
            connected = compress / args.beta
            connected_err = compress_err / args.beta
        else:
            connected = raw
            connected_err = raw_err
        spin_sz = f(sr, "MEAN_REAL")
        spin_sz_err = f(sr, "STD")
        q_rows.append({
            "mx": mx,
            "my": my,
            "qx": 2.0 * math.pi * mx / args.lx,
            "qy": 2.0 * math.pi * my / ly,
            "charge_structure_raw": raw,
            "charge_structure_raw_stderr": raw_err,
            "charge_structure_connected": connected,
            "charge_structure_connected_stderr": connected_err,
            "spin_structure_s_s": 4.0 * spin_sz,
            "spin_structure_s_s_stderr": 4.0 * spin_sz_err,
            "spin_structure_SzSz": spin_sz,
            "spin_structure_SzSz_stderr": spin_sz_err,
            "nsamples": nsamples,
            "batches": batches,
        })
    write_tsv(outdir / "equal_time_structure_factors_qmc.tsv", list(q_rows[0].keys()), q_rows, args.overwrite)

    wedge_by_vec = {(int(r["dx"]), int(r["dy"])): r for r in wedge_rows}
    shell_rows = []
    for shell in [1, 2, 3, 4]:
        vecs = wrapped_shell_vectors(shell, args.lx, ly)
        vals = [wedge_by_vec[v] for v in vecs]
        nvec = len(vals)
        shell_rows.append({
            "shell": shell,
            "r2": min(dx * dx + dy * dy for dx, dy in vecs),
            "vectors": vector_string(vecs),
            "charge_corr_raw": sum(float(r["charge_corr_raw"]) for r in vals) / nvec,
            "charge_corr_raw_stderr": quadrature(float(r["charge_corr_raw_stderr"]) for r in vals) / nvec,
            "charge_corr_connected": sum(float(r["charge_corr_connected"]) for r in vals) / nvec,
            "charge_corr_connected_stderr": quadrature(float(r["charge_corr_connected_stderr"]) for r in vals) / nvec,
            "spin_corr_s_s": sum(float(r["spin_corr_s_s"]) for r in vals) / nvec,
            "spin_corr_s_s_stderr": quadrature(float(r["spin_corr_s_s_stderr"]) for r in vals) / nvec,
            "spin_corr_SzSz": sum(float(r["spin_corr_SzSz"]) for r in vals) / nvec,
            "spin_corr_SzSz_stderr": quadrature(float(r["spin_corr_SzSz_stderr"]) for r in vals) / nvec,
            "nsamples": nsamples,
            "batches": batches,
        })
    write_tsv(outdir / "equal_time_neighbor_shells_qmc.tsv", list(shell_rows[0].keys()), shell_rows, args.overwrite)
    print(f"Wrote DQMC equal-time exports to {outdir}")


if __name__ == "__main__":
    main()
