#!/usr/bin/env python3
"""Compute finite-size superfluid density from SmoQyDQMC current outputs.

Uses the integrated x-current/current correlation measured for HOPPING_ID 1
(+x nearest-neighbor hopping in the local attractive-Hubbard drivers):

    rho_s = 1/4 * [Lambda_xx(qmin, 0) - Lambda_xx(0, qmin)]

where SmoQyDQMC's integrated momentum current correlation is already
per-site/per-unit-cell and imaginary-time integrated.

Also reports the diamagnetic estimator:

    rho_s_diamagnetic = 1/4 * [(-K_x/N) - Lambda_xx(0, qmin)]

with K_x/N read from local_stats hopping_energy HOPPING_ID 1.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def read_rows(path: Path):
    with path.open() as f:
        reader = csv.DictReader(f, delimiter=" ", skipinitialspace=True)
        return list(reader)


def as_float(row, key):
    return float(row[key])


def find_current_file(datafolder: Path) -> Path:
    candidates = sorted((datafolder / "integrated" / "current").glob("current_momentum_integrated_stats*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No integrated current momentum CSV found under {datafolder / 'integrated' / 'current'}; "
            "make sure the run initialized correlation=\"current\", integrated=true, and process_measurements(export_to_csv=true) completed."
        )
    return candidates[0]


def find_stats_file(datafolder: Path) -> Path | None:
    candidates = sorted(datafolder.glob("stats_pID-*.h5"))
    return candidates[0] if candidates else None


def find_local_file(datafolder: Path) -> Path | None:
    candidates = sorted(datafolder.glob("local_stats*.csv"))
    return candidates[0] if candidates else None


def select_current(rows, k1: int, k2: int, hopping_id: int):
    # SmoQy exports ID pair columns as HOPPING_ID_2, HOPPING_ID_1 and momentum columns as K_2, K_1.
    for row in rows:
        if int(row.get("HOPPING_ID_2", row.get("HOPPING_ID", hopping_id))) != hopping_id:
            continue
        if int(row.get("HOPPING_ID_1", row.get("HOPPING_ID", hopping_id))) != hopping_id:
            continue
        if int(row["K_1"]) == k1 and int(row["K_2"]) == k2:
            return as_float(row, "MEAN_REAL"), as_float(row, "STD")
    raise KeyError(f"Could not find current row with HOPPING_ID pair ({hopping_id},{hopping_id}) and K_1={k1}, K_2={k2}")


def select_local_hopping(local_file: Path | None, hopping_id: int):
    if local_file is None:
        return math.nan, math.nan
    rows = read_rows(local_file)
    for row in rows:
        if row.get("MEASUREMENT") == "hopping_energy" and int(row.get("ID", "-1")) == hopping_id:
            return as_float(row, "MEAN_REAL"), as_float(row, "STD")
    return math.nan, math.nan


def combine_err(*errs: float) -> float:
    vals = [e for e in errs if not math.isnan(e)]
    if not vals:
        return math.nan
    return math.sqrt(sum(e * e for e in vals))


def read_from_hdf5(datafolder: Path, hopping_id: int):
    """Read unrounded statistics directly from SmoQyDQMC stats_pID-*.h5.

    CSV exports are controlled by the driver's `decimals` option; for nearly
    saturated cases, e.g. 3x3 conventional attractive Hubbard at beta=10, the
    current response can be ~1e-8 and gets rounded to zero in CSV.
    """
    stats_file = find_stats_file(datafolder)
    if stats_file is None:
        return None
    try:
        import h5py  # type: ignore
    except ImportError:
        return None

    with h5py.File(stats_file, "r") as h5:
        # SmoQy stores current momentum arrays as [HOPPING_ID, K_2, K_1].
        idx = hopping_id - 1
        mean = h5["CORRELATIONS/STANDARD/INTEGRATED/current/MOMENTUM/MEAN"]
        std = h5["CORRELATIONS/STANDARD/INTEGRATED/current/MOMENTUM/STD"]
        lambda_l = float(mean[idx, 0, 1].real)  # K_1=1, K_2=0
        lambda_l_err = float(std[idx, 0, 1])
        lambda_t = float(mean[idx, 1, 0].real)  # K_1=0, K_2=1
        lambda_t_err = float(std[idx, 1, 0])

        hop_mean = h5["LOCAL/hopping_energy/MEAN"]
        hop_std = h5["LOCAL/hopping_energy/STD"]
        hopping_x = float(hop_mean[idx].real)
        hopping_x_err = float(hop_std[idx])

    return stats_file, lambda_l, lambda_l_err, lambda_t, lambda_t_err, hopping_x, hopping_x_err


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("datafolder", type=Path, help="SmoQyDQMC run datafolder containing integrated/current outputs")
    ap.add_argument("--hopping-id", type=int, default=1, help="HOPPING_ID for +x current; default 1 for provided square Hubbard drivers")
    ap.add_argument("--out", type=Path, default=None, help="Output TSV path; default datafolder/superfluid_density.tsv")
    args = ap.parse_args()

    datafolder = args.datafolder.resolve()
    h5_values = read_from_hdf5(datafolder, args.hopping_id)
    if h5_values is not None:
        current_file, lambda_l, lambda_l_err, lambda_t, lambda_t_err, hopping_x, hopping_x_err = h5_values
        local_file = find_local_file(datafolder)
    else:
        current_file = find_current_file(datafolder)
        rows = read_rows(current_file)
        lambda_l, lambda_l_err = select_current(rows, k1=1, k2=0, hopping_id=args.hopping_id)
        lambda_t, lambda_t_err = select_current(rows, k1=0, k2=1, hopping_id=args.hopping_id)
        local_file = find_local_file(datafolder)
        hopping_x, hopping_x_err = select_local_hopping(local_file, args.hopping_id)
    rho = 0.25 * (lambda_l - lambda_t)
    rho_err = 0.25 * combine_err(lambda_l_err, lambda_t_err)

    diamagnetic_x = -hopping_x if not math.isnan(hopping_x) else math.nan
    diamagnetic_x_err = hopping_x_err
    rho_dia = 0.25 * (diamagnetic_x - lambda_t) if not math.isnan(diamagnetic_x) else math.nan
    rho_dia_err = 0.25 * combine_err(diamagnetic_x_err, lambda_t_err)

    out = args.out if args.out is not None else datafolder / "superfluid_density.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write(
            "hopping_id\tlambda_longitudinal_qmin0\tlambda_longitudinal_qmin0_err\t"
            "lambda_transverse_0qmin\tlambda_transverse_0qmin_err\t"
            "rho_s_current\trho_s_current_err\t"
            "diamagnetic_minus_Kx_per_site\tdiamagnetic_minus_Kx_per_site_err\t"
            "rho_s_diamagnetic\trho_s_diamagnetic_err\tcurrent_file\tlocal_file\n"
        )
        f.write(
            f"{args.hopping_id}\t{lambda_l:.12g}\t{lambda_l_err:.12g}\t"
            f"{lambda_t:.12g}\t{lambda_t_err:.12g}\t"
            f"{rho:.12g}\t{rho_err:.12g}\t"
            f"{diamagnetic_x:.12g}\t{diamagnetic_x_err:.12g}\t"
            f"{rho_dia:.12g}\t{rho_dia_err:.12g}\t{current_file}\t{local_file or ''}\n"
        )

    print(f"wrote {out}")
    print(f"Lambda_L(qmin,0) = {lambda_l:.12g} ± {lambda_l_err:.3g}")
    print(f"Lambda_T(0,qmin) = {lambda_t:.12g} ± {lambda_t_err:.3g}")
    print(f"rho_s current     = {rho:.12g} ± {rho_err:.3g}")
    if not math.isnan(rho_dia):
        print(f"-Kx/N             = {diamagnetic_x:.12g} ± {diamagnetic_x_err:.3g}")
        print(f"rho_s diamagnetic = {rho_dia:.12g} ± {rho_dia_err:.3g}")


if __name__ == "__main__":
    main()
