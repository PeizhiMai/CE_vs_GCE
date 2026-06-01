#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare CE-QMC and ED Matsubara Green data.")
    p.add_argument("--qmc-path", type=Path, required=True)
    p.add_argument("--ed-path", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    qmc = pd.read_csv(args.qmc_path, sep="\t")
    ed = pd.read_csv(args.ed_path, sep="\t")
    df = qmc.merge(ed, on=["n", "omega_n"], how="inner")
    df["delta_real"] = df["Greal_mean"] - df["Greal_ed"]
    df["delta_imag"] = df["Gimag_mean"] - df["Gimag_ed"]
    df.to_csv(args.outdir / "greens_iwn_comparison.tsv", sep="\t", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(6.0, 7.0), sharex=True)
    axes[0].errorbar(df["omega_n"], df["Greal_mean"], yerr=df["Greal_stderr"], fmt="o", label="CE QMC")
    axes[0].plot(df["omega_n"], df["Greal_ed"], "-s", label="ED")
    axes[0].set_ylabel("Re G(iωₙ)")
    axes[0].legend(frameon=False)

    axes[1].errorbar(df["omega_n"], df["Gimag_mean"], yerr=df["Gimag_stderr"], fmt="o", label="CE QMC")
    axes[1].plot(df["omega_n"], df["Gimag_ed"], "-s", label="ED")
    axes[1].set_ylabel("Im G(iωₙ)")
    axes[1].set_xlabel("ωₙ")

    fig.tight_layout()
    fig.savefig(args.outdir / "greens_iwn_compare.png", dpi=200, bbox_inches="tight")
    fig.savefig(args.outdir / "greens_iwn_compare.svg", bbox_inches="tight")


if __name__ == "__main__":
    main()
