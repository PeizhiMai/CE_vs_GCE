#!/usr/bin/env python3
"""Build the L=6 equal-time comparison, mismatch, and thermometry assets.

Thermometry inverts the piecewise-linear *GCE* calibration at each actual CE
simulation point.  Plotted lines join only consecutive, actual CE points with
a unique inferred temperature.  They are explicitly broken at no-solution,
multiple-solution, flat-calibration, sign-limited, or unfinished conditions.
Every plotted/statistical bias is (T_GCE - T_CE) / T_CE.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


PRIMARY = [
    ("kinetic", "Kinetic energy/site"),
    ("double_occupancy", "Double occupancy/site"),
    ("nn_spin", "Nearest-neighbor spin correlation"),
    ("nn_charge_connected", "Nearest-neighbor connected charge correlation"),
]
NNN = [
    ("nnn_spin", "Next-nearest-neighbor spin correlation"),
    ("nnn_charge_connected", "Next-nearest-neighbor connected charge correlation"),
]
US = [-5.0, -3.0, 0.0, 3.0, 5.0]
NS = [12, 18, 26, 32]
VOLUME = 36.0
DENSITY_COLORS = {12: "#4B78D0", 18: "#D59A16", 26: "#65A35A", 32: "#D15C8A"}
DENSITY_MARKERS = {12: "o", 18: "s", 26: "D", 32: "^"}
QCOLORS = {
    "kinetic": "#3569B8", "double_occupancy": "#D58B15",
    "nn_spin": "#5C9955", "nn_charge_connected": "#C84F79",
}
STATUS_COLORS = {
    "no_solution": "#F4C7C3", "multiple_solutions": "#F6DF9B",
    "flat_calibration": "#D9DDE4", "sign_limited": "#D5C4E8",
    "unfinished": "#E8EBEF", "waiting_manifest": "#E8EBEF", "invalid": "#E2A6A0",
}


def roots_piecewise(x: np.ndarray, y: np.ndarray, target: float, tol: float = 1e-12):
    roots: list[float] = []
    slopes: list[float] = []
    flat_match = False
    for i in range(len(x) - 1):
        x0, x1, y0, y1 = x[i], x[i + 1], y[i], y[i + 1]
        scale = max(1.0, abs(y0), abs(y1), abs(target))
        tt = tol * scale
        if abs(y1 - y0) <= tt:
            if abs(target - y0) <= tt:
                flat_match = True
                roots.extend([float(x0), float(x1)])
                slopes.extend([0.0, 0.0])
            continue
        if min(y0, y1) - tt <= target <= max(y0, y1) + tt:
            root = x0 + (target - y0) * (x1 - x0) / (y1 - y0)
            if x0 - 1e-10 <= root <= x1 + 1e-10:
                roots.append(float(np.clip(root, x0, x1)))
                slopes.append(float((y1 - y0) / (x1 - x0)))
    pairs = sorted(zip(roots, slopes))
    unique_roots: list[float] = []
    unique_slopes: list[float] = []
    for root, slope in pairs:
        if not unique_roots or abs(root - unique_roots[-1]) > 1e-8:
            unique_roots.append(root)
            unique_slopes.append(slope)
        elif abs(slope) > abs(unique_slopes[-1]):
            unique_slopes[-1] = slope
    return unique_roots, unique_slopes, flat_match


def unique_xy(df: pd.DataFrame, quantity: str) -> tuple[np.ndarray, np.ndarray]:
    x = pd.to_numeric(df["T"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(df[quantity], errors="coerce").to_numpy(float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    order = np.argsort(x)
    x, y = x[order], y[order]
    if not len(x):
        return x, y
    ux, inv = np.unique(np.round(x, 12), return_inverse=True)
    uy = np.asarray([np.mean(y[inv == i]) for i in range(len(ux))])
    return ux.astype(float), uy.astype(float)


def interp_error(df: pd.DataFrame, quantity: str, t: float) -> float:
    col = quantity + "_err"
    if col not in df:
        return 0.0
    x = pd.to_numeric(df["T"], errors="coerce").to_numpy(float)
    e = pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy(float)
    good = np.isfinite(x) & np.isfinite(e)
    if not good.any():
        return 0.0
    order = np.argsort(x[good])
    return float(np.interp(t, x[good][order], e[good][order]))


def infer_temperature(tce: float, qce: float, gx: np.ndarray, gy: np.ndarray,
                      ce_error: float, gce: pd.DataFrame, quantity: str) -> dict[str, object]:
    scale = max(1.0, float(np.max(np.abs(gy))))
    flat_curve = float(np.max(gy) - np.min(gy)) <= max(1e-12, 1e-10 * scale)
    if flat_curve:
        center = float(np.mean(gy))
        tol = max(1e-12, 1e-10 * max(1.0, abs(center), abs(qce)))
        if abs(qce - center) <= tol:
            return {"status": "flat_calibration", "reason": "temperature_not_identifiable",
                    "n_roots": -1, "T_GCE": np.nan, "delta_T_over_T_CE": np.nan,
                    "sigma_ratio": np.nan, "roots": "all_sampled_T"}
        direction = "below_gce_range" if qce < center else "above_gce_range"
        return {"status": "no_solution", "reason": direction, "n_roots": 0,
                "T_GCE": np.nan, "delta_T_over_T_CE": np.nan,
                "sigma_ratio": np.nan, "roots": ""}
    roots, slopes, flat_match = roots_piecewise(gx, gy, qce)
    if not roots:
        direction = "below_gce_range" if qce < float(np.min(gy)) else "above_gce_range"
        return {"status": "no_solution", "reason": direction, "n_roots": 0,
                "T_GCE": np.nan, "delta_T_over_T_CE": np.nan,
                "sigma_ratio": np.nan, "roots": ""}
    selected = int(np.argmin(np.abs(np.asarray(roots) - tce)))
    tgce, slope = roots[selected], slopes[selected]
    status = "multiple_solutions" if len(roots) > 1 or flat_match else "unique_solution"
    ratio = (tgce - tce) / tce
    sigma_ratio = np.nan
    if abs(slope) > 1e-14:
        gce_error = interp_error(gce, quantity, tgce)
        sigma_q = math.hypot(ce_error if math.isfinite(ce_error) else 0.0, gce_error)
        sigma_ratio = sigma_q / abs(slope) / abs(tce)
    return {"status": status, "reason": "", "n_roots": len(roots), "T_GCE": tgce,
            "delta_T_over_T_CE": ratio, "sigma_ratio": sigma_ratio,
            "roots": ";".join(f"{x:.12g}" for x in roots)}


def write_table(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def save_figure(fig: plt.Figure, stem: Path, multipage: PdfPages | None = None) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    if multipage is not None:
        multipage.savefig(fig, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def status_lookup(status: pd.DataFrame) -> dict[tuple[float, int, float, str], str]:
    return {
        (round(float(r.U), 10), int(r.Ntot), round(float(r.beta), 10), str(r.ensemble)): str(r.status)
        for r in status.itertuples(index=False)
    }


def expected_points(status: pd.DataFrame, u: float, n: int, ensemble: str) -> pd.DataFrame:
    out = status[(np.isclose(status.U, u)) & (status.Ntot == n) & (status.ensemble == ensemble)].copy()
    return out.sort_values("T")


def comparison_figure(data: pd.DataFrame, status: pd.DataFrame, quantity: str, label: str) -> plt.Figure:
    # Wide canvas is deliberate: the panels fill a 16:9 slide rather than
    # shrinking into a narrow, tall image.
    fig, axes = plt.subplots(5, 4, figsize=(30.0, 13.0), sharex=True)
    for ri, u in enumerate(US):
        for ci, n in enumerate(NS):
            ax = axes[ri, ci]
            for ensemble, color, marker, linestyle in (
                ("CE", "#4B78D0", "o", "-"), ("GCE", "#D9794F", "s", "--"),
            ):
                expected = expected_points(status, u, n, ensemble)
                actual = data[(data.ensemble == ensemble) & np.isclose(data.U, u) & (data.Ntot == n)]
                amap = {round(float(r.beta), 10): r for r in actual.itertuples(index=False)}
                x, y, e = [], [], []
                for er in expected.itertuples(index=False):
                    rec = amap.get(round(float(er.beta), 10))
                    x.append(float(er.T))
                    if rec is None or str(er.status) not in ("strict_final", "exact_final"):
                        y.append(np.nan); e.append(np.nan)
                    else:
                        y.append(float(getattr(rec, quantity))); e.append(float(getattr(rec, quantity + "_err")))
                order = np.argsort(x)
                x, y, e = np.asarray(x)[order], np.asarray(y)[order], np.asarray(e)[order]
                ax.plot(x, y, color=color, marker=marker, ms=3.8, lw=1.35, ls=linestyle, label=ensemble)
                good = np.isfinite(y) & np.isfinite(e) & (e > 0)
                if good.any():
                    ax.errorbar(x[good], y[good], yerr=e[good], fmt="none", ecolor=color, lw=.65, alpha=.7)
            if ri == 0:
                ax.set_title(f"N={n}, n={n/VOLUME:.4f}", fontsize=11)
            if ci == 0:
                ax.set_ylabel(f"U={u:+g}\n{label}", fontsize=9.5)
            if ri == 4:
                ax.set_xlabel("T")
            ax.grid(alpha=.22)
    handles = [Line2D([0], [0], color="#4B78D0", marker="o", label="CE"),
               Line2D([0], [0], color="#D9794F", marker="s", ls="--", label="GCE")]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(.985, .985), frameon=False)
    fig.suptitle(f"L=6 — {label}: CE versus GCE at actual T=1/β", fontsize=18, fontweight="bold", y=.995)
    fig.tight_layout(rect=(0, 0, 1, .975))
    return fig


def mismatch_figure(mismatch: pd.DataFrame, column: str, title: str, ylabel: str) -> plt.Figure:
    fig, axes = plt.subplots(5, 4, figsize=(30.0, 13.0), sharex=True)
    for ri, u in enumerate(US):
        for qi, (quantity, qlabel) in enumerate(PRIMARY):
            ax = axes[ri, qi]
            for n in NS:
                g = mismatch[np.isclose(mismatch.U, u) & (mismatch.Ntot == n) & (mismatch.quantity == quantity)].sort_values("T")
                ax.plot(g["T"], g[column], marker=DENSITY_MARKERS[n], color=DENSITY_COLORS[n], ms=3.4, lw=1.2)
            ax.axhline(0, color="#777", lw=.65, ls="--")
            if ri == 0:
                ax.set_title(qlabel, fontsize=10.5)
            if qi == 0:
                ax.set_ylabel(f"U={u:+g}\n{ylabel}", fontsize=9.5)
            if ri == 4:
                ax.set_xlabel("T")
            ax.grid(alpha=.22)
    handles = [Line2D([0], [0], color=DENSITY_COLORS[n], marker=DENSITY_MARKERS[n], label=f"N={n}, n={n/VOLUME:.4f}") for n in NS]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(.5, .982))
    fig.suptitle(title, fontsize=18, fontweight="bold", y=.998)
    fig.tight_layout(rect=(0, 0, 1, .958))
    return fig


def point_bounds(temps: np.ndarray) -> list[tuple[float, float]]:
    if len(temps) == 1:
        return [(temps[0] - .01, temps[0] + .01)]
    mids = (temps[:-1] + temps[1:]) / 2
    left = np.r_[temps[0] - (mids[0] - temps[0]), mids]
    right = np.r_[mids, temps[-1] + (temps[-1] - mids[-1])]
    return list(zip(left, right))


def thermometer_figure(observed: pd.DataFrame, status: pd.DataFrame, quantity: str, label: str) -> plt.Figure:
    # Match the full slide width and keep the thermometer panels tall enough
    # to read; this mirrors the enlarged L=8/L=12 thermometer treatment.
    fig, axes = plt.subplots(5, 4, figsize=(30.0, 13.0), sharex=True, sharey=False)
    for ri, u in enumerate(US):
        for ci, n in enumerate(NS):
            ax = axes[ri, ci]
            expected = expected_points(status, u, n, "CE").sort_values("T")
            mapped = observed[np.isclose(observed.U, u) & (observed.Ntot == n) & (observed.quantity == quantity)]
            mmap = {round(float(r.beta_CE), 10): r for r in mapped.itertuples(index=False)}
            xs, ys, states = [], [], []
            for er in expected.itertuples(index=False):
                rec = mmap.get(round(float(er.beta), 10))
                state = str(er.status)
                if rec is not None:
                    state = str(rec.status)
                xs.append(float(er.T)); states.append(state)
                ys.append(float(rec.delta_T_over_T_CE) if rec is not None and state == "unique_solution" else np.nan)
            order = np.argsort(xs)
            xs, ys = np.asarray(xs)[order], np.asarray(ys)[order]
            states = np.asarray(states, dtype=object)[order]
            for (left, right), state in zip(point_bounds(xs), states):
                if state in STATUS_COLORS:
                    ax.axvspan(left, right, color=STATUS_COLORS[state], alpha=.58, lw=0)
            # NaNs make matplotlib break the line at every non-unique condition.
            ax.plot(xs, ys, color=QCOLORS[quantity], marker="o", ms=4.3, lw=1.45)
            ax.axhline(0, color="#666", lw=.75, ls="--")
            ymax = ax.get_ylim()[1]
            for x, state in zip(xs, states):
                if state == "no_solution":
                    ax.text(x, .97, "no T", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.2, color="#9E2F25", rotation=90)
                elif state == "sign_limited":
                    ax.text(x, .97, "sign", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.2, color="#633B86", rotation=90)
                elif state in ("multiple_solutions", "flat_calibration"):
                    ax.text(x, .97, "ambig.", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.0, color="#8B6200", rotation=90)
            if ri == 0:
                ax.set_title(f"N={n}, n={n/VOLUME:.4f}", fontsize=11)
            if ci == 0:
                ax.set_ylabel(f"U={u:+g}\n(T_GCE−T_CE)/T_CE", fontsize=9)
            if ri == 4:
                ax.set_xlabel("T_CE (actual 1/β)")
            ax.grid(alpha=.20)
    legend = [
        Line2D([0], [0], color=QCOLORS[quantity], marker="o", label="actual unique-root CE points"),
        Patch(facecolor=STATUS_COLORS["no_solution"], label="no inferred T"),
        Patch(facecolor=STATUS_COLORS["multiple_solutions"], label="ambiguous / flat"),
        Patch(facecolor=STATUS_COLORS["sign_limited"], label="sign-limited"),
    ]
    fig.legend(handles=legend, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(.5, .978))
    fig.suptitle(f"L=6 {label} thermometry — lines join actual unique-temperature points only", fontsize=18, fontweight="bold", y=.998)
    fig.tight_layout(rect=(0, 0, 1, .955))
    return fig


def scorecard_figure(summary: pd.DataFrame) -> plt.Figure:
    fig, (ax, table_ax) = plt.subplots(2, 1, figsize=(16, 9), gridspec_kw={"height_ratios": [2.2, 1.45]})
    x = np.arange(len(PRIMARY)); width = .19
    for i, (col, label, color) in enumerate((
        ("unique_fraction", "Unique inferred T", "#3C6FB6"),
        ("no_solution_fraction", "No inferred T", "#B54436"),
        ("ambiguous_or_flat_fraction", "Ambiguous / flat", "#A06A00"),
        ("sign_limited_fraction", "Sign-limited", "#73518F"),
    )):
        ax.bar(x + (i - 1.5) * width, summary[col], width, label=label, color=color)
    ax.set_xticks(x, [label for _, label in PRIMARY], rotation=12, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of actual CE conditions")
    ax.grid(axis="y", alpha=.2)
    ax.legend(ncol=4, frameon=False, loc="upper center")
    ax.set_title("L=6 thermometry scorecard", fontsize=18, fontweight="bold")
    table_ax.axis("off")
    cells = []
    for row in summary.itertuples(index=False):
        cells.append([
            row.quantity_label,
            f"{100*row.unique_fraction:.1f}%",
            f"{100*row.no_solution_fraction:.1f}%",
            f"{100*row.ambiguous_or_flat_fraction:.1f}%",
            f"{100*row.sign_limited_fraction:.1f}%",
            "—" if not math.isfinite(row.median_abs_deltaT_over_TCE) else f"{100*row.median_abs_deltaT_over_TCE:.1f}%",
        ])
    table = table_ax.table(cellText=cells,
                           colLabels=["Observable", "Unique T", "No T", "Ambig./flat", "Sign-limited", "Median |ΔT|/T_CE"],
                           cellLoc="center", loc="center", colWidths=[.27, .12, .11, .13, .13, .18])
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 1.65)
    fig.text(.5, .02, "All temperature differences use (T_GCE−T_CE)/T_CE; median bias uses unique roots only.", ha="center", fontsize=11)
    fig.tight_layout(rect=(0, .04, 1, 1))
    return fig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--status", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.snapshot, sep="\t", low_memory=False)
    status = pd.read_csv(args.status, sep="\t", low_memory=False)
    for frame in (data, status):
        for col in ("U", "Ntot", "beta", "T"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    required_unfinished = status[~status.status.isin(["strict_final", "exact_final", "sign_limited"])]
    if len(required_unfinished) and not args.allow_partial:
        raise SystemExit(f"refusing final analysis: {len(required_unfinished)} required ensemble rows unfinished")

    # Matched same-beta CE/GCE differences for the three mismatch slides.
    matched_rows: list[dict[str, object]] = []
    for (u, n, beta), group in data.groupby(["U", "Ntot", "beta"]):
        ce = group[group.ensemble == "CE"]
        gce = group[group.ensemble == "GCE"]
        if len(ce) != 1 or len(gce) != 1:
            continue
        ce, gce = ce.iloc[0], gce.iloc[0]
        for quantity, label in PRIMARY:
            delta = float(ce[quantity]) - float(gce[quantity])
            denom = abs(float(gce[quantity]))
            curve = data[(data.ensemble == "GCE") & np.isclose(data.U, u) & (data.Ntot == n)]
            qrange = float(curve[quantity].max() - curve[quantity].min())
            matched_rows.append({
                "L": 6, "U": u, "Ntot": int(n), "density": n / VOLUME,
                "beta": beta, "T": 1.0 / beta, "quantity": quantity,
                "quantity_label": label, "CE": float(ce[quantity]), "GCE": float(gce[quantity]),
                "delta_CE_minus_GCE": delta,
                "relative_CE_minus_GCE_over_abs_GCE": delta / denom if denom > 1e-14 else np.nan,
                "gce_range_normalized_CE_minus_GCE": delta / qrange if qrange > 1e-14 else np.nan,
            })
    mismatch = pd.DataFrame(matched_rows)
    write_table(args.outdir / "data" / "l6_mismatch_vs_T.tsv", mismatch)

    # Invert the GCE calibration at actual expected CE conditions, preserving
    # explicit placeholder statuses for sign-limited and unfinished points.
    observed: list[dict[str, object]] = []
    for u in US:
        for n in NS:
            ce = data[(data.ensemble == "CE") & np.isclose(data.U, u) & (data.Ntot == n)]
            gce = data[(data.ensemble == "GCE") & np.isclose(data.U, u) & (data.Ntot == n)]
            expected = expected_points(status, u, n, "CE")
            ce_map = {round(float(r.beta), 10): r for r in ce.itertuples(index=False)}
            for quantity, label in PRIMARY:
                gx, gy = unique_xy(gce, quantity)
                for er in expected.itertuples(index=False):
                    base = {
                        "L": 6, "U": u, "Ntot": n, "density": n / VOLUME,
                        "quantity": quantity, "quantity_label": label,
                        "beta_CE": float(er.beta), "T_CE": float(er.T),
                    }
                    rec = ce_map.get(round(float(er.beta), 10))
                    if str(er.status) == "sign_limited":
                        observed.append({**base, "Q_CE": np.nan, "CE_stderr": np.nan,
                                         "status": "sign_limited", "reason": "abs_average_phase_below_0p002",
                                         "n_roots": 0, "T_GCE": np.nan,
                                         "delta_T_over_T_CE": np.nan, "sigma_ratio": np.nan, "roots": ""})
                    elif rec is None or len(gx) < 2:
                        observed.append({**base, "Q_CE": np.nan, "CE_stderr": np.nan,
                                         "status": str(er.status) if rec is None else "no_gce_curve",
                                         "reason": "unfinished_or_insufficient_curve", "n_roots": 0,
                                         "T_GCE": np.nan, "delta_T_over_T_CE": np.nan,
                                         "sigma_ratio": np.nan, "roots": ""})
                    else:
                        qce = float(getattr(rec, quantity)); qerr = float(getattr(rec, quantity + "_err"))
                        ans = infer_temperature(float(er.T), qce, gx, gy, qerr, gce, quantity)
                        observed.append({**base, "Q_CE": qce, "CE_stderr": qerr, **ans})
    obs = pd.DataFrame(observed)
    write_table(args.outdir / "data" / "l6_thermometry_observed.tsv", obs)

    summaries: list[dict[str, object]] = []
    for quantity, label in PRIMARY:
        g = obs[obs.quantity == quantity]
        total = len(g)
        unique = g.status.eq("unique_solution")
        ambiguous = g.status.isin(["multiple_solutions", "flat_calibration"])
        vals = np.abs(pd.to_numeric(g.loc[unique, "delta_T_over_T_CE"], errors="coerce").dropna().to_numpy(float))
        summaries.append({
            "L": 6, "quantity": quantity, "quantity_label": label, "points": total,
            "unique_fraction": float(unique.mean()),
            "no_solution_fraction": float(g.status.eq("no_solution").mean()),
            "ambiguous_or_flat_fraction": float(ambiguous.mean()),
            "sign_limited_fraction": float(g.status.eq("sign_limited").mean()),
            "unfinished_fraction": float(g.status.isin(["incomplete", "waiting_manifest", "invalid", "no_gce_curve"]).mean()),
            "median_abs_deltaT_over_TCE": float(np.median(vals)) if len(vals) else np.nan,
            "p90_abs_deltaT_over_TCE": float(np.quantile(vals, .9)) if len(vals) else np.nan,
            "median_signed_deltaT_over_TCE": float(np.median(g.loc[unique, "delta_T_over_T_CE"])) if unique.any() else np.nan,
        })
    summary = pd.DataFrame(summaries)
    write_table(args.outdir / "data" / "l6_thermometry_quantity_summary.tsv", summary)

    assets: list[dict[str, object]] = []
    figures = args.outdir / "figures"
    with PdfPages(args.outdir / "L6_equal_time_thermometry_14slide_assets.pdf") as pdf:
        order = 1
        for quantity, label in PRIMARY:
            stem = figures / f"{order:02d}_L6_comparison_{quantity}"
            save_figure(comparison_figure(data, status, quantity, label), stem, pdf)
            assets.append({"order": order, "kind": "primary_comparison", "quantity": quantity,
                           "title": f"L=6 — {label}: CE versus GCE", "png": str(stem.with_suffix('.png')), "pdf": str(stem.with_suffix('.pdf'))})
            order += 1
        for quantity, label in NNN:
            stem = figures / f"{order:02d}_L6_comparison_{quantity}"
            save_figure(comparison_figure(data, status, quantity, label), stem, pdf)
            assets.append({"order": order, "kind": "nnn_comparison", "quantity": quantity,
                           "title": f"L=6 — {label}: CE versus GCE", "png": str(stem.with_suffix('.png')), "pdf": str(stem.with_suffix('.pdf'))})
            order += 1
        mismatch_specs = (
            ("delta_CE_minus_GCE", "L=6 — Absolute CE−GCE mismatch versus T", "O_CE−O_GCE", "absolute"),
            ("relative_CE_minus_GCE_over_abs_GCE", "L=6 — Fractional CE−GCE mismatch versus T", "(O_CE−O_GCE)/|O_GCE|", "relative"),
            ("gce_range_normalized_CE_minus_GCE", "L=6 — GCE-range-normalized mismatch versus T", "ΔO/range_T(O_GCE)", "range"),
        )
        for column, title, ylabel, name in mismatch_specs:
            stem = figures / f"{order:02d}_L6_mismatch_{name}"
            save_figure(mismatch_figure(mismatch, column, title, ylabel), stem, pdf)
            assets.append({"order": order, "kind": "mismatch", "quantity": name, "title": title,
                           "png": str(stem.with_suffix('.png')), "pdf": str(stem.with_suffix('.pdf'))})
            order += 1
        stem = figures / f"{order:02d}_L6_thermometry_scorecard"
        save_figure(scorecard_figure(summary), stem, pdf)
        assets.append({"order": order, "kind": "thermometry_scorecard", "quantity": "all",
                       "title": "L=6 thermometry scorecard", "png": str(stem.with_suffix('.png')), "pdf": str(stem.with_suffix('.pdf'))})
        order += 1
        for quantity, label in PRIMARY:
            stem = figures / f"{order:02d}_L6_thermometer_{quantity}"
            save_figure(thermometer_figure(obs, status, quantity, label), stem, pdf)
            assets.append({"order": order, "kind": "thermometer", "quantity": quantity,
                           "title": f"L=6 — {label} thermometry", "png": str(stem.with_suffix('.png')), "pdf": str(stem.with_suffix('.pdf'))})
            order += 1
    if order != 15:
        raise SystemExit(f"expected 14 slide assets, generated {order - 1}")
    write_table(args.outdir / "l6_slide_assets.tsv", pd.DataFrame(assets))
    print(f"wrote {len(assets)} slide assets under {args.outdir}")
    print("thermometry convention: (T_GCE-T_CE)/T_CE; actual unique CE points only")


if __name__ == "__main__":
    main()
