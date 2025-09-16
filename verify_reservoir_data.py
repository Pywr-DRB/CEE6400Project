#!/usr/bin/env python3
"""
verify_reservoir_data.py

Comprehensive verification and visualization utility for reservoir
time series used in DRB modeling. The script inspects observed inflow,
release, and storage against configuration “context” bounds, proposes
data-driven bound suggestions from percentiles, and (optionally) makes a
three-panel diagnostic plot (inflow / storage / release) with guideline
lines for quick visual QA.

------------------------------------------------------------------------------
WHAT THIS SCRIPT DOES
------------------------------------------------------------------------------
1) Loads observational training data
   - Uses methods.load.observations.get_observational_training_data to read
     time-aligned daily series for each reservoir:
       • inflow  (units of flow, e.g., MGD)
       • release (units of flow, e.g., MGD)
       • storage (units of volume, e.g., MG)
   - If --scaled-inflows is set (default: True), inflows are the scaled /
     preprocessed series consistent with the modeling workflow.

2) Builds a unified “policy context” for bounds
   - Pulls canonical settings from pywrdrb.release_policies.config:
       • reservoir_capacity
       • inflow_bounds_by_reservoir (I_min/I_max)
       • drbc_conservation_releases (R_min)
   - Uses methods.utils.release_constraints.get_release_minmax_release_dict()
     (if available) to incorporate observed min/max releases as fallbacks.
   - Calls get_policy_context(...) to assemble a single consistent context,
     applying overrides so the script’s verification uses the same bounds
     the policies would see.
   - Final context (per reservoir) includes:
       • storage_capacity               → used as S_max guideline
       • x_min[1], x_max[1]             → I_min, I_max inflow bounds
       • release_min, release_max       → R_min, R_max release bounds

3) Computes descriptive statistics and checks
   - For each series (inflow/release/storage), reports:
       • min, max, mean, and selected percentiles (p5, p50, p95)
   - Compares observed ranges to configured context bounds with a small
     tolerance:
       • inflow:   [I_min_cfg, I_max_cfg]
       • release:  [R_min_cfg, R_max_cfg]
       • storage:  [0, S_cap]
     Flags any exceedances (e.g., obs_release_exceeds_Rmax).
   - Emits magnitudes for exceedances to help tune bounds.

4) Proposes “suggested” bounds from percentiles
   - Upper suggestions: percentile(hi) * sf (default hi=p99, sf=1.05)
   - Lower suggestions: percentile(lo) / lsf (default lo=p01, lsf=1.05)
   - Lower suggestions are optionally floored at 0 for physical realism.
   - These are meant as *starting points* for tuning configuration bounds.

5) (Optional) Plots a 3-panel diagnostic figure
   - Inflow panel: series with dashed I_min / I_max guideline lines.
   - Storage panel: series with dashed Capacity line (S_cap).
   - Release panel: series (optionally smoothed/rolling mean) with dashed
     R_min / R_max guidelines; can be plotted on a log y-axis.
   - All y-axis labels include units (flow_units for inflow/release and
     storage_units for storage). Units are labels only; no conversion is
     applied here.

6) (Optional) Writes a CSV summary
   - One row per reservoir containing:
       • Context bounds used
       • Observed min/max/mean
       • Suggested min/max and deltas vs. context
       • Boolean flags for out-of-range observations

------------------------------------------------------------------------------
UNITS & GUIDELINE LINES ON THE PLOTS
------------------------------------------------------------------------------
• Units displayed on axes are controlled by:
    --flow-units (default "MGD") for inflow and release axes
    --storage-units (default "MG") for storage axis
  The script only *labels* units; it does not convert values.

• Guideline lines come from the context produced by get_policy_context(...):
    Inflow panel:  I_min = ctx["x_min"][1],  I_max = ctx["x_max"][1]
    Storage panel: S_cap = ctx["storage_capacity"]
    Release panel: R_min = ctx["release_min"], R_max = ctx["release_max"]

------------------------------------------------------------------------------
CLI ARGUMENTS
------------------------------------------------------------------------------
--reservoir            Name in methods.config.reservoir_options (case-insensitive)
                       or "ALL" to process every reservoir. (default: ALL)

--scaled-inflows       Use scaled inflows as provided by the workflow. (default: True)

--csv                  Optional path to write a summary CSV (one line per reservoir).

--hi                   Percentile for *upper* suggestions (choices: p95, p99, p995).
                       Default: p99.

--sf                   Safety factor applied to the upper percentile (multiply).
                       Default: 1.05.

--lo                   Percentile for *lower* suggestions (choices: p01, p05).
                       Default: p01.

--lsf                  Safety factor applied to the lower percentile (divide).
                       Default: 1.05. Lower suggestions are floored at 0.

--tol                  Tolerance for numeric comparisons to avoid equality noise.
                       Default: 1e-6.

--plots                If set, create the 3-panel inflow/storage/release plot.

--figdir               Directory where plots are saved when --save-figs is set.
                       Default: ./figures/verify

--save-figs            Save the generated plots to --figdir.

--show-figs            Show plots interactively (useful when running locally).

--release-log-scale    Plot the release panel on a logarithmic y-axis.

--smooth-release       Optional rolling window (in days) to smooth the release
                       series for plotting (0 = no smoothing). Default: 0.

--flow-units           Label for inflow/release axes (e.g., "MGD", "cfs"). Default: "MGD".

--storage-units        Label for storage axis (e.g., "MG", "ac-ft"). Default: "MG".

------------------------------------------------------------------------------
USAGE EXAMPLES
------------------------------------------------------------------------------
# 1) Run checks for all reservoirs and draw/save plots
python verify_reservoir_data.py \
  --plots --save-figs --figdir ./figures/verify \
  --release-log-scale --smooth-release 7 \
  --flow-units MGD --storage-units MG

# 2) Single reservoir with CSV output (no plots)
python verify_reservoir_data.py --reservoir prompton --csv ./outputs/verify.csv

# 3) Explore different suggestion settings
python verify_reservoir_data.py --hi p995 --sf 1.10 --lo p05 --lsf 1.02

------------------------------------------------------------------------------
OUTPUTS
------------------------------------------------------------------------------
Console:
  • Per-reservoir report of context bounds, observed stats, suggestion values,
    and warnings if observations fall outside context bounds.

CSV (optional):
  • One row per reservoir with context, observed ranges, suggestions, deltas,
    and boolean flags. Good for auditing or bulk tuning.

Figures (optional):
  • <figdir>/<reservoir>_inflow_storage_release.png
    Three-panel time-series plot with guideline lines for quick QA.

------------------------------------------------------------------------------
NOTES & GOTCHAS
------------------------------------------------------------------------------
• “ALL” uses methods.config.reservoir_options to iterate reservoirs. Ensure it
  is up to date and matches your data files.

• The script does not alter or write configuration—its suggestions are on
  screen/CSV only. Use them to *inform* updates to your config files.

• Units are labels only; upstream data should already be in consistent units.

• If observed data exceed context bounds frequently, consider:
    (a) verifying data quality and preprocessing,
    (b) revisiting configured bounds with domain knowledge,
    (c) using percentile-based suggestions here as a starting point.

• If release_min/release_max are not available from the observed min/max
  dictionary, DRBC conservation releases and configured max are used via the
  policy context assembly.

------------------------------------------------------------------------------
IMPLEMENTATION OVERVIEW
------------------------------------------------------------------------------
1) Load data  → 2) Build context  → 3) Summarize & check  → 4) Suggest bounds
→ 5) (Optional) Plot  → 6) (Optional) Write CSV

"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- Project imports (adjust paths if needed) ---
from methods.load.observations import get_observational_training_data
from methods.config import PROCESSED_DATA_DIR, reservoir_options 
from methods.utils.release_constraints import get_release_minmax_release_dict

from pywrdrb.release_policies.config import (
    reservoir_capacity,
    inflow_bounds_by_reservoir,
    drbc_conservation_releases,
    get_policy_context,
)

# -------------------------------
# Helpers
# -------------------------------

def pcts(x, q=(5, 25, 50, 75, 95)):
    x = np.asarray(x, dtype=float)
    return {f"p{qq}": float(np.nanpercentile(x, qq)) for qq in q}

def summarize_series(name, s):
    s = np.asarray(s, dtype=float)
    return {
        "name": name,
        "count": int(np.isfinite(s).sum()),
        "nans": int(np.isnan(s).sum()),
        "min": float(np.nanmin(s)),
        "max": float(np.nanmax(s)),
        "mean": float(np.nanmean(s)),
        **pcts(s),
    }

def suggest_upper(series, hi="p99", sf=1.05):
    """Upper bound = percentile(hi) * sf."""
    vals = np.asarray(series, dtype=float)
    qmap = {"p95": 95.0, "p99": 99.0, "p995": 99.5}
    if hi not in qmap:
        raise ValueError(f"hi must be one of {list(qmap)}, got {hi}")
    return float(np.nanpercentile(vals, qmap[hi]) * sf)

def suggest_lower(series, lo="p01", lsf=1.05, floor_zero=True):
    """Lower bound = percentile(lo) / lsf (optionally floored at 0)."""
    vals = np.asarray(series, dtype=float)
    qmap = {"p01": 1.0, "p05": 5.0}
    if lo not in qmap:
        raise ValueError(f"lo must be one of {list(qmap)}, got {lo}")
    lb = float(np.nanpercentile(vals, qmap[lo]) / lsf)
    if floor_zero:
        lb = max(0.0, lb)
    return lb

def check_bounds(res_name, inflow, release, storage, ctx, tol=1e-6):
    """Compare obs to config/context with a small tolerance."""
    results = {}

    # Context / config
    cap   = float(ctx["storage_capacity"])
    I_min = float(ctx["x_min"][1]); I_max = float(ctx["x_max"][1])
    R_min = float(ctx["release_min"]);   R_max = float(ctx["release_max"])

    # Observed ranges
    i_min = float(np.nanmin(inflow));  i_max = float(np.nanmax(inflow))
    r_min = float(np.nanmin(release)); r_max = float(np.nanmax(release))
    s_min = float(np.nanmin(storage)); s_max = float(np.nanmax(storage))

    # Checks
    results["obs_inflow_exceeds_Imax"]  = (i_max - I_max) > tol
    results["obs_inflow_below_Imin"]    = (I_min - i_min) > tol
    results["obs_release_exceeds_Rmax"] = (r_max - R_max) > tol
    results["obs_release_below_Rmin"]   = (R_min - r_min) > tol
    results["obs_storage_exceeds_cap"]  = (s_max - cap)  > tol
    results["obs_storage_negative"]     = (0.0 - s_min)  > tol

    # Magnitudes
    results["obs_storage_max_minus_cap"]  = float(s_max - cap)
    results["obs_release_max_minus_Rmax"] = float(r_max - R_max)
    results["obs_inflow_max_minus_Imax"]  = float(i_max - I_max)

    return results

def plot_inflow_storage_release(idx, inflow, storage, release, ctx, res_name,
                                outdir=None, save=False, show=False,
                                log_release=False, smooth_release=0, dpi=160,
                                flow_units="MGD", storage_units="MG"):
    S_cap = float(ctx["storage_capacity"])
    I_min = float(ctx["x_min"][1]); I_max = float(ctx["x_max"][1])
    R_min = float(ctx["release_min"]); R_max = float(ctx["release_max"])

    rel_plot = release.copy()
    if smooth_release and smooth_release > 0:
        rel_plot = pd.Series(rel_plot, index=idx).rolling(smooth_release, min_periods=1).mean().values

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # 1) Inflow
    axes[0].plot(idx, inflow, linewidth=1.0, label="Inflow")
    axes[0].axhline(I_min, ls="--", linewidth=0.9, label=f"I_min ({flow_units})")
    axes[0].axhline(I_max, ls="--", linewidth=0.9, label=f"I_max ({flow_units})")
    axes[0].set_ylabel(f"Inflow ({flow_units})")
    axes[0].set_title(f"{res_name} — Inflow / Storage / Release")
    axes[0].legend(loc="upper right", frameon=False)

    # 2) Storage
    axes[1].plot(idx, storage, linewidth=1.0, label="Storage")
    axes[1].axhline(S_cap, ls="--", linewidth=0.9, label=f"Capacity ({storage_units})")
    axes[1].set_ylabel(f"Storage ({storage_units})")
    axes[1].legend(loc="upper right", frameon=False)

    # 3) Release
    axes[2].plot(idx, rel_plot, linewidth=1.0, label="Release")
    axes[2].axhline(R_min, ls="--", linewidth=0.9, label=f"R_min ({flow_units})")
    axes[2].axhline(R_max, ls="--", linewidth=0.9, label=f"R_max ({flow_units})")
    if log_release:
        axes[2].set_yscale("log")
    axes[2].set_ylabel(f"Release ({flow_units})")
    axes[2].set_xlabel("Date")
    axes[2].legend(loc="upper right", frameon=False)

    # nice dates
    axes[2].xaxis.set_major_locator(mdates.AutoDateLocator())
    axes[2].xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(axes[2].xaxis.get_major_locator())
    )

    fig.tight_layout()

    if save and outdir:
        os.makedirs(outdir, exist_ok=True)
        fname = os.path.join(outdir, f"{res_name}_inflow_storage_release.png")
        fig.savefig(fname, dpi=dpi)
        print(f"[✓] Saved figure: {fname}")
    if show:
        plt.show()
    plt.close(fig)


# -------------------------------
# Main
# -------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reservoir", default="ALL", help="Name in methods.config.reservoir_options, or ALL")
    parser.add_argument("--scaled-inflows", action="store_true", default=True, help="Use scaled inflows")
    parser.add_argument("--csv", default=None, help="Optional path to write a one-line-per-reservoir summary CSV")
    parser.add_argument("--hi", default="p99", choices=["p95","p99","p995"],
                        help="Percentile for suggested *upper* bounds (default: p99)")
    parser.add_argument("--sf", type=float, default=1.05,
                        help="Safety factor for suggested *upper* bounds (default: 1.05)")
    parser.add_argument("--lo", default="p01", choices=["p01","p05"],
                        help="Percentile for suggested *lower* bounds (default: p01)")
    parser.add_argument("--lsf", type=float, default=1.05,
                        help="Safety factor for suggested *lower* bounds (divides; default: 1.05)")
    parser.add_argument("--tol", type=float, default=1e-6,
                        help="Tolerance for bound checks to avoid equality noise (default: 1e-6)")
    parser.add_argument("--plots", action="store_true",
                        help="Make 3-panel plots (inflow, storage, release) for each reservoir")
    parser.add_argument("--figdir", default="./figures/verify",
                        help="Directory to save figures when --save-figs is set")
    parser.add_argument("--save-figs", action="store_true",
                        help="Save plots to --figdir")
    parser.add_argument("--show-figs", action="store_true",
                        help="Show plots interactively")
    parser.add_argument("--release-log-scale", action="store_true",
                        help="Use log scale on release axis")
    parser.add_argument("--smooth-release", type=int, default=0,
                        help="Optional rolling window for release smoothing (days). 0 = no smoothing")
    parser.add_argument("--flow-units", default="MGD",
                        help="Units label for inflow/release axes (e.g., MGD, cfs)")
    parser.add_argument("--storage-units", default="MG",
                        help="Units label for storage axis (e.g., MG, ac-ft)")

    args = parser.parse_args()

    # Observed min/max release dicts (if available)
    try:
        obs_release_min, obs_release_max = get_release_minmax_release_dict()
    except Exception:
        obs_release_min, obs_release_max = {}, {}

    targets = reservoir_options if args.reservoir.upper() == "ALL" else [args.reservoir]

    rows = []
    for r in targets:
        try:
            inflow_obs, release_obs, storage_obs = get_observational_training_data(
                reservoir_name=r,
                data_dir=PROCESSED_DATA_DIR,
                as_numpy=False,
                scaled_inflows=args.scaled_inflows,
            )

            # Basic time info
            idx = inflow_obs.index
            start, end = idx.min(), idx.max()

            inflow  = inflow_obs.values.flatten()
            release = release_obs.values.flatten()
            storage = storage_obs.values.flatten()

            # Context assembled from config (overrides with canonical values)
            cap_cfg   = reservoir_capacity.get(r, np.nan)
            I_bnds    = inflow_bounds_by_reservoir.get(r, {"I_min": 0.0, "I_max": np.nan})
            R_min_cfg = drbc_conservation_releases.get(r, obs_release_min.get(r, 0.0))
            R_max_cfg = obs_release_max.get(r, np.nan)

            ctx = get_policy_context(
                r,
                release_min_override=R_min_cfg,
                release_max_override=R_max_cfg,
                capacity_override=cap_cfg,
                inflow_bounds_override=(I_bnds["I_min"], I_bnds["I_max"]),
            )

            # Summaries
            sinfo = summarize_series("storage", storage)
            rinfo = summarize_series("release", release)
            iinfo = summarize_series("inflow",  inflow)

            # Suggestions (upper and lower)
            R_max_suggest = suggest_upper(release, hi=args.hi, sf=args.sf)
            I_max_suggest = suggest_upper(inflow,  hi=args.hi, sf=args.sf)
            S_max_suggest = suggest_upper(storage, hi=args.hi, sf=args.sf)

            R_min_suggest = suggest_lower(release, lo=args.lo, lsf=args.lsf, floor_zero=True)
            I_min_suggest = suggest_lower(inflow,  lo=args.lo, lsf=args.lsf, floor_zero=True)
            S_min_suggest = suggest_lower(storage, lo=args.lo, lsf=args.lsf, floor_zero=True)  # info only; cfg min is 0

            # Checks against context
            checks = check_bounds(r, inflow, release, storage, ctx, tol=args.tol)

            # Convenience cfg bounds for printing/CSV
            S_min_cfg = 0.0
            S_cap     = float(ctx["storage_capacity"])
            S_max_cfg = S_cap
            I_min_cfg = float(ctx["x_min"][1]); I_max_cfg = float(ctx["x_max"][1])
            R_min_cfg = float(ctx["release_min"]); R_max_cfg = float(ctx["release_max"])

            # Print block
            print("=" * 80)
            print(f"{r.upper()}  |  {start.date()} → {end.date()}  |  n={len(idx)}")
            print(f"[CTX] storage_cfg∈[{S_min_cfg:.2f}, {S_max_cfg:.2f}] | "
                  f"inflow_cfg∈[{I_min_cfg:.2f}, {I_max_cfg:.2f}] | "
                  f"release_cfg∈[{R_min_cfg:.2f}, {R_max_cfg:.2f}]")
            print(f"[OBS] storage: min={sinfo['min']:.2f} max={sinfo['max']:.2f} mean={sinfo['mean']:.2f} "
                  f"(p5={sinfo['p5']:.2f}, p50={sinfo['p50']:.2f}, p95={sinfo['p95']:.2f})")
            print(f"[OBS] release: min={rinfo['min']:.2f} max={rinfo['max']:.2f} mean={rinfo['mean']:.2f} "
                  f"(p5={rinfo['p5']:.2f}, p50={rinfo['p50']:.2f}, p95={rinfo['p95']:.2f})")
            print(f"[OBS] inflow : min={iinfo['min']:.2f} max={iinfo['max']:.2f} mean={iinfo['mean']:.2f} "
                  f"(p5={iinfo['p5']:.2f}, p50={iinfo['p50']:.2f}, p95={iinfo['p95']:.2f})")
            print(f"[SUGGEST UPPER] R_max≈{R_max_suggest:.2f}, I_max≈{I_max_suggest:.2f}, S_max≈{S_max_suggest:.2f} "
                  f"(hi={args.hi}, sf={args.sf})")
            print(f"[SUGGEST LOWER] R_min≈{R_min_suggest:.2f}, I_min≈{I_min_suggest:.2f}, S_min≈{S_min_suggest:.2f} "
                  f"(lo={args.lo}, lsf={args.lsf})")

            # Warning lines
            warnings = []
            if checks["obs_storage_exceeds_cap"]:
                warnings.append(f"storage_max({sinfo['max']:.2f}) > S_max_cfg({S_max_cfg:.2f})")
            if checks["obs_storage_negative"]:
                warnings.append("storage_min < 0")
            if checks["obs_release_exceeds_Rmax"]:
                warnings.append(f"release_max({rinfo['max']:.2f}) > R_max_cfg({R_max_cfg:.2f})")
            if checks["obs_release_below_Rmin"]:
                warnings.append(f"release_min({rinfo['min']:.2f}) < R_min_cfg({R_min_cfg:.2f})")
            if checks["obs_inflow_exceeds_Imax"]:
                warnings.append(f"inflow_max({iinfo['max']:.2f}) > I_max_cfg({I_max_cfg:.2f})")
            if checks["obs_inflow_below_Imin"]:
                warnings.append(f"inflow_min({iinfo['min']:.2f}) < I_min_cfg({I_min_cfg:.2f})")

            if warnings:
                print("[WARN]", "; ".join(warnings))
            else:
                print("[OK] Observations fall within configured ranges.")

            # Row for CSV (all cfg bounds + observed + suggestions + deltas)
            rows.append({
                "reservoir": r,
                "n_days": len(idx),
                "date_start": start,
                "date_end": end,

                # Configured (context) bounds
                "S_min_cfg": S_min_cfg, "S_max_cfg": S_max_cfg,
                "I_min_cfg": I_min_cfg, "I_max_cfg": I_max_cfg,
                "R_min_cfg": R_min_cfg, "R_max_cfg": R_max_cfg,

                # Observed ranges (and means)
                "S_min": sinfo["min"], "S_max": sinfo["max"], "S_mean": sinfo["mean"],
                "R_min": rinfo["min"], "R_max": rinfo["max"], "R_mean": rinfo["mean"],
                "I_min": iinfo["min"], "I_max": iinfo["max"], "I_mean": iinfo["mean"],

                # Suggested upper/lower
                "I_max_suggest": I_max_suggest, "I_min_suggest": I_min_suggest,
                "R_max_suggest": R_max_suggest, "R_min_suggest": R_min_suggest,
                "S_max_suggest": S_max_suggest, "S_min_suggest": S_min_suggest,

                # Deltas vs. current config
                "I_max_suggest_minus_cfg": float(I_max_suggest - I_max_cfg),
                "I_min_suggest_minus_cfg": float(I_min_suggest - I_min_cfg),
                "R_max_suggest_minus_cfg": float(R_max_suggest - R_max_cfg),
                "R_min_suggest_minus_cfg": float(R_min_suggest - R_min_cfg),
                "S_max_suggest_minus_cfg": float(S_max_suggest - S_max_cfg),
                "S_min_suggest_minus_cfg": float(S_min_suggest - S_min_cfg),

                # Flags & magnitudes
                **checks
            })

        except Exception as e:
            print("=" * 80)
            print(f"{r.upper()}  |  ERROR: {e}")
            rows.append({"reservoir": r, "error": str(e)})

        if args.plots:
            plot_inflow_storage_release(
                idx=idx,
                inflow=inflow,
                storage=storage,
                release=release,
                ctx=ctx,
                res_name=r,
                outdir=args.figdir,
                save=args.save_figs,
                show=args.show_figs,
                log_release=args.release_log_scale,
                smooth_release=args.smooth_release,
                flow_units=args.flow_units,
                storage_units=args.storage_units
            )
    if args.csv:
        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        df.to_csv(args.csv, index=False)
        print(f"\n[✓] Wrote summary CSV to {args.csv}")

if __name__ == "__main__":
    main()
