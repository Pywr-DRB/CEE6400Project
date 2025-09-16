#!/usr/bin/env python3
"""
verify_policies.py
-------------------------------------------------------------------------------
One-off, reproducible runner to **simulate a single reservoir with a chosen
release policy and parameter vector**, compute the same objective metrics used by
the optimization workflow, and optionally verify physical/operational
consistency (mass balance and constraint parity). Designed for quick
“does-this-set-of-params-make-sense?” checks and for producing tidy, portable
artifacts (PNG + CSV) for reports.

WHY THIS EXISTS
---------------
During policy development and calibration, it’s useful to bypass the full MOEA
pipeline and instead run a single forward simulation with a known parameter
vector. This script:
  • Loads the observed training series (inflow, release, storage) for a
    specified reservoir.
  • Constructs a `Reservoir` with the selected policy (STARFIT | RBF | PWL),
    pulling its **operating context** from config (capacity, R_min/R_max,
    inflow bounds, scaling) so that the forward-sim matches the optimization
    environment.
  • Runs the simulation and **computes release and storage objectives** using
    the same `ObjectiveCalculator` metrics the optimizer uses.
  • Optionally **verifies mass balance** (right-indexed flows) and **summarizes
    constraint binding/violation fractions** to diagnose unrealistic behavior.
  • Saves a side-by-side **storage/release time-series plot** and a **CSV
    summary** of parameters + objectives.

ASSUMPTIONS & UNITS
-------------------
• Time step: daily (irregular day lengths supported via the DatetimeIndex).
• Units (by convention of the DRB preprocessing):
    - Inflow, Release, Spill: **MGD** (million gallons per day).
    - Storage: **MG** (million gallons).
  The script labels figures with these units; it does not perform unit
  conversion internally.
• Mass-balance convention: **right-indexed flows** — the flow values at time
  *t* represent the volume over the interval *(t-1, t]*. Residuals are computed
  as ΔS − (In − Rel − Spill)·Δt, where Δt is the day length in days.

DATA/CONTEXT SOURCES
--------------------
• Observations are loaded from `methods.load.observations.get_observational_training_data`
  (returning Pandas Series with a shared DatetimeIndex).
• Reservoir policy context (capacity, bounds, scaling) is assembled inside
  `Reservoir` via `pywrdrb.release_policies.config.get_policy_context(...)`.
  This ensures the simulation sees the **same bounds and scaling rules** as the
  optimizer (e.g., storage_capacity, release_min/max, inflow x_min/x_max).

POLICY PARAMETERS
-----------------
• You can provide parameters:
    (a) **Inline** with `--params var1,var2,...` (exact length must match the
        policy’s NVARS), or
    (b) **From CSV** with `--params-csv` (columns `var1..varN` or the first N
        numeric columns), optionally selecting `--row-idx`.
• For STARFIT only, the script includes a **baseline** example vector for
  `fewalter`. For other reservoirs or policies, **you must supply params**.
• Bounds:
    - Expected length and numeric bounds per policy are read from
      `methods.config.policy_n_params` and `policy_param_bounds`.
    - If `--clip-to-bounds` is set, inputs are clipped to legal ranges before
      running; otherwise a warning is printed for out-of-bounds entries.

OBJECTIVES & OPTIONAL CHECKS
----------------------------
• Objectives:
    - Computed with `methods.metrics.objectives.ObjectiveCalculator`, using
      `RELEASE_METRICS` and `STORAGE_METRICS` from `methods.config`.
    - If `--add-spill-to-release` (default **True**), spill is added to the
      simulated release before computing **release** objectives (to match some
      down-basin accounting conventions). Mass-balance checks still use the
      raw release component to avoid double-counting spill.

• Mass balance (`--verify-mass-balance`):
    - Computes residual time series (MG) and summarizes max/RMSE/percentiles.
    - Computes cumulative closure (Σ netQ − ΔS_total).
    - Reports capacity/positivity violations and a “spill sanity” check
      (spill only when storage ≈ capacity).
    - Saves artifacts to `outdir`:
        * CSV of residual components (`*_mass_balance_right_residuals.csv`)
        * Time-series PNG (`*_mass_balance_right_ts.png`)
        * Histogram PNG (`*_mass_balance_right_hist.png`)
    - Tolerances: `--mb-tol-abs` (MG), `--mb-tol-rel` (fraction of a typical
      daily volume based on the 95th percentile of |netQ|).

• Constraint parity (`--verify-constraints`):
    - Reports **fractions of timesteps** at or beyond:
        * storage at capacity (binding S_max),
        * release at min/max (binding R_min/R_max),
        * violations (S<0, S>cap, R<R_min, R>R_max).
    - Useful for diagnosing over-constrained or ineffective parameterizations.

• Optimization parity (`--verify-optimization`):
    - Re-runs a minimal optimization-like evaluation on the same period via
      `evaluate_like_optimization(...)` and compares the concatenated
      [release_objs, storage_objs] with the ones computed in this run.
    - Helpful to ensure metric calculators and series alignment match the
      optimization code path.

CLI
---
--reservoir (required)           Reservoir key (e.g., fewalter | beltzvilleCombined | prompton).
--policy    (required)           One of: STARFIT | RBF | PWL.

--params <csvlist>               Inline comma-separated parameter vector (var1,var2,...).
--params-csv <path>              CSV containing var1..varN (or first N numeric columns).
--row-idx <int>                  Row to read from --params-csv (default: 0).

--clip-to-bounds                 Clip provided params to policy bounds before sim.
--print-param-order              Print policy’s exact parameter order and exit.

--scaled-inflows                 Use pre-scaled inflows from the workflow.
--add-spill-to-release           Add spill to release before release objectives (default: ON).

--print-context                  Print one-line summary of S_cap, I_min/max, R_min/max.
--print-violations               If the policy accumulates a violation log, print its summary.

--verify-mass-balance            Run mass-balance QA and save residual CSV/plots.
--mb-tol-abs <float>             Absolute tolerance (MG) for step/cumulative checks (default: 1e-6).
--mb-tol-rel <float>             Relative tolerance vs typical daily volume (default: 1e-6).

--verify-constraints             Print constraint binding/violation fractions.

--verify-optimization            Compare objectives to an optimization-like evaluation.

--start YYYY-MM-DD               Optional plot/window start date (defaults to first obs date).
--end   YYYY-MM-DD               Optional plot/window end date (defaults to last obs date).

--outdir <dir>                   Output directory (default: FIG_DIR/simple_runs).
--save-figs                      Save the combined storage/release PNG.
--show-figs                      Show the figure interactively.

OUTPUTS
-------
1) **Console report**
   - Reservoir/policy, parameter vector summary.
   - Release and storage objective values (one line per metric).
   - Optional: context line; mass-balance summary; constraint parity; optimization parity.

2) **CSV summary**
   - `<outdir>/summary_<reservoir>_<policy>.csv`
   - Contains reservoir/policy, the full parameter vector, and all objective values.

3) **Figure**
   - `<outdir>/<reservoir>_<policy>_storage_release_<start>_<end>.png`
   - Two-panel: observed vs simulated **storage** (top) and **release** (bottom).
   - Release curve uses (release + spill) if `--add-spill-to-release` is set.

4) **(If mass balance enabled)** residual artifacts (see above).

TYPICAL WORKFLOWS
-----------------
• Use baseline STARFIT for FEWALTER:
    python verify_policies.py --reservoir fewalter --policy STARFIT --save-figs

• Inline STARFIT params for PROMPTON:
    python verify_policies.py --reservoir prompton --policy STARFIT \
        --params 15.08,5.0,20.0,0.0,-15.0,9.0,1.6,14.2,-1.0,-30.0,0.2118,-0.0357,0.1302,-0.0248,-0.123,0.183,0.732 \
        --verify-mass-balance --save-figs

• From CSV (first row), clip to bounds, and compare objectives to the
  optimization-like pipeline:
    python verify_policies.py --reservoir beltzvilleCombined --policy STARFIT \
        --params-csv ./my_params.csv --clip-to-bounds --verify-optimization --save-figs

IMPLEMENTATION NOTES
--------------------
• `Reservoir` loads and applies policy context internally (`get_policy_context`),
  so the forward simulation’s scaling/bounds match the optimization setup.
• STARFIT-only guard: `test_nor_constraint()` if available; if it fails, the
  parameterization is not considered feasible under its internal NOR
  structure (script does not abort; it warns).
• Mass-balance uses the **raw** release component (no added spill) to prevent
  double counting; release objectives optionally include spill if requested.
• Plot labels: “Storage (MG)” and “Release / Flow (MGD)”.
• Exit status: exceptions propagate as non-zero exit; otherwise zero.

GOTCHAS & TIPS
--------------
• If you see persistent mass-balance residuals, check:
    - DatetimeIndex regularity (no gaps/duplicates),
    - Units consistency (storage MG vs flows MGD),
    - Whether spill is being double-counted outside the MB routine.
• If objectives differ under `--verify-optimization`, ensure:
    - The same inflow scaling setting is used,
    - The same add-spill convention is applied on both sides,
    - Date windows and indexing assumptions match (right-indexed flows).
• For RBF/PWL, use `--print-param-order` to get the exact parameter layout
  (depends on `n_rbfs`, `n_rbf_inputs`, `n_segments`, `n_pwl_inputs`).

-------------------------------------------------------------------------------
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Project imports (paths assumed correct within repo) ---
from methods.reservoir.model import Reservoir
from methods.load.observations import get_observational_training_data
from methods.metrics.objectives import ObjectiveCalculator
from methods.config import RELEASE_METRICS, STORAGE_METRICS
from methods.config import n_rbfs, n_rbf_inputs, n_segments, n_pwl_inputs
from methods.config import (
    DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR, FIG_DIR,
    reservoir_capacity,
    policy_n_params, policy_param_bounds,
)

# ------------------------------
# Baseline STARFIT params (within-bounds exemplar)
# Note: Only 'fewalter' included to avoid out-of-bounds issues.
# For other reservoirs, pass --params or --params-csv.
# ------------------------------
BASELINE_STARFIT_PARAMS = {
    "fewalter": np.array([
        15.08, 5.0, 20.0,   0.0, -15.0,
        9.0, 1.6, 14.2,    -1.0, -30.0,
        0.2118, -0.0357, 0.1302, -0.0248,
        -0.123, 0.183, 0.732
    ], dtype=float),
}

# ------------------------------
# Utilities
# ------------------------------
def parse_params_inline(s: str, expected_n: int) -> np.ndarray:
    vals = [float(x.strip()) for x in s.split(",") if x.strip() != ""]
    if len(vals) != expected_n:
        raise ValueError(f"Expected {expected_n} params, got {len(vals)} from --params.")
    return np.array(vals, dtype=float)

def load_params_from_csv(path: str, expected_n: int, row_idx: int = 0) -> np.ndarray:
    df = pd.read_csv(path)
    # Accept either columns named var1..varN OR just the first N numeric columns
    var_cols = [c for c in df.columns if c.lower().startswith("var")]
    if var_cols:
        var_cols = sorted(var_cols, key=lambda c: int(''.join(ch for ch in c if ch.isdigit()) or 0))
    else:
        var_cols = df.columns.tolist()
    if len(var_cols) < expected_n:
        raise ValueError(f"CSV has only {len(var_cols)} columns; need at least {expected_n}.")
    row = df.iloc[row_idx]
    arr = np.array([row[c] for c in var_cols[:expected_n]], dtype=float)
    if arr.shape[0] != expected_n:
        raise ValueError(f"Row {row_idx} did not yield {expected_n} variables.")
    return arr

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p

# =========================
# Verification helpers
# =========================
def _summary_stats(x):
    x = np.asarray(x, dtype=float)
    return {
        "count": x.size,
        "min": float(np.nanmin(x)),
        "max": float(np.nanmax(x)),
        "mean": float(np.nanmean(x)),
        "rmse": float(np.sqrt(np.nanmean(np.square(x)))),
        "p95": float(np.nanpercentile(x, 95)),
        "p99": float(np.nanpercentile(x, 99)),
    }

def verify_mass_balance(storage, inflow, release, spill, s_cap,
                        dt_index, tol_abs=1e-6, tol_rel=1e-6,
                        outdir=None, prefix=""):
    """
    Right-indexed mass-balance check:
    Uses flows at time t to represent the interval (t-1, t].
    storage: MG (len T)
    inflow, release, spill: MGD (len T)
    dt_index: pandas.DatetimeIndex (len T)
    """
    storage = np.asarray(storage, float)
    inflow  = np.asarray(inflow,  float)
    release = np.asarray(release, float)
    spill   = np.asarray(spill,   float)

    # Δt per step (days)
    step_days = np.diff(pd.to_datetime(dt_index).values.astype("datetime64[ns]")) / np.timedelta64(1, "D")
    step_days = step_days.astype(float)

    # Storage change (MG) over each interval
    dS   = storage[1:] - storage[:-1]

    # Net flow over each interval (MG), using RIGHT indexing
    netQ = (inflow[1:] - release[1:] - spill[1:]) * step_days

    # Residual (MG)
    resid = dS - netQ

    # Cumulative closure
    cum_balance = float(np.sum(netQ) - (storage[-1] - storage[0]))

    # Stats & tolerances
    stats = _summary_stats(resid)
    stats["cum_balance"] = cum_balance
    scale = max(1.0, np.nanpercentile(np.abs(netQ), 95.0))
    ok_step = (np.nanmax(np.abs(resid)) <= max(tol_abs, tol_rel * scale))
    ok_cum  = (abs(cum_balance) <= max(10 * tol_abs, tol_rel * np.sum(np.abs(netQ))))

    # capacity/positivity + spill sanity
    cap_viol = np.nanmax(storage - (s_cap + 1e-6))
    neg_viol = -np.nanmin(storage)
    stats["cap_violation_max"] = float(max(0.0, cap_viol))
    stats["neg_storage_violation"] = float(max(0.0, neg_viol))
    spill_idxs = np.where(spill > 0.0)[0]
    spill_tol = max(1e-3, 1e-3 * s_cap)
    stats["spill_sanity_ok"] = (True if spill_idxs.size == 0
                                else bool(np.all(storage[spill_idxs] >= s_cap - spill_tol)))

    # Save artifacts
    if outdir is not None:
        df = pd.DataFrame({
            "residual_MG": resid,
            "dS_MG": dS,
            "netQ_MG": netQ,
            "step_days": step_days,
        }, index=dt_index[1:])
        fbase = f"{prefix}mass_balance_right"
        csv_path = os.path.join(outdir, f"{fbase}_residuals.csv")
        df.to_csv(csv_path, index_label="date")
        print(f"[✓] Wrote: {csv_path}")

        plt.figure(figsize=(12,4)); plt.plot(df.index, resid)
        plt.title("Mass-balance residual (MG) — right-indexed flows")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{fbase}_ts.png"), dpi=160); plt.close()

        plt.figure(figsize=(6,4)); plt.hist(resid, bins=60)
        plt.title("Residual histogram (MG) — right")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{fbase}_hist.png"), dpi=160); plt.close()

    # Print summary
    print("\n# Mass balance check (RIGHT-indexed flows)")
    print(f"  step_resid | max={stats['max']:.6g} MG, rmse={stats['rmse']:.6g} MG, p95={stats['p95']:.6g} MG")
    print(f"  cum_balance: {stats['cum_balance']:.6g} MG (should be ~0)")
    print(f"  cap_violation_max: {stats['cap_violation_max']:.6g} MG; neg_storage_violation: {stats['neg_storage_violation']:.6g} MG")
    print(f"  spill_sanity_ok: {stats['spill_sanity_ok']}")
    print("  [OK] Mass balance within tolerance." if (ok_step and ok_cum) else "  [WARN] Mass balance outside tolerance.")
    return stats

def verify_constraints_parity(reservoir, ctx, tol=1e-6):
    """Report fractions of timesteps binding or violating constraints."""
    S = reservoir.storage_array.astype(float)
    R = reservoir.release_array.astype(float)

    S_cap = float(ctx["storage_capacity"])
    Rmin = float(ctx.get("release_min", np.nan))
    Rmax = float(ctx.get("release_max", np.nan))

    eps = tol
    bind_S_max = np.mean(S >= S_cap - eps)
    bind_R_min = np.mean(R <= Rmin + eps) if np.isfinite(Rmin) else np.nan
    bind_R_max = np.mean(R >= Rmax - eps) if np.isfinite(Rmax) else np.nan

    vio_S_neg = np.mean(S < -eps)
    vio_S_cap = np.mean(S > S_cap + eps)
    vio_R_min = np.mean(R < Rmin - eps) if np.isfinite(Rmin) else np.nan
    vio_R_max = np.mean(R > Rmax + eps) if np.isfinite(Rmax) else np.nan

    print("\n# Constraint binding/violation summary (fractions)")
    print(f"  bind_S_max: {bind_S_max:.4f}, bind_R_min: {bind_R_min}, bind_R_max: {bind_R_max}")
    print(f"  vio_S<0: {vio_S_neg:.6f}, vio_S>cap: {vio_S_cap:.6f}, vio_R<Rmin: {vio_R_min}, vio_R>Rmax: {vio_R_max}")

def evaluate_like_optimization(inflow_array, release_obs, storage_obs, dt_index,
                               policy_type, policy_params, ctx, reservoir_name,
                               add_spill=True):
    """Mirror the optimization evaluation pipeline; return concatenated objectives."""
    test_res = Reservoir(
        inflow=inflow_array,
        dates=dt_index,
        capacity=ctx["storage_capacity"],
        policy_type=policy_type,
        policy_params=policy_params,
        initial_storage=float(storage_obs[0]),
        name=reservoir_name,
    )
    test_res.reset()
    test_res.run()
    sim_R = test_res.release_array.astype(float)
    sim_S = test_res.storage_array.astype(float)
    if add_spill:
        sim_R = sim_R + test_res.spill_array.astype(float)

    rel_calc = ObjectiveCalculator(metrics=RELEASE_METRICS)
    sto_calc = ObjectiveCalculator(metrics=STORAGE_METRICS)
    return np.r_[rel_calc.calculate(obs=release_obs, sim=sim_R),
                 sto_calc.calculate(obs=storage_obs, sim=sim_S)]
# ------------------------------
# Parameter order helpers (DROP-IN)
# ------------------------------
def get_param_names(policy: str):
    """
    Return the exact flat-vector parameter names (in order) expected by the policy.
    Use this to build/inspect '--params' lists.
    """
    policy = str(policy).upper()

    if policy == "STARFIT":
        # Fixed length = 17, matches STARFIT.parse_policy_params()
        return [
            "NORhi_mu", "NORhi_min", "NORhi_max", "NORhi_alpha", "NORhi_beta",
            "NORlo_mu", "NORlo_min", "NORlo_max", "NORlo_alpha", "NORlo_beta",
            "Release_alpha1", "Release_alpha2", "Release_beta1", "Release_beta2",
            "Release_c", "Release_p1", "Release_p2",
        ]

    if policy == "RBF":
        # Length = n_rbfs * (2*n_rbf_inputs + 1)
        # Layout: [w_1..w_n, c(1,S),c(1,I),c(1,D),..., c(n,S),c(n,I),c(n,D), r(1,S),..., r(n,D)]
        var_labels = ["storage", "inflow", "doy"][:n_rbf_inputs]
        names = []
        # weights
        for i in range(1, n_rbfs + 1):
            names.append(f"w{i}")
        # centers
        for i in range(1, n_rbfs + 1):
            for v in var_labels:
                names.append(f"c{i}_{v}")
        # scales
        for i in range(1, n_rbfs + 1):
            for v in var_labels:
                names.append(f"r{i}_{v}")
        return names

    if policy == "PWL":
        # Length per input = (2*M - 1), total = (2*M - 1)*n_inputs
        # Block order (vector is split into 3 equal blocks): storage | inflow | day
        # Within each block: [x1..x_{M-1}, theta1..thetaM]
        names = []
        block_labels = ["storage", "inflow", "day"][:n_pwl_inputs]
        for lab in block_labels:
            # breakpoints
            for k in range(1, n_segments):
                names.append(f"{lab}_x{k}")
            # angles
            for k in range(1, n_segments + 1):
                names.append(f"{lab}_theta{k}")
        return names

    raise ValueError(f"Unknown policy '{policy}'.")


def print_param_order(policy: str):
    """Pretty-print index → name mapping and a template --params string."""
    names = get_param_names(policy)
    print("\n=== PARAMETER ORDER =====================================================")
    print(f"Policy: {policy} | NVARS={len(names)}")
    for i, n in enumerate(names):
        print(f"[{i:02d}] {n}")
    # Build a template string (comma-separated) with placeholder zeros
    template_vals = ",".join("0" for _ in names)
    print("\n--params template (all zeros, edit values as needed):")
    print(template_vals)
    print("=========================================================================\n")
# ------------------------------
# Main
# ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reservoir", required=True,
                        help="one of: fewalter | beltzvilleCombined | prompton")
    parser.add_argument("--policy", required=True, choices=["STARFIT", "RBF", "PWL"])
    parser.add_argument("--params", default=None, help="Comma-separated list of policy params (var1,var2,...)")
    parser.add_argument("--params-csv", default=None, help="CSV with var1..varN columns (or first N columns)")
    parser.add_argument("--row-idx", type=int, default=0, help="Row index to read from --params-csv")
    parser.add_argument("--scaled-inflows", action="store_true", help="Use scaled inflows")
    parser.add_argument("--outdir", default=os.path.join(FIG_DIR, "simple_runs"), help="Output directory")
    parser.add_argument("--save-figs", action="store_true", help="Save PNG figures (dynamics)")
    parser.add_argument("--show-figs", action="store_true", help="Show interactive figures")
    parser.add_argument("--add-spill-to-release", action="store_true", default=True,
                        help="Add spill to release series for objective calc")
    parser.add_argument("--clip-to-bounds", action="store_true",
                        help="Clip provided params to policy bounds before running")
    parser.add_argument("--print-context", action="store_true",
                        help="Print one-line policy context (S_cap, I_min/max, R_min/max)")
    parser.add_argument("--print-violations", action="store_true",
                        help="Print a summary of constraint bindings/violations")
    parser.add_argument("--verify-mass-balance", action="store_true",
                        help="Run mass-balance checks and save residuals/plots")
    parser.add_argument("--verify-constraints", action="store_true",
                        help="Summarize constraint binding/violation fractions")
    parser.add_argument("--verify-optimization", action="store_true",
                        help="Re-run an optimization-like evaluation and compare objectives")
    parser.add_argument("--mb-tol-abs", type=float, default=1e-6, help="Abs tolerance for mass balance (MG)")
    parser.add_argument("--mb-tol-rel", type=float, default=1e-6, help="Rel tolerance vs typical daily volume")
    parser.add_argument("--print-param-order", action="store_true",
                        help="Print the exact --params order for the chosen --policy and exit")
    parser.add_argument("--start", default=None,
                        help="Start date (e.g., 2005-01-01). If omitted, uses first obs date.")
    parser.add_argument("--end", default=None,
                        help="End date (e.g., 2015-12-31). If omitted, uses last obs date.")


    args = parser.parse_args()

    RESERVOIR_NAME = str(args.reservoir)
    POLICY_TYPE = str(args.policy)

    if args.print_param_order:
        print_param_order(POLICY_TYPE)
        return 

    # Expected number of variables for the policy
    NVARS = policy_n_params[POLICY_TYPE]
    BOUNDS = policy_param_bounds[POLICY_TYPE]

    # Resolve parameters: CLI inline > CSV > baseline (STARFIT only)
    if args.params:
        policy_params = parse_params_inline(args.params, NVARS)
    elif args.params_csv:
        policy_params = load_params_from_csv(args.params_csv, NVARS, row_idx=args.row_idx)
    else:
        if POLICY_TYPE == "STARFIT":
            try:
                policy_params = BASELINE_STARFIT_PARAMS[RESERVOIR_NAME]
                if policy_params.shape[0] != NVARS:
                    raise ValueError(f"Baseline STARFIT params for {RESERVOIR_NAME} do not match NVARS={NVARS}.")
            except KeyError:
                raise KeyError(
                    "No baseline STARFIT params for this reservoir; "
                    "please pass --params or --params-csv."
                )
        else:
            raise ValueError("For policies other than STARFIT, please pass --params or --params-csv.")

    # Bounds check (warn or clip)
    lo = np.array([b[0] for b in BOUNDS], dtype=float)
    hi = np.array([b[1] for b in BOUNDS], dtype=float)
    if args.clip_to_bounds:
        policy_params = np.clip(policy_params, lo, hi)
    else:
        out_of_bounds = np.logical_or(policy_params < lo, policy_params > hi)
        if np.any(out_of_bounds):
            idxs = np.where(out_of_bounds)[0].tolist()
            print(f"[WARN] Provided params violate bounds at indices: {idxs}")

    # Ensure output dir
    outdir = ensure_dir(args.outdir)

    # Load observed data (optionally scaled)
    inflow_obs, release_obs, storage_obs = get_observational_training_data(
        reservoir_name=RESERVOIR_NAME,
        data_dir=PROCESSED_DATA_DIR,
        as_numpy=False,
        scaled_inflows=args.scaled_inflows,
    )
    dt_index = inflow_obs.index
    inflow_array  = inflow_obs.values.flatten().astype(np.float64)
    release_array = release_obs.values.flatten().astype(np.float64)
    storage_array = storage_obs.values.flatten().astype(np.float64)
    initial_storage = float(storage_array[0])

    # Build reservoir (context pulled internally from config)
    reservoir = Reservoir(
        inflow=inflow_array,
        dates=dt_index,
        capacity=reservoir_capacity[RESERVOIR_NAME],
        policy_type=POLICY_TYPE,
        policy_params=policy_params,
        initial_storage=initial_storage,
        name=RESERVOIR_NAME,
    )
    # keep step debug off for clean output
    reservoir.policy.debug = False

    # Optional one-line context print
    if args.print_context:
        try:
            ctx = reservoir.policy.get_context()
            print(f"[CTX] {RESERVOIR_NAME}: "
                  f"S_cap={ctx['storage_capacity']:.2f}, "
                  f"I∈[{ctx['I_min']:.2f},{ctx['I_max']:.2f}], "
                  f"R∈[{ctx['release_min']:.2f},{ctx['release_max']:.2f}]")
        except Exception as e:
            print(f"[CTX] (unavailable) {e}")

    # (Optional) STARFIT NOR constraint test to mimic optimization reject
    if POLICY_TYPE == "STARFIT":
        try:
            if not reservoir.policy.test_nor_constraint():
                print("[!] STARFIT NOR constraint violated for provided params.")
        except Exception as e:
            print(f"[!] NOR constraint check skipped: {e}")

    # Run
    reservoir.reset()
    reservoir.run()

    # Simulated time series
    sim_release = reservoir.release_array.astype(np.float64)
    if args.add_spill_to_release:
        sim_release = sim_release + reservoir.spill_array.astype(np.float64)
    sim_storage = reservoir.storage_array.astype(np.float64)

    # Objectives (same calculators as optimization)
    release_calc = ObjectiveCalculator(metrics=RELEASE_METRICS)
    storage_calc = ObjectiveCalculator(metrics=STORAGE_METRICS)
    release_objs = release_calc.calculate(obs=release_array, sim=sim_release)
    storage_objs = storage_calc.calculate(obs=storage_array, sim=sim_storage)

    # === VERIFICATIONS ===
    # Try to fetch context from policy (or reconstruct from config if your policy provides a get_context())
    try:
        ctx = reservoir.policy.get_context()
    except Exception:
        # reconstruct minimal context from what you passed in
        ctx = {
            "storage_capacity": float(reservoir_capacity[RESERVOIR_NAME]),
            "release_min": np.nan, "release_max": np.nan,
            "I_min": np.nan, "I_max": np.nan
        }

    if args.verify_mass_balance:
        verify_mass_balance(
            storage=sim_storage,
            inflow=inflow_array,
            release=reservoir.release_array,   # raw release (don’t double-count spill)
            spill=reservoir.spill_array,
            s_cap=float(ctx["storage_capacity"]),
            dt_index=dt_index,                 # Δt computed from this
            tol_abs=args.mb_tol_abs,
            tol_rel=args.mb_tol_rel,
            outdir=outdir,
            prefix=f"{RESERVOIR_NAME}_{POLICY_TYPE}_"
        )

    if args.verify_constraints:
        verify_constraints_parity(reservoir, ctx, tol=1e-6)

    if args.verify_optimization:
        objs_like_opt = evaluate_like_optimization(
            inflow_array, release_array, storage_array, dt_index,
            POLICY_TYPE, policy_params, {"storage_capacity": float(ctx["storage_capacity"])},
            RESERVOIR_NAME, 
            add_spill=args.add_spill_to_release
        )
        objs_here = np.r_[release_objs, storage_objs]
        diff = np.asarray(objs_here) - np.asarray(objs_like_opt)
        print("\n# Optimization parity check")
        print("  here:", np.array2string(objs_here, precision=6))
        print("  opt :", np.array2string(objs_like_opt, precision=6))
        print("  diff:", np.array2string(diff, precision=6))
        if not np.allclose(objs_here, objs_like_opt, rtol=1e-10, atol=1e-10):
            print("  [WARN] Objective parity mismatch with optimization-like eval.")
        else:
            print("  [OK] Objectives match optimization-like pipeline.")

    # Optional violation summary
    if args.print_violations and hasattr(reservoir.policy, "get_violation_summary"):
        summary = reservoir.policy.get_violation_summary()
        if any(summary.values()):
            print("[violations]", summary)

    # Print small report
    print("=" * 80)
    print(f"Reservoir: {RESERVOIR_NAME} | Policy: {POLICY_TYPE}")
    print(f"Params ({len(policy_params)}): {np.array2string(policy_params, precision=4, separator=', ')}")
    print("\n# Release Objectives:")
    for name, val in zip(RELEASE_METRICS, release_objs):
        print(f"  {name}: {val:.6f}")
    print("\n# Storage Objectives:")
    for name, val in zip(STORAGE_METRICS, storage_objs):
        print(f"  {name}: {val:.6f}")
    print("=" * 80)

    # Save summary CSV
    summary = {
        "reservoir": RESERVOIR_NAME,
        "policy": POLICY_TYPE,
        **{f"param{i+1}": v for i, v in enumerate(policy_params)},
        **{f"release_{m}": v for m, v in zip(RELEASE_METRICS, release_objs)},
        **{f"storage_{m}": v for m, v in zip(STORAGE_METRICS, storage_objs)},
    }
    summary_path = os.path.join(outdir, f"summary_{RESERVOIR_NAME}_{POLICY_TYPE}.csv")
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    print(f"[✓] Wrote summary: {summary_path}")

    # --- build pandas Series for easy slicing ---
    obs_R = pd.Series(release_array, index=dt_index, name="obs_release")
    sim_R = pd.Series(sim_release,   index=dt_index, name="sim_release")
    obs_S = pd.Series(storage_array, index=dt_index, name="obs_storage")
    sim_S = pd.Series(sim_storage,   index=dt_index, name="sim_storage")

    # --- optional windowing ---
    START = args.start or obs_R.index[0]
    END   = args.end   or obs_R.index[-1]

    obs_Rw = obs_R.loc[START:END]
    sim_Rw = sim_R.loc[START:END]
    obs_Sw = obs_S.loc[START:END]
    sim_Sw = sim_S.loc[START:END]

    # --- single 2x1 figure ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # STORAGE (top)
    axes[0].plot(obs_Sw.index, obs_Sw.values, label="Observed storage", linestyle="--", linewidth=2)
    axes[0].plot(sim_Sw.index, sim_Sw.values, label="Simulated storage", linewidth=1.6, alpha=0.9)
    axes[0].set_title(f"{RESERVOIR_NAME} — Storage ({pd.to_datetime(START).date()} to {pd.to_datetime(END).date()})")
    axes[0].set_ylabel("Storage (MG)")
    axes[0].grid(True, alpha=0.4)
    axes[0].legend(loc="upper left")

    # RELEASE (bottom)
    axes[1].plot(obs_Rw.index, obs_Rw.values, label="Observed release (gage)", linestyle="--", linewidth=2)
    axes[1].plot(sim_Rw.index, sim_Rw.values, label=f"Simulated release ({POLICY_TYPE})", linewidth=1.6, alpha=0.9)
    axes[1].set_title(f"{RESERVOIR_NAME} — Release ({pd.to_datetime(START).date()} to {pd.to_datetime(END).date()})")
    axes[1].set_ylabel("Release / Flow (MGD)")
    axes[1].set_xlabel("Date")
    axes[1].grid(True, alpha=0.4)
    axes[1].legend(loc="upper left")

    plt.tight_layout()

    combo_png = os.path.join(
        outdir,
        f"{RESERVOIR_NAME}_{POLICY_TYPE}_storage_release_{pd.to_datetime(START).date()}_{pd.to_datetime(END).date()}.png"
    )
    if args.save_figs:
        fig.savefig(combo_png, dpi=300)
        print(f"[✓] Saved: {combo_png}")
    if args.show_figs:
        plt.show()
    plt.close(fig)

    # (No policy-surface / NOR plots here to avoid side-effects.)
if __name__ == "__main__":
    main()