#!/usr/bin/env python3
# methods/plotting/plot_sixpanel_timeseries_exceedance_fdc.py

from __future__ import annotations
import os
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

# ---- styles (match your other figures) ---------------------------------------
SIM_STYLES = {
    "Independent":     dict(color="#1f77b4", lw=0.5, ls="-",  alpha=0.95, zorder=6, antialiased=True),
    "Pywr Parametric": dict(color="#2ca02c", lw=0.5, ls=":",  alpha=0.95, zorder=5, antialiased=True),
    "Pywr Default":    dict(color="#d62728", lw=0.5, ls="--", alpha=0.95, zorder=4, antialiased=True),
}
OBS_STYLE = dict(color="black", lw=0.5, ls="-", alpha=1.0, zorder=10)

# ---- helpers -----------------------------------------------------------------
def _pct_rank(s: pd.Series) -> pd.Series:
    """Return percentile rank (0..100) for a series (NaN-safe)."""
    s = pd.to_numeric(s, errors="coerce")
    r = s.rank(pct=True) * 100.0
    # keep original index/NaNs
    r.loc[s.isna()] = np.nan
    return r

def _fdc(series: pd.Series, positive_only: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Return (exceedance %, values) for Flow Duration Curve."""
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if positive_only:
        s = s[s > 0]
    if s.size == 0:
        return np.array([]), np.array([])
    vals = np.sort(s)[::-1]
    p = np.linspace(0, 100, len(vals))
    return p, vals

def _to_percent_storage(storage_MG: pd.Series, capacity_MG: float) -> pd.Series:
    s = pd.to_numeric(storage_MG, errors="coerce")
    if capacity_MG <= 0:
        return s * np.nan
    return 100.0 * (s / float(capacity_MG))

def _downsample(s: Optional[pd.Series], step: Optional[int]) -> Optional[pd.Series]:
    if s is None or step is None or step <= 1:
        return s
    return s.iloc[::step]

def _align_index(*series: Optional[pd.Series]) -> pd.DatetimeIndex:
    idx = None
    for s in series:
        if s is None:
            continue
        idx = s.index if idx is None else idx.intersection(s.index)
    return idx.sort_values() if idx is not None else pd.DatetimeIndex([])

def _ax_identity_line(ax, xmin=0, xmax=100, **kw):
    ax.plot([xmin, xmax], [xmin, xmax], color="0.4", lw=1.0, ls="--", zorder=2, **kw)

# ---- main --------------------------------------------------------------------
def plot_sixpanel_timeseries_exceedance_fdc(
    reservoir: str,
    capacity_MG: float,
    # time series (DatetimeIndex)
    indie_R: pd.Series, indie_S: pd.Series,
    pywr_R:  pd.Series, pywr_S:  pd.Series,
    def_R:   pd.Series, def_S:   pd.Series,
    obs_R: Optional[pd.Series] = None,
    obs_S: Optional[pd.Series] = None,
    # labels / options
    date_label: Optional[str] = None,
    save_path: Optional[str] = None,
    downsample_step: Optional[int] = 2,
    max_date_ticks: int = 10,
):
    """
    6-panel diagnostic:
      Row 1: [Storage time series %cap] [Release time series MGD] [Storage 1:1 exceedance]
      Row 2: [Release 1:1 exceedance]   [Storage FDC (%cap)]      [Release FDC (MGD, log y)]
    Overlays Obs + (Independent, Pywr Parametric, Pywr Default).
    """
    # align + convert storage to % cap
    idx = _align_index(indie_R, pywr_R, def_R, obs_R, indie_S, pywr_S, def_S, obs_S)
    if len(idx) == 0:
        print(f"[6-panel] {reservoir}: no overlapping index; skipping.")
        return

    # reindex
    R_ind = indie_R.reindex(idx)
    R_par = pywr_R.reindex(idx)
    R_def = def_R.reindex(idx)
    R_obs = obs_R.reindex(idx) if obs_R is not None else None

    S_ind = _to_percent_storage(indie_S.reindex(idx), capacity_MG)
    S_par = _to_percent_storage(pywr_S.reindex(idx), capacity_MG)
    S_def = _to_percent_storage(def_S.reindex(idx), capacity_MG)
    S_obs = _to_percent_storage(obs_S.reindex(idx), capacity_MG) if obs_S is not None else None

    # optional downsampling for time series panels
    R_ind_d = _downsample(R_ind, downsample_step)
    R_par_d = _downsample(R_par, downsample_step)
    R_def_d = _downsample(R_def, downsample_step)
    R_obs_d = _downsample(R_obs, downsample_step) if R_obs is not None else None

    S_ind_d = _downsample(S_ind, downsample_step)
    S_par_d = _downsample(S_par, downsample_step)
    S_def_d = _downsample(S_def, downsample_step)
    S_obs_d = _downsample(S_obs, downsample_step) if S_obs is not None else None

    # figure
    fig, axs = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=False)
    axs = axs.ravel()

    # ---- (1) Storage time series (% cap)
    ax = axs[0]
    if S_obs_d is not None: ax.plot(S_obs_d.index, S_obs_d.values, label="Observed", **OBS_STYLE)
    ax.plot(S_ind_d.index, S_ind_d.values, label="Independent", **SIM_STYLES["Independent"])
    ax.plot(S_par_d.index, S_par_d.values, label="Pywr Parametric", **SIM_STYLES["Pywr Parametric"])
    ax.plot(S_def_d.index, S_def_d.values, label="Pywr Default", **SIM_STYLES["Pywr Default"])
    ax.set_title(f"{reservoir}: Storage (% cap)")
    ax.set_ylabel("Storage (% of capacity)")
    ax.grid(True, alpha=0.3); ax.margins(y=0.06)

    # ---- (2) Release time series (MGD)
    ax = axs[1]
    if R_obs_d is not None: ax.plot(R_obs_d.index, R_obs_d.values, label="Observed", **OBS_STYLE)
    ax.plot(R_ind_d.index, R_ind_d.values, label="Independent (release+spill)", **SIM_STYLES["Independent"])
    ax.plot(R_par_d.index, R_par_d.values, label="Pywr Parametric", **SIM_STYLES["Pywr Parametric"])
    ax.plot(R_def_d.index, R_def_d.values, label="Pywr Default", **SIM_STYLES["Pywr Default"])
    ax.set_title(f"{reservoir}: Release (MGD)")
    ax.set_ylabel("Release (MGD)")
    ax.grid(True, alpha=0.3); ax.margins(y=0.06)

    # Date ticks for the two time-series panels
    locator = mdates.AutoDateLocator(minticks=5, maxticks=max_date_ticks)
    formatter = mdates.ConciseDateFormatter(locator)
    for ax in (axs[0], axs[1]):
        ax.xaxis.set_major_locator(locator); ax.xaxis.set_major_formatter(formatter)

    # ---- (3) Storage 1:1 exceedance (percentiles)
    # Build exceedance (% rank) *against the same reference period*
    ax = axs[2]
    if S_obs is not None and not S_obs.dropna().empty:
        # Obs reference ranking
        Sref = _pct_rank(S_obs)
        # Align sims to obs index for pairwise comparisons
        pairs = {
            "Independent": (Sref, _pct_rank(S_ind.reindex(Sref.index))),
            "Pywr Parametric": (Sref, _pct_rank(S_par.reindex(Sref.index))),
            "Pywr Default": (Sref, _pct_rank(S_def.reindex(Sref.index))),
        }
        for name, (p_obs, p_sim) in pairs.items():
            ok = p_obs.notna() & p_sim.notna()
            ax.scatter(p_obs[ok], p_sim[ok], s=5, alpha=0.25, c=SIM_STYLES[name]["color"], label=name, zorder=4)
        _ax_identity_line(ax)
        ax.set_xlim(0, 100); ax.set_ylim(0, 100)
        ax.set_xlabel("Obs exceedance percentile"); ax.set_ylabel("Sim exceedance percentile")
        ax.set_title(f"{reservoir}: Storage Exceedance 1:1")
        ax.grid(True, alpha=0.3)
    else:
        ax.set_title(f"{reservoir}: Storage Exceedance 1:1 (no obs)")
        ax.axis("off")

    # ---- (4) Release 1:1 exceedance (percentiles)
    ax = axs[3]
    if R_obs is not None and not R_obs.dropna().empty:
        Rref = _pct_rank(R_obs)
        pairs = {
            "Independent": (Rref, _pct_rank(R_ind.reindex(Rref.index))),
            "Pywr Parametric": (Rref, _pct_rank(R_par.reindex(Rref.index))),
            "Pywr Default": (Rref, _pct_rank(R_def.reindex(Rref.index))),
        }
        for name, (p_obs, p_sim) in pairs.items():
            ok = p_obs.notna() & p_sim.notna()
            ax.scatter(p_obs[ok], p_sim[ok], s=5, alpha=0.25, c=SIM_STYLES[name]["color"], label=name, zorder=4)
        _ax_identity_line(ax)
        ax.set_xlim(0, 100); ax.set_ylim(0, 100)
        ax.set_xlabel("Obs exceedance percentile"); ax.set_ylabel("Sim exceedance percentile")
        ax.set_title(f"{reservoir}: Release Exceedance 1:1")
        ax.grid(True, alpha=0.3)
    else:
        ax.set_title(f"{reservoir}: Release Exceedance 1:1 (no obs)")
        ax.axis("off")

    # ---- (5) Storage FDC (% cap; linear y)
    ax = axs[4]
    # Use distributions over the whole aligned window
    p_obs_s, v_obs_s = _fdc(S_obs, positive_only=False) if S_obs is not None else (np.array([]), np.array([]))
    p_ind_s, v_ind_s = _fdc(S_ind, positive_only=False)
    p_par_s, v_par_s = _fdc(S_par, positive_only=False)
    p_def_s, v_def_s = _fdc(S_def, positive_only=False)
    if v_obs_s.size: ax.plot(p_obs_s, v_obs_s, label="Observed", **OBS_STYLE)
    ax.plot(p_ind_s, v_ind_s, label="Independent", **SIM_STYLES["Independent"])
    ax.plot(p_par_s, v_par_s, label="Pywr Parametric", **SIM_STYLES["Pywr Parametric"])
    ax.plot(p_def_s, v_def_s, label="Pywr Default", **SIM_STYLES["Pywr Default"])
    ax.set_title(f"{reservoir}: Storage FDC (% cap)")
    ax.set_xlabel("Exceedance (%)"); ax.set_ylabel("Storage (% cap)")
    ax.grid(True, alpha=0.3); ax.margins(y=0.06)

    # ---- (6) Release FDC (MGD; log y)
    ax = axs[5]
    p_obs_r, v_obs_r = _fdc(R_obs, positive_only=True) if R_obs is not None else (np.array([]), np.array([]))
    p_ind_r, v_ind_r = _fdc(R_ind, positive_only=True)
    p_par_r, v_par_r = _fdc(R_par, positive_only=True)
    p_def_r, v_def_r = _fdc(R_def, positive_only=True)
    if v_obs_r.size: ax.plot(p_obs_r, v_obs_r, label="Observed", **OBS_STYLE)
    ax.plot(p_ind_r, v_ind_r, label="Independent", **SIM_STYLES["Independent"])
    ax.plot(p_par_r, v_par_r, label="Pywr Parametric", **SIM_STYLES["Pywr Parametric"])
    ax.plot(p_def_r, v_def_r, label="Pywr Default", **SIM_STYLES["Pywr Default"])
    ax.set_yscale("log")
    ax.set_title(f"{reservoir}: Release FDC (log y)")
    ax.set_xlabel("Exceedance (%)"); ax.set_ylabel("Release (MGD)")
    ax.grid(True, which="both", alpha=0.3); ax.margins(y=0.06)

    # global title + legends
    title = f"{reservoir} — Time Series, Exceedance, and FDCs"
    if date_label:
        title += f"   [{date_label}]"
    fig.suptitle(title, fontsize=15, weight="bold")

    # build consistent proxy handles so legend looks the same regardless of panel types
    proxy_handles = []
    proxy_labels  = []

    # Observed first (only if present anywhere)
    has_obs = (
        (R_obs is not None and not pd.Series(R_obs).dropna().empty) or
        (S_obs is not None and not pd.Series(S_obs).dropna().empty) or
        (len(p_obs_s) > 0) or (len(p_obs_r) > 0)
    )
    if has_obs:
        proxy_handles.append(Line2D([0], [0], **{**OBS_STYLE, "marker": None}))
        proxy_labels.append("Observed")

    # The three simulation series (always shown)
    for name in ["Independent", "Pywr Parametric", "Pywr Default"]:
        style = SIM_STYLES[name].copy()
        style["marker"] = None  # keep legend tidy as lines
        proxy_handles.append(Line2D([0], [0], **style))
        proxy_labels.append(name)

    # remove any per-axes legends so we only display the shared one
    for ax in axs:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    # figure-level legend on the right
    fig.legend(
        proxy_handles, proxy_labels,
        loc="center left",
        bbox_to_anchor=(0.985, 0.5),
        framealpha=0.92,
        title="Series"
    )

    # leave margin for legend (right) and suptitle (top)
    fig.tight_layout(rect=[0, 0, 0.96, 0.93])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()
