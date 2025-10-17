#!/usr/bin/env python3
# methods/plotting/plot_release_storage_9panel.py

from __future__ import annotations
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from methods.config import reservoir_capacity


# ---------- helpers ----------
def to_percent_storage(storage_series: pd.Series, reservoir: str) -> pd.Series:
    """Convert MG storage to % of capacity for the given reservoir."""
    cap = float(reservoir_capacity[reservoir])
    s = pd.to_numeric(storage_series, errors="coerce")
    return 100.0 * (s / cap)


def _fdc(series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Return exceedance (%) and sorted values (desc) for an FDC. Drop NaN/inf; ignore <=0 for log axes."""
    s = pd.Series(series, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return np.array([]), np.array([])
    vals = np.sort(s.values)[::-1]
    p = np.linspace(0, 100, len(vals))
    return p, vals


def _safe_monthly_means(s: pd.Series) -> pd.Series:
    """Monthly means with calendar month index 1..12 (handles missing months)."""
    if s is None or s.empty:
        return pd.Series(index=range(1, 13), dtype=float)
    m = s.resample("ME").mean()
    mg = m.groupby(m.index.month).mean()
    # ensure all 12 months show, even if NaN
    out = pd.Series(index=range(1, 13), dtype=float)
    out.loc[mg.index] = mg.values
    return out


def _safe_annual_means(s: pd.Series) -> pd.Series:
    if s is None or s.empty:
        return pd.Series(dtype=float)
    y = s.resample("YE").mean()
    # make the x-axis integer years
    y.index = y.index.year
    return y


def _fdc_positive(series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """FDC using only positive values (friendlier for log y)."""
    p, v = _fdc(series)
    keep = v > 0
    return p[keep], v[keep]


# ---------- main ----------
def plot_release_storage_9panel(
    reservoir: str,
    sim_release: pd.Series,              # DatetimeIndex, MGD
    obs_release: pd.Series | None,       # DatetimeIndex, MGD (optional)
    sim_storage_MG: pd.Series,           # DatetimeIndex, MG
    obs_storage_MG: pd.Series | None,    # DatetimeIndex, MG (optional)
    start: str, end: str,
    ylabel: str = "Flow (MGD)",
    storage_ylabel: str = "Storage (% cap)",
    save_path: str | None = None,
):
    """
    3×3 diagnostic: rows = {daily, monthly mean, annual mean};
    cols = {storage, release, FDC (release)}. Storage panels use % capacity.

    Readability tweaks:
      - slim lines (sims lw=0.35, obs lw=0.5), small markers (ms=3)
      - relaxed limits via default Matplotlib + margins to avoid clipping
      - FDC uses log y (positive-only)
      - legends outside, consistent titles, clean layout
    """
    # slice + convert
    sim_r = sim_release.loc[start:end]
    sim_s_pct = to_percent_storage(sim_storage_MG.loc[start:end], reservoir)

    obs_r = obs_s_pct = None
    if obs_release is not None:
        obs_r = obs_release.loc[start:end]
    if obs_storage_MG is not None:
        obs_s_pct = to_percent_storage(obs_storage_MG.loc[start:end], reservoir)

    # monthly & annual aggregates
    m_sim_r = _safe_monthly_means(sim_r)
    m_sim_s = _safe_monthly_means(sim_s_pct)
    m_obs_r = _safe_monthly_means(obs_r) if obs_r is not None else pd.Series(index=range(1, 13), dtype=float)
    m_obs_s = _safe_monthly_means(obs_s_pct) if obs_s_pct is not None else pd.Series(index=range(1, 13), dtype=float)

    y_sim_r = _safe_annual_means(sim_r)
    y_sim_s = _safe_annual_means(sim_s_pct)
    y_obs_r = _safe_annual_means(obs_r) if obs_r is not None else pd.Series(dtype=float)
    y_obs_s = _safe_annual_means(obs_s_pct) if obs_s_pct is not None else pd.Series(dtype=float)

    # FDCs (daily / monthly mean / annual mean releases)
    fdc_d_sim = _fdc_positive(sim_r)
    fdc_m_sim = _fdc_positive(m_sim_r)
    fdc_y_sim = _fdc_positive(y_sim_r)
    fdc_d_obs = fdc_m_obs = fdc_y_obs = (np.array([]), np.array([]))
    if obs_r is not None:
        fdc_d_obs = _fdc_positive(obs_r)
        fdc_m_obs = _fdc_positive(m_obs_r)
        fdc_y_obs = _fdc_positive(y_obs_r)

    # plotting
    month_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

    fig, axs = plt.subplots(3, 3, figsize=(18, 12))
    axs = axs  # 3x3 array

    # === Row 1: Daily ===
    # Storage (%)
    ax = axs[0, 0]
    ax.plot(sim_s_pct.index, sim_s_pct.values, label="Sim", color="tab:green", lw=0.5)
    if obs_s_pct is not None:
        ax.plot(obs_s_pct.index, obs_s_pct.values, label="Obs", color="black", lw=0.5)
    ax.set_title(f"{reservoir}: Daily Storage (%)")
    ax.set_ylabel(storage_ylabel)
    ax.grid(True, alpha=0.3)
    ax.margins(y=0.06)

    # Release (MGD)
    ax = axs[0, 1]
    if obs_r is not None:
        ax.plot(obs_r.index, obs_r.values, label="Obs", color="black", lw=0.5)
    ax.plot(sim_r.index, sim_r.values, label="Sim", ls="--", color="tab:blue", lw=0.5)
    ax.set_title(f"{reservoir}: Daily Release")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.margins(y=0.06)

    # FDC (daily release)
    ax = axs[0, 2]
    if len(fdc_d_obs[0]):
        ax.plot(*fdc_d_obs, label="Obs", color="black", lw=0.5)
    if len(fdc_d_sim[0]):
        ax.plot(*fdc_d_sim, label="Sim", ls="--", color="tab:blue", lw=0.5)
    ax.set_title(f"{reservoir}: Daily Release FDC")
    ax.set_xlabel("Exceedance (%)")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.margins(y=0.06)

    # === Row 2: Monthly means ===
    # Storage (%)
    ax = axs[1, 0]
    ax.plot(month_labels, m_sim_s.values, label="Sim", color="tab:green", marker="o", lw=0.5, ms=3)
    if not m_obs_s.isna().all():
        ax.plot(month_labels, m_obs_s.values, label="Obs", color="black", marker="o", lw=0.5, ms=3)
    ax.set_title(f"{reservoir}: Monthly Avg Storage")
    ax.set_ylabel(storage_ylabel)
    ax.grid(True, alpha=0.3)
    ax.margins(y=0.06)

    # Release (MGD)
    ax = axs[1, 1]
    if not m_obs_r.isna().all():
        ax.plot(month_labels, m_obs_r.values, label="Obs", color="black", marker="o", lw=0.5, ms=3)
    ax.plot(month_labels, m_sim_r.values, label="Sim", ls="--", color="tab:blue", marker="o", lw=0.5, ms=3)
    ax.set_title(f"{reservoir}: Monthly Avg Release")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.margins(y=0.06)

    # FDC (monthly mean release)
    ax = axs[1, 2]
    if len(fdc_m_obs[0]):
        ax.plot(*fdc_m_obs, label="Obs", color="black", lw=0.5)
    if len(fdc_m_sim[0]):
        ax.plot(*fdc_m_sim, label="Sim", ls="--", color="tab:blue", lw=0.5)
    ax.set_title(f"{reservoir}: Monthly Release FDC")
    ax.set_xlabel("Exceedance (%)")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.margins(y=0.06)

    # === Row 3: Annual means ===
    # Storage (%)
    ax = axs[2, 0]
    ax.plot(y_sim_s.index, y_sim_s.values, label="Sim", color="tab:green", marker="o", lw=0.5, ms=3)
    if not y_obs_s.empty:
        ax.plot(y_obs_s.index, y_obs_s.values, label="Obs", color="black", marker="o", lw=0.5, ms=3)
    ax.set_title(f"{reservoir}: Annual Avg Storage")
    ax.set_xlabel("Year")
    ax.set_ylabel(storage_ylabel)
    ax.grid(True, alpha=0.3)
    ax.margins(y=0.06)

    # Release (MGD)
    ax = axs[2, 1]
    if not y_obs_r.empty:
        ax.plot(y_obs_r.index, y_obs_r.values, label="Obs", color="black", marker="o", lw=0.5, ms=3)
    ax.plot(y_sim_r.index, y_sim_r.values, label="Sim", ls="--", color="tab:blue", marker="o", lw=0.5, ms=3)
    ax.set_title(f"{reservoir}: Annual Avg Release")
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.margins(y=0.06)

    # FDC (annual mean release)
    ax = axs[2, 2]
    if len(fdc_y_obs[0]):
        ax.plot(*fdc_y_obs, label="Obs", color="black", lw=0.5)
    if len(fdc_y_sim[0]):
        ax.plot(*fdc_y_sim, label="Sim", ls="--", color="tab:blue", lw=0.5)
    ax.set_title(f"{reservoir}: Annual Release FDC")
    ax.set_xlabel("Exceedance (%)")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.margins(y=0.06)

    # global title
    fig = plt.gcf()
    fig.suptitle(f"{reservoir} — Storage & Release Diagnostics\n{start} to {end}",
                 fontsize=16, weight="bold")

    # --- shared legend on the right (collect unique labels across panels) ---
    handles, labels = [], []
    for ax in axs.ravel():
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll and ll not in labels:
                handles.append(hh)
                labels.append(ll)
        # remove any per-axes legend so we only show the shared one
        if ax.get_legend() is not None:
            ax.legend_.remove()

    # add figure-level legend
    fig.legend(
        handles, labels,
        loc="center left",
        bbox_to_anchor=(0.985, 0.5),  # nudge further right if needed
        framealpha=0.92,
        title="Series"
    )

    # leave right margin for the legend and a bit up top for the suptitle
    fig.tight_layout(rect=[0, 0, 0.96, 0.90])

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()
