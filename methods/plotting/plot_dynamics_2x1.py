#!/usr/bin/env python3
# methods/plotting/plot_dynamics_2x1.py

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ---------- helpers ----------
def _robust_limits(series_list, lo: float = 0.2, hi: float = 99.8, pad: float = 0.12):
    """Percentile-based limits with extra padding to avoid clipping."""
    vals = pd.concat([pd.Series(s, dtype=float) for s in series_list if s is not None], axis=0)
    vals = vals.replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return None
    ql, qh = np.nanpercentile(vals.values, [lo, hi])
    span = max(qh - ql, 1e-9)
    return (ql - pad * span, qh + pad * span)


def _positive_log_limits(series_list, lo: float = 0.2, hi: float = 99.8, pad: float = 0.12):
    """Percentile limits using only positive values; returns (lo, hi) or None."""
    vals = []
    for s in series_list:
        if s is None:
            continue
        v = (
            pd.Series(s, dtype=float)
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .to_numpy()
        )
        v = v[v > 0]
        if v.size:
            vals.append(v)
    if not vals:
        return None
    vals = np.concatenate(vals)
    ql, qh = np.nanpercentile(vals, [lo, hi])
    span = max(qh - ql, 1e-9)
    ql = max(ql * 0.8, 1e-6)  # keep lower bound above 0 on log scale
    return (ql, qh + pad * span)


def _mask_nonpositive(s: Optional[pd.Series]) -> Optional[pd.Series]:
    if s is None:
        return None
    out = s.astype(float).copy()
    out[out <= 0] = np.nan
    return out


def _apply_scale_then_limits(
    ax,
    values_concat: np.ndarray,
    scale: Optional[str],
    ylims: Optional[Tuple[float, float]],
    linthresh: float,
):
    """Set the y-scale first, then apply limits (so ticks are computed in that space)."""
    if scale in (None, "linear"):
        pass
    elif scale == "log":
        ax.set_yscale("log")
        ax.yaxis.set_minor_formatter(plt.NullFormatter())
    elif scale == "symlog":
        ax.set_yscale("symlog", linthresh=linthresh)
    else:
        raise ValueError("yscale must be None|'linear'|'log'|'symlog'")
    if ylims is not None:
        ax.set_ylim(*ylims)


def _downsample(s: Optional[pd.Series], step: Optional[int]) -> Optional[pd.Series]:
    if s is None or step is None or step <= 1:
        return s
    return s.iloc[::step]


# ---------- main ----------
def plot_2x1_dynamics(
    reservoir: str,
    policy: str,
    indie_R: pd.Series, indie_S: pd.Series,
    pywr_R: pd.Series,  pywr_S: pd.Series,
    def_R: pd.Series,   def_S: pd.Series,
    obs_R: Optional[pd.Series] = None,
    obs_S: Optional[pd.Series] = None,
    date_label: Optional[str] = None,
    ylims_storage: Optional[Tuple[float, float]] = None,
    ylims_release: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
    # display knobs
    yscale_storage: Optional[str] = None,      # None | 'linear' | 'log' | 'symlog'
    yscale_release: Optional[str] = None,      # None | 'linear' | 'log' | 'symlog'
    linthresh_release: float = 10.0,           # used only for symlog
    max_date_ticks: int = 10,
    downsample_step: Optional[int] = 2,        # light decimation for dense daily series
):
    """
    2×1 overlay:
      top  = Storage (MG)
      bot  = Release/Flow (MGD)

    Series: Observed (if provided), Independent, Pywr-DRB Parametric, Pywr-DRB Default.

    Readability:
      - slimmer linework (sims lw=0.35, obs lw=0.5)
      - relaxed auto-limits (0.2..99.8 pctile + 12% padding)
      - optional downsampling
      - concise date ticks
    """

    # aligned index across all series
    idx = indie_R.index
    for s in (pywr_R, def_R, obs_R, indie_S, pywr_S, def_S, obs_S):
        if s is not None:
            idx = idx.intersection(s.index)
    idx = idx.sort_values()
    if len(idx) == 0:
        print(f"[Plot] {reservoir}/{policy}: no overlap for 2x1 dynamics; skipping.")
        return

    # align
    R1, R2, R3 = indie_R.reindex(idx), pywr_R.reindex(idx), def_R.reindex(idx)
    S1, S2, S3 = indie_S.reindex(idx), pywr_S.reindex(idx), def_S.reindex(idx)
    Robs = obs_R.reindex(idx) if obs_R is not None else None
    Sobs = obs_S.reindex(idx) if obs_S is not None else None

    # log-aware prep (mask <=0, compute limits in the right space)
    if yscale_release == "log":
        ylims_release = ylims_release or _positive_log_limits([R1, R2, R3, Robs])
        R1p, R2p, R3p, Robsp = _mask_nonpositive(R1), _mask_nonpositive(R2), _mask_nonpositive(R3), _mask_nonpositive(Robs)
    else:
        R1p, R2p, R3p, Robsp = R1, R2, R3, Robs
        if ylims_release is None:
            ylims_release = _robust_limits([R1, R2, R3, Robs])

    if yscale_storage == "log":
        ylims_storage = ylims_storage or _positive_log_limits([S1, S2, S3, Sobs])
        S1p, S2p, S3p, Sobsp = _mask_nonpositive(S1), _mask_nonpositive(S2), _mask_nonpositive(S3), _mask_nonpositive(Sobs)
    else:
        S1p, S2p, S3p, Sobsp = S1, S2, S3, Sobs
        if ylims_storage is None:
            ylims_storage = _robust_limits([S1, S2, S3, Sobs])

    # optional downsampling
    R1p = _downsample(R1p, downsample_step); R2p = _downsample(R2p, downsample_step)
    R3p = _downsample(R3p, downsample_step); Robsp = _downsample(Robsp, downsample_step)
    S1p = _downsample(S1p, downsample_step); S2p = _downsample(S2p, downsample_step)
    S3p = _downsample(S3p, downsample_step); Sobsp = _downsample(Sobsp, downsample_step)

    # figure
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, constrained_layout=False)

    # ----- Storage (top) -----
    storage_values = pd.concat([x for x in [S1p, S2p, S3p, Sobsp] if x is not None]).values
    _apply_scale_then_limits(axes[0], storage_values, yscale_storage, ylims_storage, linthresh_release)

    if Sobsp is not None:
        axes[0].plot(Sobsp.index, Sobsp.values, label="Observed", lw=0.5, color="black", zorder=10)
    axes[0].plot(S1p.index, S1p.values, label="Independent", lw=0.5, alpha=0.95, zorder=6)
    axes[0].plot(S2p.index, S2p.values, label="Pywr-DRB Parametric", lw=0.5, ls=":", alpha=0.95, zorder=5)
    axes[0].plot(S3p.index, S3p.values, label="Pywr-DRB Default",    lw=0.5, ls="--", alpha=0.95, zorder=4)
    axes[0].set_ylabel("Storage (MG)")
    axes[0].grid(True, alpha=0.3)
    axes[0].margins(y=0.04)

    # ----- Release (bottom) -----
    release_values = pd.concat([x for x in [R1p, R2p, R3p, Robsp] if x is not None]).values
    _apply_scale_then_limits(axes[1], release_values, yscale_release, ylims_release, linthresh_release)

    if Robsp is not None:
        axes[1].plot(Robsp.index, Robsp.values, label="Observed", lw=0.5, color="black", zorder=10)
    axes[1].plot(R1p.index, R1p.values, label="Independent (release+spill)", lw=0.5, alpha=0.95, zorder=6)
    axes[1].plot(R2p.index, R2p.values, label="Pywr-DRB Parametric",         lw=0.5, ls=":", alpha=0.95, zorder=5)
    axes[1].plot(R3p.index, R3p.values, label="Pywr-DRB Default",            lw=0.5, ls="--", alpha=0.95, zorder=4)
    axes[1].set_ylabel("Release (MGD)")
    axes[1].set_xlabel("Date")
    axes[1].grid(True, alpha=0.3)
    axes[1].margins(y=0.04)

    # Date ticks
    locator = mdates.AutoDateLocator(minticks=5, maxticks=max_date_ticks)
    formatter = mdates.ConciseDateFormatter(locator)
    for ax in axes:
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

    # Title + legend outside
    header = f"{reservoir} — {policy}"
    if date_label:
        header += f"   [{date_label}]"
    fig.suptitle(header, fontsize=14, fontweight="bold")

    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), framealpha=0.9)

    # layout space for legend & title
    fig.tight_layout(rect=[0, 0, 0.82, 0.92])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()
