import os
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, Tuple

def plot_2x1_dynamics(
    reservoir: str,
    policy: str,
    indie_R: pd.Series, indie_S: pd.Series,
    pywr_R: pd.Series,  pywr_S: pd.Series,
    def_R: pd.Series,   def_S: pd.Series,
    obs_R: Optional[pd.Series] = None,
    obs_S: Optional[pd.Series] = None,
    date_label: Optional[str] = None,
    ylims_storage: Optional[Tuple[float,float]] = None,
    ylims_release: Optional[Tuple[float,float]] = None,
    save_path: str | None = None,
):
    """Overlay storage (top) and release (bottom) for
    Independent, Pywr-DRB (Parametric), Pywr-DRB (Default), and Observations (if provided)."""

    # aligned index
    idx = indie_R.index
    for s in (pywr_R, def_R, obs_R, indie_S, pywr_S, def_S, obs_S):
        if s is not None:
            idx = idx.intersection(s.index)
    idx = idx.sort_values()
    if len(idx) == 0:
        print(f"[Plot] {reservoir}/{policy}: no overlap for 2x1 dynamics; skipping.")
        return

    R1, R2, R3 = indie_R.reindex(idx), pywr_R.reindex(idx), def_R.reindex(idx)
    S1, S2, S3 = indie_S.reindex(idx), pywr_S.reindex(idx), def_S.reindex(idx)
    Robs = obs_R.reindex(idx) if obs_R is not None else None
    Sobs = obs_S.reindex(idx) if obs_S is not None else None

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, constrained_layout=False)

    # Storage (MG)
    if Sobs is not None:
        axes[0].plot(Sobs.index, Sobs.values, label="Observed", lw=1.6, color="black")
    axes[0].plot(S1.index, S1.values, label="Independent", lw=1.6)
    axes[0].plot(S2.index, S2.values, label="Pywr-DRB Parametric", lw=1.6, ls=":")
    axes[0].plot(S3.index, S3.values, label="Pywr-DRB Default", lw=1.6, ls="--")
    if ylims_storage is not None:
        axes[0].set_ylim(*ylims_storage)
    axes[0].set_ylabel("Storage (MG)")
    axes[0].grid(True, alpha=0.3)

    # Release / Flow (MGD)
    if Robs is not None:
        axes[1].plot(Robs.index, Robs.values, label="Observed", lw=1.6, color="black")
    axes[1].plot(R1.index, R1.values, label="Independent (release+spill)", lw=1.6)
    axes[1].plot(R2.index, R2.values, label="Pywr-DRB Parametric", lw=1.6, ls=":")
    axes[1].plot(R3.index, R3.values, label="Pywr-DRB Default", lw=1.6, ls="--")
    if ylims_release is not None:
        axes[1].set_ylim(*ylims_release)
    axes[1].set_ylabel("Release (MGD)")
    axes[1].set_xlabel("Date")
    axes[1].grid(True, alpha=0.3)

    # Titles + legend outside to avoid overlap/cutoff
    header = f"{reservoir} — {policy}"
    if date_label:
        header += f"   [{date_label}]"
    fig.suptitle(header, fontsize=14, fontweight="bold")

    # One combined legend on the bottom axis, outside right
    handles, labels = axes[1].get_legend_handles_labels()
    if Sobs is not None or Robs is not None:
        # ensure Observed appears first
        # (matplotlib already keeps order of plotting; okay as-is)
        pass
    axes[1].legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), framealpha=0.9)

    # keep room for suptitle and side legend
    fig.tight_layout(rect=[0, 0, 0.86, 0.92])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()
