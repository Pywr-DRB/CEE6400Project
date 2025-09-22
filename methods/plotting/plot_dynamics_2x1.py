# methods/plotting/plot_dynamics_2x1.py
import os
import matplotlib.pyplot as plt
import pandas as pd

def plot_2x1_dynamics(
    reservoir: str,
    policy: str,
    indie_R: pd.Series, indie_S: pd.Series,
    pywr_R: pd.Series,  pywr_S: pd.Series,
    def_R: pd.Series,   def_S: pd.Series,
    save_path: str | None = None,
):
    """Overlay storage (top) and release (bottom) time series for
    independent run, parametric Pywr-DRB, and default Pywr-DRB."""
    idx = indie_R.index.intersection(pywr_R.index).intersection(def_R.index).sort_values()
    if len(idx) == 0:
        print(f"[Plot] {reservoir}/{policy}: no overlap for 2x1 dynamics; skipping.")
        return

    R1, R2, R3 = indie_R.reindex(idx), pywr_R.reindex(idx), def_R.reindex(idx)
    S1, S2, S3 = indie_S.reindex(idx), pywr_S.reindex(idx), def_S.reindex(idx)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Storage
    axes[0].plot(S1.index, S1.values, label="Independent", lw=1.6)
    axes[0].plot(S2.index, S2.values, label="Pywr-DRB Parametric", lw=1.6, ls=":")
    axes[0].plot(S3.index, S3.values, label="Pywr-DRB Default", lw=1.6, ls="--")
    axes[0].set_title(f"{reservoir} — Storage"); axes[0].set_ylabel("MG")
    axes[0].grid(True, alpha=0.3); axes[0].legend()

    # Release
    axes[1].plot(R1.index, R1.values, label="Independent (release+spill)", lw=1.6)
    axes[1].plot(R2.index, R2.values, label="Pywr-DRB Parametric", lw=1.6, ls=":")
    axes[1].plot(R3.index, R3.values, label="Pywr-DRB Default", lw=1.6, ls="--")
    axes[1].set_title(f"{reservoir} — Release / Flow"); axes[1].set_ylabel("MGD"); axes[1].set_xlabel("Date")
    axes[1].grid(True, alpha=0.3); axes[1].legend()

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()
