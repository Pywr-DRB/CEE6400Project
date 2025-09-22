# methods/plotting/plot_release_storage_9panel.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from methods.config import reservoir_capacity

def to_percent_storage(storage_series: pd.Series, reservoir: str) -> pd.Series:
    cap = float(reservoir_capacity[reservoir])
    return 100.0 * (pd.to_numeric(storage_series, errors="coerce") / cap)

def _fdc(series: pd.Series):
    s = pd.Series(series).dropna()
    if s.empty: return np.array([]), np.array([])
    vals = np.sort(s.values)[::-1]
    p = np.linspace(0, 100, len(vals))
    return p, vals

def plot_release_storage_9panel(
    reservoir: str,
    sim_release: pd.Series,              # DatetimeIndex
    obs_release: pd.Series | None,
    sim_storage_MG: pd.Series,           # DatetimeIndex, MG
    obs_storage_MG: pd.Series | None,
    start: str, end: str,
    ylabel="Flow (MGD)", storage_ylabel="Storage (% cap)",
    save_path: str | None = None,
):
    # slice + convert
    sim_r = sim_release.loc[start:end]
    sim_s_pct = to_percent_storage(sim_storage_MG.loc[start:end], reservoir)

    obs_r = obs_s_pct = None
    if obs_release is not None:   obs_r = obs_release.loc[start:end]
    if obs_storage_MG is not None: obs_s_pct = to_percent_storage(obs_storage_MG.loc[start:end], reservoir)

    # monthly/annual means
    m_sim_r = sim_r.resample("ME").mean();     y_sim_r = sim_r.resample("YE").mean()
    m_sim_s = sim_s_pct.resample("ME").mean(); y_sim_s = sim_s_pct.resample("YE").mean()
    m_obs_r = y_obs_r = m_obs_s = y_obs_s = None
    if obs_r is not None:
        m_obs_r = obs_r.resample("ME").mean(); y_obs_r = obs_r.resample("YE").mean()
    if obs_s_pct is not None:
        m_obs_s = obs_s_pct.resample("ME").mean(); y_obs_s = obs_s_pct.resample("YE").mean()

    fdc_d_sim = _fdc(sim_r); fdc_m_sim = _fdc(m_sim_r); fdc_y_sim = _fdc(y_sim_r)
    fdc_d_obs = fdc_m_obs = fdc_y_obs = (np.array([]), np.array([]))
    if obs_r is not None:
        fdc_d_obs = _fdc(obs_r); fdc_m_obs = _fdc(m_obs_r); fdc_y_obs = _fdc(y_obs_r)

    mnames = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    mg_sim_r = m_sim_r.groupby(m_sim_r.index.month).mean()
    mg_sim_s = m_sim_s.groupby(m_sim_s.index.month).mean()
    mg_obs_r = mg_obs_s = None
    if m_obs_r is not None: mg_obs_r = m_obs_r.groupby(m_obs_r.index.month).mean()
    if m_obs_s is not None: mg_obs_s = m_obs_s.groupby(m_obs_s.index.month).mean()

    fig, axs = plt.subplots(3, 3, figsize=(18, 12))
    # row 1: daily
    axs[0,0].plot(sim_s_pct.index, sim_s_pct.values, label="Sim", color="tab:green")
    if obs_s_pct is not None: axs[0,0].plot(obs_s_pct.index, obs_s_pct.values, label="Obs", color="black")
    axs[0,0].set_title(f"{reservoir}: Daily Storage (%)"); axs[0,0].set_ylabel(storage_ylabel); axs[0,0].grid(True); axs[0,0].legend()

    if obs_r is not None: axs[0,1].plot(obs_r.index, obs_r.values, label="Obs", color="black")
    axs[0,1].plot(sim_r.index, sim_r.values, label="Sim", ls="--", color="tab:blue")
    axs[0,1].set_title(f"{reservoir}: Daily Release"); axs[0,1].set_ylabel(ylabel); axs[0,1].grid(True); axs[0,1].legend()

    if len(fdc_d_obs[0]): axs[0,2].plot(*fdc_d_obs, label="Obs", color="black")
    if len(fdc_d_sim[0]): axs[0,2].plot(*fdc_d_sim, label="Sim", ls="--", color="tab:blue")
    axs[0,2].set_title(f"{reservoir}: Daily Release FDC"); axs[0,2].set_xlabel("Exceedance (%)"); axs[0,2].set_yscale("log"); axs[0,2].grid(True); axs[0,2].legend()

    # row 2: monthly means
    axs[1,0].plot(mnames, mg_sim_s.values, label="Sim", color="tab:green", marker="o")
    if mg_obs_s is not None: axs[1,0].plot(mnames, mg_obs_s.values, label="Obs", color="black", marker="o")
    axs[1,0].set_title(f"{reservoir}: Monthly Avg Storage"); axs[1,0].set_ylabel(storage_ylabel); axs[1,0].grid(True); axs[1,0].legend()

    if mg_obs_r is not None: axs[1,1].plot(mnames, mg_obs_r.values, label="Obs", color="black", marker="o")
    axs[1,1].plot(mnames, mg_sim_r.values, label="Sim", ls="--", color="tab:blue", marker="o")
    axs[1,1].set_title(f"{reservoir}: Monthly Avg Release"); axs[1,1].set_ylabel(ylabel); axs[1,1].grid(True); axs[1,1].legend()

    if len(fdc_m_obs[0]): axs[1,2].plot(*fdc_m_obs, label="Obs", color="black")
    if len(fdc_m_sim[0]): axs[1,2].plot(*fdc_m_sim, label="Sim", ls="--", color="tab:blue")
    axs[1,2].set_title(f"{reservoir}: Monthly Release FDC"); axs[1,2].set_xlabel("Exceedance (%)"); axs[1,2].set_yscale("log"); axs[1,2].grid(True); axs[1,2].legend()

    # row 3: annual means
    axs[2,0].plot(y_sim_s.index.year, y_sim_s.values, label="Sim", color="tab:green", marker="o")
    if y_obs_s is not None: axs[2,0].plot(y_obs_s.index.year, y_obs_s.values, label="Obs", color="black", marker="o")
    axs[2,0].set_title(f"{reservoir}: Annual Avg Storage"); axs[2,0].set_ylabel(storage_ylabel); axs[2,0].set_xlabel("Year"); axs[2,0].grid(True); axs[2,0].legend()

    if y_obs_r is not None: axs[2,1].plot(y_obs_r.index.year, y_obs_r.values, label="Obs", color="black", marker="o")
    axs[2,1].plot(y_sim_r.index.year, y_sim_r.values, label="Sim", ls="--", color="tab:blue", marker="o")
    axs[2,1].set_title(f"{reservoir}: Annual Avg Release"); axs[2,1].set_ylabel(ylabel); axs[2,1].set_xlabel("Year"); axs[2,1].grid(True); axs[2,1].legend()

    if len(fdc_y_obs[0]): axs[2,2].plot(*fdc_y_obs, label="Obs", color="black")
    if len(fdc_y_sim[0]): axs[2,2].plot(*fdc_y_sim, label="Sim", ls="--", color="tab:blue")
    axs[2,2].set_title(f"{reservoir}: Annual Release FDC"); axs[2,2].set_xlabel("Exceedance (%)"); axs[2,2].set_yscale("log"); axs[2,2].grid(True); axs[2,2].legend()

    fig.suptitle(f"{reservoir} — Storage & Release Diagnostics\n{start} to {end}", fontsize=16, weight="bold")
    fig.tight_layout(rect=[0,0,1,0.95])
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300); plt.close(fig)
    else:
        plt.show()
