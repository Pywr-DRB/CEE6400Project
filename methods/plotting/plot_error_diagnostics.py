#!/usr/bin/env python3
# methods/plotting/plot_error_diagnostics.py

import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess as _lowess
    _HAS_LOWESS = True
except Exception:
    _HAS_LOWESS = False

SEASON_ORDER  = ["Winter","Spring","Summer","Fall"]
DECADE_ORDER  = ["1980s","1990s","2000s","2010s","2020s"]
DECADE_COLORS = {"1980s":"#1f77b4","1990s":"#ff7f0e","2000s":"#2ca02c","2010s":"#d62728","2020s":"#9467bd"}
SEASON_COLORS = {"Winter":"#1f77b4","Spring":"#2ca02c","Summer":"#ff7f0e","Fall":"#9467bd"}
SEASON_MAP = {12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",
              6:"Summer",7:"Summer",8:"Summer",9:"Fall",10:"Fall",11:"Fall"}

# high-contrast but slim styles for residual time series
SIM_STYLES: Dict[str, Dict] = {
    "Independent":     dict(color="#1f77b4", lw=0.5, ls="-",  alpha=0.95, zorder=6, antialiased=True),
    "Pywr Parametric": dict(color="#2ca02c", lw=0.5, ls=":",  alpha=0.95, zorder=5, antialiased=True),
    "Pywr Default":    dict(color="#d62728", lw=0.5, ls="--", alpha=0.95, zorder=4, antialiased=True),
}

def _season(m): return SEASON_MAP.get(int(m), "NA")
def _decade_label(y): d = (int(y)//10)*10; return f"{d}s"
def _safe_label(s: str) -> str: return str(s).replace(" ", "_")

def _nse(obs: np.ndarray, sim: np.ndarray) -> float:
    obs = np.asarray(obs, float); sim = np.asarray(sim, float)
    m = np.nanmean(obs); denom = np.nansum((obs - m)**2)
    if denom == 0 or np.isnan(denom): return np.nan
    return 1.0 - (np.nansum((sim - obs)**2) / denom)

def _kge(obs: np.ndarray, sim: np.ndarray) -> float:
    obs = np.asarray(obs, float); sim = np.asarray(sim, float)
    ok = np.isfinite(obs) & np.isfinite(sim)
    if ok.sum() < 2: return np.nan
    r = np.corrcoef(obs[ok], sim[ok])[0,1]
    sstd, ostd = np.nanstd(sim), np.nanstd(obs)
    mu_s, mu_o = np.nanmean(sim), np.nanmean(obs)
    alpha = (sstd / ostd) if ostd > 0 else np.nan
    beta  = (mu_s / mu_o) if mu_o != 0 else np.nan
    return 1.0 - np.sqrt((r-1.0)**2 + (alpha-1.0)**2 + (beta-1.0)**2)

def _calc_skill(obs: pd.Series, sim: pd.Series) -> dict:
    idx = obs.index.intersection(sim.index)
    o = obs.loc[idx].to_numpy(); s = sim.loc[idx].to_numpy()
    return {"nse": _nse(o, s), "kge": _kge(o, s)}

def _make_residual_frame(obs_s: pd.Series, sim_s: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"obs": obs_s}).join(pd.DataFrame({"sim": sim_s}), how="inner").dropna()
    if df.empty: return df
    df["residual"] = df["sim"] - df["obs"]
    df["flow_pct"] = df["obs"].rank(pct=True) * 100.0  # 0..100
    idx = pd.DatetimeIndex(df.index)
    df["year"]   = idx.year
    df["month"]  = idx.month
    df["season"] = df["month"].map(_season)
    df["decade"] = df["year"].map(_decade_label)
    return df

def _robust_sym_limits_improved(
    y, lo: float = 0.5, hi: float = 99.5, pad: float = 0.10,
    include_zero: bool = True, include_band: Optional[float] = None
) -> Tuple[float, float]:
    """Symmetric limits from broad percentiles; include 0 and optional band; add padding."""
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return (-1, 1)
    ql, qh = np.nanpercentile(y, [lo, hi])
    m = max(abs(ql), abs(qh))
    if include_band is not None:
        m = max(m, abs(include_band))
    if include_zero:
        m = max(m, 0.0)
    m = m * (1 + pad)
    m = max(m, 1e-6)
    return (-m, m)

def _lowess_line(x_num, y, frac=0.10):
    if len(x_num) < 20: return None, None
    order = np.argsort(x_num)
    x_sorted = x_num[order]; y_sorted = y[order]
    if _HAS_LOWESS:
        sm = _lowess(y_sorted, x_sorted, frac=frac, it=1, return_sorted=True)
        return sm[:,0], sm[:,1]
    # fallback: rolling median
    w = max(11, int(0.05*len(y_sorted)//2*2+1))
    y_med = pd.Series(y_sorted).rolling(w, center=True).median().to_numpy()
    return x_sorted, y_med

def _downsample_series(s: pd.Series, step: Optional[int]) -> pd.Series:
    if step is None or step <= 1: return s
    return s.iloc[::step]


# ========= 1) Multi-simulation Error Time Series =========
def plot_error_time_series_enhanced_multi(
    obs: pd.Series,
    sim_dict: Dict[str, pd.Series],  # {"Independent": Series, "Pywr Parametric": Series, "Pywr Default": Series}
    title_prefix: str = "",
    start: Optional[str] = None, end: Optional[str] = None,
    save_path: Optional[str] = None,
    window_days: int = 60, lowess_frac: float = 0.10,
    acceptable_band: Optional[float] = None,
    max_date_ticks: int = 10,
    put_skill_outside: bool = True,
    figsize_ts: Tuple[int,int] = (16, 6),
    downsample_step: Optional[int] = 2,       # light decimation
):
    """
    Plots residual time series for multiple simulations vs the same obs baseline.
    - Slimmer lines (lw=0.35) with high contrast
    - Broader y-lims to avoid clipping
    - Optional 60-day mean and LOWESS shown for the 'Pywr Parametric' (or first)
    - Legend & optional decadal skill table outside
    """
    slicer = slice(start, end) if (start or end) else slice(None)
    obs = obs.loc[slicer].dropna()

    # residual frames (+ optional downsampling)
    residuals: Dict[str, pd.DataFrame] = {}
    for name, sim in sim_dict.items():
        s = _downsample_series(sim.reindex(obs.index), downsample_step)
        df = _make_residual_frame(obs, s)
        if df.empty:
            continue
        residuals[name] = df

    if not residuals:
        return

    # y-lims across all sims
    all_y = np.concatenate([df["residual"].values for df in residuals.values()])
    ylo, yhi = _robust_sym_limits_improved(
        all_y, lo=0.5, hi=99.5, pad=0.12, include_zero=True, include_band=acceptable_band
    )

    fig, ax = plt.subplots(figsize=figsize_ts)

    for name, df in residuals.items():
        style = SIM_STYLES.get(name, dict(lw=0.5, alpha=0.95, zorder=5))
        ax.plot(df.index, df["residual"], label=name, **style)

    if acceptable_band is not None:
        ax.axhspan(-acceptable_band, acceptable_band, color="grey", alpha=0.10, zorder=0)
        ax.text(0.005, 0.95, f"±{acceptable_band:g} MGD band",
                transform=ax.transAxes, va="top", ha="left", fontsize=9, color="gray")

    # moving-average & LOWESS for a reference series
    key = "Pywr Parametric" if "Pywr Parametric" in residuals else next(iter(residuals.keys()))
    if window_days and window_days > 1:
        run = residuals[key]["residual"].rolling(window_days, center=True).mean()
        ax.plot(run.index, run.values, lw=0.5, color="orange", label=f"{key}: {window_days}-day mean", zorder=7)
    if lowess_frac:
        dfk = residuals[key]
        tnum = mdates.date2num(dfk.index.to_pydatetime())
        xs, ys = _lowess_line(tnum, dfk["residual"].to_numpy(), frac=lowess_frac)
        if xs is not None:
            ax.plot(mdates.num2date(xs), ys, lw=0.5, color="firebrick", alpha=0.9,
                    label=f"{key}: LOWESS (frac={lowess_frac})", zorder=8)

    ax.axhline(0, color="black", ls="--", lw=1)
    ax.set_ylim(ylo, yhi)
    ax.margins(y=0.06)
    ax.set_ylabel("Error (MGD)")
    ax.set_xlabel("Date")
    ax.set_title(f"{title_prefix} — Error Evolution Over Time", fontsize=14, weight="bold", wrap=True)
    ax.grid(True, alpha=0.3)

    locator = mdates.AutoDateLocator(minticks=6, maxticks=max_date_ticks)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    # Legend outside
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), framealpha=0.92)

    # Decadal skill table outside (to the right)
    if put_skill_outside:
        lines = []
        for name, df in residuals.items():
            sim_series = (df["residual"] + obs.loc[df.index]).rename("sim")
            obs_series = obs.loc[df.index]
            by_dec = []
            for decade, g in df.groupby("decade"):
                m = _calc_skill(obs_series.loc[g.index], sim_series.loc[g.index])
                if np.isnan(m.get("nse", np.nan)):
                    continue
                by_dec.append(f"{decade}: NSE={m['nse']:.2f}  KGE={m['kge']:.2f}")
            if by_dec:
                lines.append(f"{name}\n" + "\n".join(by_dec))

        if lines:
            fig.text(
                0.82, 0.98,
                "Decadal skill (daily)\n\n" + "\n\n".join(lines),
                va="top", ha="left", fontsize=10,
                bbox=dict(facecolor="white", alpha=0.9, edgecolor="0.8", boxstyle="round,pad=0.4")
            )
            fig.tight_layout(rect=[0, 0, 0.80, 1.0])
    else:
        fig.tight_layout(rect=[0, 0, 0.82, 1.0])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()


# ========= 2) Multi-simulation Error vs Flow Percentile =========
def plot_error_vs_flow_percentile_enhanced_multi(
    obs: pd.Series,
    sim_dict: Dict[str, pd.Series],                 # {"Independent": Series, ...}
    title_prefix: str,
    save_path: str,
    acceptable_band: Optional[float] = None,
    lowess_frac: float = 0.10,
    max_x_ticks: int = 6,
    figsize_scatter: Tuple[int,int] = (12, 7),
    point_size: float = 6.0,
):
    """
    Overlays residual vs observed flow percentile for multiple simulations.
    Slimmer lines, moderate point size, optional LOWESS on key series.
    """
    frames: Dict[str, pd.DataFrame] = {}
    for name, sim in sim_dict.items():
        df = _make_residual_frame(obs.dropna(), sim)
        if not df.empty:
            frames[name] = df
    if not frames:
        return

    all_y = np.concatenate([df["residual"].values for df in frames.values()])
    ylo, yhi = _robust_sym_limits_improved(all_y, lo=0.5, hi=99.5, pad=0.12, include_zero=True, include_band=acceptable_band)

    fig, ax = plt.subplots(figsize=figsize_scatter)
    for name, df in frames.items():
        style = SIM_STYLES.get(name, {})
        ax.scatter(df["flow_pct"], df["residual"], s=point_size, alpha=0.30,
                   label=name, color=style.get("color", "C0"), zorder=4)

    key = "Pywr Parametric" if "Pywr Parametric" in frames else next(iter(frames.keys()))
    xs, ys = _lowess_line(frames[key]["flow_pct"].to_numpy(), frames[key]["residual"].to_numpy(), frac=lowess_frac)
    if xs is not None:
        ax.plot(xs, ys, lw=2.0, color="black", label=f"{key}: LOWESS", zorder=6)

    if acceptable_band is not None:
        ax.axhspan(-acceptable_band, acceptable_band, color="grey", alpha=0.10, zorder=0)
        ax.text(0.01, 0.95, f"±{acceptable_band:g} MGD", transform=ax.transAxes,
                va="top", ha="left", fontsize=10, color="gray")

    ax.axhline(0, color="black", lw=1, ls="--")
    ax.set_xlim(0, 100); ax.set_ylim(ylo, yhi)
    ax.set_xticks(np.linspace(0, 100, max_x_ticks))
    ax.set_xlabel("Observed Flow Percentile"); ax.set_ylabel("Error (Sim − Obs, MGD)")
    ax.set_title(f"{title_prefix} — Error vs Flow Percentile", fontsize=14, weight="bold", wrap=True)
    ax.grid(True, alpha=0.3)

    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), framealpha=0.92)
    fig.tight_layout(rect=[0,0,0.82,1])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300); plt.close(fig)


# ========= 3) Season 4-panel (single-simulation for readability) =========
def plot_seasonal_decadal_panels(
    df_obs: pd.DataFrame, df_sim: pd.DataFrame, reservoirs: list[str], period_label: str,
    save_folder: str = "figures/error_vs_pct_seasons",
    lowess_frac: float = 0.12, acceptable_band: Optional[float] = None,
    figsize_panels: Tuple[int,int] = (14, 10),
):
    os.makedirs(save_folder, exist_ok=True)

    for res in reservoirs:
        if res not in df_obs.columns or res not in df_sim.columns:
            continue
        df = _make_residual_frame(df_obs[res].dropna(), df_sim[res])
        if df.empty:
            continue

        ylo, yhi = _robust_sym_limits_improved(df["residual"].values, lo=0.5, hi=99.5, pad=0.12, include_zero=True, include_band=acceptable_band)
        fig, axes = plt.subplots(2, 2, figsize=figsize_panels, sharex=True, sharey=True)
        axes = axes.ravel()

        for i, sea in enumerate(SEASON_ORDER):
            ax = axes[i]
            sub = df[df["season"] == sea]
            if sub.empty:
                ax.set_title(sea, wrap=True); ax.grid(True, alpha=0.3); continue

            for dec in DECADE_ORDER:
                d2 = sub[sub["decade"] == dec]
                if d2.empty: continue
                ax.scatter(d2["flow_pct"], d2["residual"], s=16, alpha=0.30,
                           color=DECADE_COLORS.get(dec, "0.5"), label=dec, zorder=3)

            xs, ys = _lowess_line(sub["flow_pct"].to_numpy(), sub["residual"].to_numpy(), frac=lowess_frac)
            if xs is not None: ax.plot(xs, ys, lw=2.0, color="black", zorder=5)

            if acceptable_band is not None:
                ax.axhspan(-acceptable_band, acceptable_band, color="grey", alpha=0.08, zorder=0)

            ax.axhline(0, color="black", ls="--", lw=1)
            ax.set_title(sea, wrap=True); ax.set_xlim(0, 100); ax.set_ylim(ylo, yhi)
            ax.grid(True, alpha=0.3)

        for ax in (axes[2], axes[3]): ax.set_xlabel("Observed Flow Percentile")
        for ax in (axes[0], axes[2]): ax.set_ylabel("Error (Sim − Obs, MGD)")
        for ax in axes: ax.set_xticks([0,20,40,60,80,100])

        handles = [Line2D([0],[0], marker='o', color='none',
                          markerfacecolor=DECADE_COLORS[d], markersize=8, label=d)
                   for d in DECADE_ORDER]
        fig.legend(handles, [h.get_label() for h in handles],
                   loc="center left", bbox_to_anchor=(1.02, 0.5),
                   title="Decade", framealpha=0.94)

        fig.suptitle(f"{period_label} — {res}: Residual vs Flow Percentile by Season & Decade",
                     fontsize=15, weight="bold", wrap=True)
        fig.tight_layout(rect=[0,0,0.86,0.96])
        out = f"{save_folder}/error_vs_flow_percentile_season4_{_safe_label(res)}.png"
        fig.savefig(out, dpi=300); plt.close(fig)
        print(f"Saved: {out}")
