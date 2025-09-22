# methods/plotting/plot_error_diagnostics.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.offsetbox import AnchoredText
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

def _season(m): return SEASON_MAP.get(int(m), "NA")
def _decade_label(y): d = (int(y)//10)*10; return f"{d}s"
def _safe_label(s: str) -> str: return str(s).replace(" ", "_")

# basic metrics (standalone)
def _nse(obs: np.ndarray, sim: np.ndarray) -> float:
    obs = np.asarray(obs, float); sim = np.asarray(sim, float)
    m = np.nanmean(obs)
    denom = np.nansum((obs - m)**2)
    if denom == 0 or np.isnan(denom): return np.nan
    return 1.0 - (np.nansum((sim - obs)**2) / denom)

def _kge(obs: np.ndarray, sim: np.ndarray) -> float:
    obs = np.asarray(obs, float); sim = np.asarray(sim, float)
    # Pearson r (nan-safe)
    ok = np.isfinite(obs) & np.isfinite(sim)
    if ok.sum() < 2: return np.nan
    r = np.corrcoef(obs[ok], sim[ok])[0,1]
    alpha = (np.nanstd(sim) / np.nanstd(obs)) if np.nanstd(obs) > 0 else np.nan
    beta  = (np.nanmean(sim) / np.nanmean(obs)) if np.nanmean(obs) != 0 else np.nan
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

def _robust_sym_limits(y, lo=1.0, hi=99.0, pad=0.08):
    if len(y) == 0: return (-1, 1)
    ql, qh = np.nanpercentile(y, [lo, hi])
    m = max(abs(ql), abs(qh))
    return (-m*(1+pad), m*(1+pad))

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

# 1) Error time-series
def plot_error_time_series_enhanced(
    df_obs: pd.DataFrame,
    df_sim: pd.DataFrame,
    reservoirs: list[str],
    start: str | None = None, end: str | None = None,
    period_label: str = "",
    save_folder: str = "figures/error_ts",
    window_days: int = 60, lowess_frac: float = 0.10,
    acceptable_band: float | None = None,
    annotate_decadal_metrics: bool = True,
    color_points_by_decade: bool = False
):
    os.makedirs(save_folder, exist_ok=True)
    slicer = slice(start, end) if (start or end) else slice(None)

    for res in reservoirs:
        if res not in df_obs.columns or res not in df_sim.columns: continue
        obs = df_obs[res].loc[slicer].dropna()
        sim = df_sim[res].reindex(obs.index)
        df  = _make_residual_frame(obs, sim)
        if df.empty: continue

        ylo, yhi = _robust_sym_limits(df["residual"].values)
        fig, ax = plt.subplots(figsize=(12, 4))

        if color_points_by_decade:
            for dec, sub in df.groupby("decade"):
                ax.plot(sub.index, sub["residual"], lw=0.6,
                        color=DECADE_COLORS.get(dec, "0.5"), label=dec, alpha=0.7)
        else:
            ax.plot(df.index, df["residual"], lw=0.8, color="steelblue",
                    label="Residual (Sim − Obs)")

        if acceptable_band is not None:
            ax.axhspan(-acceptable_band, acceptable_band, color="grey", alpha=0.10, zorder=0)
            ax.text(0.005, 0.95, f"±{acceptable_band:g} MGD band",
                    transform=ax.transAxes, va="top", ha="left", fontsize=8, color="gray")

        if window_days and window_days > 1:
            run = df["residual"].rolling(window_days, center=True).mean()
            ax.plot(df.index, run, lw=1.6, color="orange", label=f"{window_days}-day mean")

        if lowess_frac:
            tnum = mdates.date2num(df.index.to_pydatetime())
            xs, ys = _lowess_line(tnum, df["residual"].to_numpy(), frac=lowess_frac)
            if xs is not None:
                ax.plot(mdates.num2date(xs), ys, lw=2.0, color="firebrick", alpha=0.9,
                        label=f"LOWESS (frac={lowess_frac})")

        ax.axhline(0, color="black", ls="--", lw=1)
        ax.set_ylim(ylo, yhi)
        ax.set_title(f"{period_label} — Error Evolution Over Time — {res}", fontsize=13, weight="bold")
        ax.set_ylabel("Error (MGD)"); ax.set_xlabel("Date")
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.grid(True, alpha=0.25)

        if annotate_decadal_metrics:
            lines = []
            for decade, g in df.groupby("decade"):
                m = _calc_skill(obs.loc[g.index], sim.loc[g.index])
                if np.isnan(m.get("nse", np.nan)): continue
                lines.append(f"{decade}: NSE={m['nse']:.2f}  KGE={m['kge']:.2f}")
            if lines:
                at = AnchoredText("Decadal skill (daily)\n" + "\n".join(lines),
                                  loc=1, prop=dict(size=9), frameon=True, borderpad=0.4)
                at.patch.set_alpha(0.85); ax.add_artist(at)

        if color_points_by_decade:
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), framealpha=0.9)
            fig.tight_layout(rect=[0,0,0.86,1])
        else:
            ax.legend(loc="lower left", framealpha=0.85)
            fig.tight_layout()

        out = f"{save_folder}/error_timeseries_{_safe_label(res)}.png"
        fig.savefig(out, dpi=300); plt.close(fig)
        print(f"Saved: {out}")

# 2) Error vs flow percentile
def plot_error_vs_flow_percentile_enhanced(
    df_obs: pd.DataFrame, df_sim: pd.DataFrame, reservoirs: list[str], period_label: str,
    save_folder: str = "figures/error_vs_pct",
    acceptable_band: float | None = None, lowess_frac: float = 0.10,
    color_by: str = "decade",  # or "season"
    alpha: float = 0.25, s: float = 10
):
    os.makedirs(save_folder, exist_ok=True)

    for res in reservoirs:
        if res not in df_obs.columns or res not in df_sim.columns: continue
        df = _make_residual_frame(df_obs[res].dropna(), df_sim[res])
        if df.empty: continue

        x = df["flow_pct"].to_numpy()
        y = df["residual"].to_numpy()
        ylo, yhi = _robust_sym_limits(y)

        fig, ax = plt.subplots(figsize=(9, 6))
        if color_by == "decade":
            for dec in DECADE_ORDER:
                sub = df[df["decade"] == dec]
                if sub.empty: continue
                ax.scatter(sub["flow_pct"], sub["residual"], s=s, alpha=alpha,
                           color=DECADE_COLORS.get(dec, "0.5"), label=dec)
        else:
            for sea in SEASON_ORDER:
                sub = df[df["season"] == sea]
                if sub.empty: continue
                ax.scatter(sub["flow_pct"], sub["residual"], s=s, alpha=alpha,
                           color=SEASON_COLORS.get(sea, "0.5"), label=sea)

        xs, ys = _lowess_line(x, y, frac=lowess_frac)
        if xs is not None:
            ax.plot(xs, ys, lw=2.0, color="black", label="LOWESS trend")

        if acceptable_band is not None:
            ax.axhspan(-acceptable_band, acceptable_band, color="grey", alpha=0.10, zorder=0)
            ax.text(0.01, 0.95, f"±{acceptable_band:g} MGD", transform=ax.transAxes,
                    va="top", ha="left", fontsize=8, color="gray")

        ax.axhline(0, color="black", lw=1, ls="--")
        ax.set_xlim(0, 100); ax.set_ylim(ylo, yhi)
        ax.set_xticks([0,20,40,60,80,100])
        ax.set_title(f"{period_label} — Error vs Flow Percentile — {res}", fontsize=13, weight="bold")
        ax.set_xlabel("Observed Flow Percentile"); ax.set_ylabel("Error (Sim − Obs, MGD)")
        ax.grid(True, alpha=0.25)

        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), framealpha=0.9)
        fig.tight_layout(rect=[0,0,0.86,1])
        out = f"{save_folder}/error_vs_flow_percentile_{_safe_label(res)}.png"
        fig.savefig(out, dpi=300); plt.close(fig)
        print(f"Saved: {out}")

# 3) Season 4-panel
def plot_seasonal_decadal_panels(
    df_obs: pd.DataFrame, df_sim: pd.DataFrame, reservoirs: list[str], period_label: str,
    save_folder: str = "figures/error_vs_pct_seasons",
    lowess_frac: float = 0.12, acceptable_band: float | None = None
):
    os.makedirs(save_folder, exist_ok=True)

    for res in reservoirs:
        if res not in df_obs.columns or res not in df_sim.columns: continue
        df = _make_residual_frame(df_obs[res].dropna(), df_sim[res])
        if df.empty: continue

        ylo, yhi = _robust_sym_limits(df["residual"].values)
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
        axes = axes.ravel()

        for i, sea in enumerate(SEASON_ORDER):
            ax = axes[i]
            sub = df[df["season"] == sea]
            if sub.empty:
                ax.set_title(sea); ax.grid(True, alpha=0.25); continue

            for dec in DECADE_ORDER:
                d2 = sub[sub["decade"] == dec]
                if d2.empty: continue
                ax.scatter(d2["flow_pct"], d2["residual"], s=13, alpha=0.28,
                           color=DECADE_COLORS.get(dec, "0.5"), label=dec)

            xs, ys = _lowess_line(sub["flow_pct"].to_numpy(), sub["residual"].to_numpy(), frac=lowess_frac)
            if xs is not None: ax.plot(xs, ys, lw=2.0, color="black")

            if acceptable_band is not None:
                ax.axhspan(-acceptable_band, acceptable_band, color="grey", alpha=0.08, zorder=0)

            ax.axhline(0, color="black", ls="--", lw=1)
            ax.set_title(sea); ax.set_xlim(0, 100); ax.set_ylim(ylo, yhi)
            ax.grid(True, alpha=0.25)

        for ax in (axes[2], axes[3]): ax.set_xlabel("Observed Flow Percentile")
        for ax in (axes[0], axes[2]): ax.set_ylabel("Error (Sim − Obs, MGD)")
        for ax in axes: ax.set_xticks([0,20,40,60,80,100])

        handles = [Line2D([0],[0], marker='o', color='none',
                          markerfacecolor=DECADE_COLORS[d], markersize=7, label=d)
                   for d in DECADE_ORDER]
        fig.legend(handles, [h.get_label() for h in handles],
                   loc="center left", bbox_to_anchor=(1.02, 0.5),
                   title="Decade", framealpha=0.9)

        fig.suptitle(f"{period_label} — {res}: Residual vs Flow Percentile by Season & Decade",
                     fontsize=14, weight="bold")
        fig.tight_layout(rect=[0,0,0.86,0.96])
        out = f"{save_folder}/error_vs_flow_percentile_season4_{_safe_label(res)}.png"
        fig.savefig(out, dpi=300); plt.close(fig)
        print(f"Saved: {out}")
