from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, Tuple, List
import numpy as np
import pandas as pd

# Optional: only used by pick_focal_solution
from methods.config import (
    OUTPUT_DIR, OBJ_LABELS, OBJ_FILTER_BOUNDS, NFE, ISLANDS, SEED
)
from methods.load.results import load_results

# ===================== Colors =====================

LEGACY_COLORS = {
    "Best Release NSE": "red",
    "Best Storage NSE": "green",
    "Best Average NSE": "purple",
    "Best Average All": "blue",
    "Other": "lightgray",
}

# NOTE: labels here must match what select_candidate_policies() emits.
ADVANCED_COLORS = {
    "Compromise L2 (Euclidean)": "red",
    "Tchebycheff L∞": "darkorange",
    "Manhattan L1": "purple",
    "Knee (max curvature)": "green",
    "ε-constraint Release NSE ≥ Q50": "blue",
    "ε-constraint Release NSE ≥ Q80": "navy",
    "Diverse #1 (FPS)": "brown",
    "Diverse #2 (FPS)": "olive",
    "Max HV Contribution": "teal",
    "Other": "lightgray",
}

# ===================== Normalization =====================

def _bounds_for(
    col: str, df: pd.DataFrame, bounds: Dict[str, Tuple[float, float]] | None
) -> Tuple[float, float]:
    if bounds and col in bounds:
        lo, hi = bounds[col]
        if hi > lo:
            return float(lo), float(hi)
    lo, hi = float(df[col].min()), float(df[col].max())
    if hi <= lo:
        hi = lo + 1e-12
    return lo, hi

def normalize_objectives(
    df: pd.DataFrame,
    objectives: List[str],
    senses: Dict[str, str],
    bounds: Dict[str, Tuple[float,float]] | None = None
) -> pd.DataFrame:
    out = df.copy()
    for col in objectives:
        lo, hi = _bounds_for(col, df, bounds)
        raw = df[col].astype(float)
        if senses.get(col, "max").lower().startswith("max"):
            norm = (raw - lo) / (hi - lo)
        else:
            norm = (hi - raw) / (hi - lo)
        out[f"{col}__norm"] = np.clip(norm.values, 0.0, 1.0)
    return out

# ===================== Selection Math =====================

def nondominated_mask_2d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    idx = np.argsort(-x, kind="mergesort")
    y_best = -np.inf
    keep = np.zeros_like(idx, dtype=bool)
    for k in idx:
        if y[k] > y_best + 1e-12:
            keep[k] = True
            y_best = y[k]
    return keep

def lp_distance_to_ideal(
    norm_df: pd.DataFrame,
    objectives: List[str],
    p: float = 2.0,
    weights: Iterable[float] | None = None
) -> pd.Series:
    cols = [f"{c}__norm" for c in objectives]
    X = norm_df[cols].to_numpy(dtype=float)
    w = np.ones(X.shape[1]) if weights is None else np.asarray(list(weights), dtype=float)
    w = w / w.sum()
    D = np.abs(1.0 - X)
    if np.isinf(p):
        d = (w * D).max(axis=1)
    elif p == 1:
        d = (w * D).sum(axis=1)
    else:
        d = np.power((w * np.power(D, p)).sum(axis=1), 1.0 / p)
    return pd.Series(d, index=norm_df.index, name=f"L{p}_dist")

def knee_point_2d(norm_df: pd.DataFrame, objectives: List[str]) -> int:
    """Knee = point with max distance to the chord between the axis-wise bests."""
    assert len(objectives) == 2
    x = norm_df[f"{objectives[0]}__norm"].to_numpy(dtype=float)
    y = norm_df[f"{objectives[1]}__norm"].to_numpy(dtype=float)

    i_x = int(np.argmax(x))  # best on obj0
    i_y = int(np.argmax(y))  # best on obj1
    P1 = np.array([x[i_x], y[i_x]], dtype=float)
    P2 = np.array([x[i_y], y[i_y]], dtype=float)

    v = P2 - P1
    denom = np.hypot(v[0], v[1]) + 1e-12

    dx = x - P1[0]
    dy = y - P1[1]
    d = np.abs(v[0] * dy - v[1] * dx) / denom
    d[[i_x, i_y]] = -np.inf  # avoid endpoints

    return int(np.argmax(d))

def eps_constraint_sweep(
    norm_df: pd.DataFrame,
    objectives: List[str],
    eps_on: str,
    optimize: str,
    eps_quantiles: Iterable[float] = (0.5, 0.75)
) -> List[int]:
    eps_col = f"{eps_on}__norm"; opt_col = f"{optimize}__norm"
    qs = np.clip(np.array(list(eps_quantiles), dtype=float), 0.0, 1.0)
    picks = []
    thr_vals = norm_df[eps_col].quantile(qs, interpolation="nearest").to_numpy()
    for thr in thr_vals:
        sub = norm_df[norm_df[eps_col] >= thr]
        if len(sub) == 0:
            continue
        picks.append(int(sub[opt_col].idxmax()))
    return picks

def farthest_point_sampling(norm_df: pd.DataFrame, objectives: List[str], k: int = 2) -> List[int]:
    cols = [f"{c}__norm" for c in objectives]
    X = norm_df[cols].to_numpy()
    seeds = set([int(norm_df[f"{objectives[0]}__norm"].idxmax()),
                 int(norm_df[f"{objectives[1]}__norm"].idxmax())])
    picks = list(seeds) if len(seeds) else [int(np.argmax(X.sum(axis=1)))]
    while len(picks) < len(seeds) + k:
        dmin = np.min(np.linalg.norm(X[:, None, :] - X[picks][None, :, :], axis=2), axis=1)
        dmin[picks] = -np.inf
        nxt = int(np.argmax(dmin))
        if not np.isfinite(dmin[nxt]) or dmin[nxt] < 0:
            break
        picks.append(nxt)
    return picks[len(seeds):]

def hypervolume_2d(points: np.ndarray) -> float:
    pts = points[np.argsort(-points[:,0], kind="mergesort")]
    hv, y_star = 0.0, 0.0
    for i in range(len(pts)):
        x_i, y_i = pts[i]
        x_next = pts[i+1,0] if i < len(pts)-1 else 0.0
        y_star = max(y_star, y_i)
        hv += y_star * max(x_i - x_next, 0.0)
    return float(hv)

def hv_contribution_top_2d(norm_df: pd.DataFrame, objectives: List[str]) -> int | None:
    X = norm_df[[f"{c}__norm" for c in objectives]].to_numpy()
    mask_nd = nondominated_mask_2d(X[:,0], X[:,1])
    idxs = norm_df.index.to_numpy()
    P = X[mask_nd]
    nd_idxs = idxs[mask_nd]
    if len(P) == 0:
        return None
    hv_total = hypervolume_2d(P)
    contribs = []
    for j in range(len(P)):
        P_minus = np.delete(P, j, axis=0) if len(P) > 1 else np.empty((0,2))
        c = hv_total - (hypervolume_2d(P_minus) if len(P) > 0 else 0.0)
        contribs.append(c)
    return int(nd_idxs[int(np.argmax(contribs))])

def select_candidate_policies(
    obj_df: pd.DataFrame,
    objectives: List[str] = ("Release NSE","Storage NSE"),
    senses: Dict[str,str] | None = None,
    bounds: Dict[str, Tuple[float,float]] | None = None,
    weights: Iterable[float] | None = None,
    eps_qs: Iterable[float] = (0.5, 0.75),
    add_k_diverse: int = 2,
    include_hv: bool = True
) -> pd.DataFrame:
    """Return a small table of candidate picks with global indices and normalized scores."""
    if senses is None:
        senses = {o: "max" for o in objectives}  # use "min" if your CSV contains negated metrics

    nd = normalize_objectives(obj_df, list(objectives), senses, bounds=bounds)
    picks: Dict[str,int] = {}
    picks["Compromise L2 (Euclidean)"] = int(lp_distance_to_ideal(nd, list(objectives), p=2, weights=weights).idxmin())
    picks["Tchebycheff L∞"]           = int(lp_distance_to_ideal(nd, list(objectives), p=np.inf, weights=weights).idxmin())
    picks["Manhattan L1"]             = int(lp_distance_to_ideal(nd, list(objectives), p=1, weights=weights).idxmin())

    if len(objectives) == 2 and len(obj_df) >= 3:
        picks["Knee (max curvature)"] = knee_point_2d(nd, list(objectives))

    if len(objectives) >= 2:
        eps_on, optimize = objectives[0], objectives[1]
        for q_idx, i in enumerate(eps_constraint_sweep(nd, list(objectives), eps_on=eps_on, optimize=optimize, eps_quantiles=eps_qs)):
            picks[f"ε-constraint {eps_on} ≥ Q{int(100*list(eps_qs)[q_idx])}"] = i

    if add_k_diverse > 0:
        for k_idx, i in enumerate(farthest_point_sampling(nd, list(objectives), k=add_k_diverse), start=1):
            picks[f"Diverse #{k_idx} (FPS)"] = i

    if include_hv and len(objectives) == 2 and len(obj_df) >= 2:
        hv_idx = hv_contribution_top_2d(nd, list(objectives))
        if hv_idx is not None:
            picks["Max HV Contribution"] = hv_idx

    seen = set(); rows = []
    for label, idx in picks.items():
        if idx in seen:
            continue
        seen.add(idx)
        row = {"label": label, "index": idx}
        for c in objectives:
            row[c] = float(obj_df.loc[idx, c])
            row[f"{c}__norm"] = float(nd.loc[idx, f"{c}__norm"])
        rows.append(row)

    out = pd.DataFrame(rows).sort_values(
        by=[f"{objectives[0]}__norm", f"{objectives[1]}__norm"], ascending=False
    ).reset_index(drop=True)
    return out

def legacy_best_indices(
    obj_df: pd.DataFrame,
    senses: Dict[str, str],
    avg_pair: Tuple[str, str] = ("Release NSE","Storage NSE"),
    all_objectives_for_legacy: List[str] | None = None,
    bounds: Dict[str, Tuple[float,float]] | None = None,
) -> Dict[str, int]:
    """Sense-aware legacy picks for back-compat."""
    def _best_idx(col: str) -> int:
        sense = senses.get(col, "max").lower()
        return int((obj_df[col].idxmin() if sense.startswith("min") else obj_df[col].idxmax()))

    i_rel = _best_idx(avg_pair[0])
    i_sto = _best_idx(avg_pair[1])

    nd2 = normalize_objectives(obj_df, list(avg_pair), senses, bounds=bounds)
    i_avg = int(nd2[[f"{avg_pair[0]}__norm", f"{avg_pair[1]}__norm"]].mean(axis=1).idxmax())

    objs_all = list(all_objectives_for_legacy) if all_objectives_for_legacy else list(avg_pair)
    nd_all = normalize_objectives(obj_df, objs_all, senses, bounds=bounds)
    i_all = int(nd_all[[f"{c}__norm" for c in objs_all]].mean(axis=1).idxmax())

    return {
        "Best Release NSE": i_rel,
        "Best Storage NSE": i_sto,
        "Best Average NSE": i_avg,
        "Best Average All": i_all,
    }

# ===================== Glue: stamping highlights =====================

def apply_selections_to_obj_df(
    obj_df: pd.DataFrame,
    candidates: str | Path | pd.DataFrame | Dict[str, int],
    *,
    label_col: str = "label",
    index_col: str = "index",
    out_label_col: str = "highlight",
    default_label: str = "Other",
) -> tuple[pd.DataFrame, Dict[str, int]]:
    """
    Stamp a label column (e.g., 'highlight' or 'highlight_adv') into obj_df
    based on a set of candidate selections.
    """
    # normalize input -> DataFrame[label, index]
    if isinstance(candidates, (str, Path)):
        cdf = pd.read_csv(candidates)
    elif isinstance(candidates, dict):
        label_to_idx = {str(k): int(v) for k, v in candidates.items()}
        cdf = pd.DataFrame({label_col: list(label_to_idx.keys()),
                            index_col: list(label_to_idx.values())})
    elif isinstance(candidates, pd.DataFrame):
        cdf = candidates.copy()
    else:
        raise TypeError("`candidates` must be CSV path, DataFrame, or dict {label: index}.")

    if label_col not in cdf.columns or index_col not in cdf.columns:
        raise ValueError(f"Candidates missing required columns: {label_col}, {index_col}")

    cdf = cdf.drop_duplicates(subset=[label_col], keep="first")

    valid = cdf[index_col].astype(int)
    valid = valid[(valid >= 0) & (valid < len(obj_df))]
    cdf = cdf.loc[valid.index]

    label_to_idx: Dict[str, int] = dict(zip(cdf[label_col].astype(str), cdf[index_col].astype(int)))

    out = obj_df.copy()
    out[out_label_col] = default_label
    for lab, idx in label_to_idx.items():
        out.loc[idx, out_label_col] = lab

    return out, label_to_idx

def compute_and_apply_advanced_highlights(
    obj_df: pd.DataFrame,
    *,
    objectives: List[str] = ("Release NSE", "Storage NSE"),
    senses: Dict[str, str] | None = None,
    bounds: Dict[str, Tuple[float, float]] | None = None,
    weights: Iterable[float] | None = None,
    eps_qs: Iterable[float] = (0.5, 0.8),
    add_k_diverse: int = 2,
    include_hv: bool = True,
    out_label_col: str = "highlight_adv",
    default_label: str = "Other",
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """
    One-call helper:
      - select advanced candidates,
      - stamp them into a new label column on obj_df,
      - return (obj_df_with_highlight_adv, candidates_df, mapping).
    """
    if senses is None:
        senses = {o: "max" for o in objectives}

    cand_df = select_candidate_policies(
        obj_df=obj_df,
        objectives=list(objectives),
        senses=senses,
        bounds=bounds,
        weights=weights,
        eps_qs=eps_qs,
        add_k_diverse=add_k_diverse,
        include_hv=include_hv,
    )
    out_df, mapping = apply_selections_to_obj_df(
        obj_df, candidates=cand_df, out_label_col=out_label_col, default_label=default_label
    )
    return out_df, cand_df, mapping

# ===================== Optional: end-to-end pick loader =====================

def pick_focal_solution(
    reservoir: str,
    policy: str,
    outputs_dir: Path = OUTPUT_DIR,
    pick: str = "Best Average NSE",
    objectives: List[str] = ("Release NSE","Storage NSE"),
    senses: Dict[str,str] | None = None,
    bounds: Dict[str, Tuple[float,float]] | None = None,
    weights: Iterable[float] | None = None,
    eps_qs: Iterable[float] = (0.5, 0.8),
    add_k_diverse: int = 2,
    include_hv: bool = True,
    save_candidates_csv: bool = True,
    all_objectives_for_legacy: List[str] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, Dict[str,int]]:
    outputs_dir = Path(outputs_dir)
    fname = outputs_dir / f"MMBorg_{ISLANDS}M_{policy}_{reservoir}_nfe{NFE}_seed{SEED}.csv"
    if not fname.exists():
        raise FileNotFoundError(f"Expected CSV not found: {fname}")

    obj_df, var_df = load_results(
        str(fname),
        obj_labels=OBJ_LABELS,
        filter=True,
        obj_bounds=OBJ_FILTER_BOUNDS
    )
    if obj_df is None or obj_df.empty:
        raise FileNotFoundError(f"No solutions in {fname}")

    if senses is None:
        senses = {o: "max" for o in objectives}

    use_hv = (include_hv and len(objectives) == 2)

    cand_df = select_candidate_policies(
        obj_df,
        objectives=list(objectives),
        senses=senses,
        bounds=bounds if bounds is not None else OBJ_FILTER_BOUNDS,
        weights=weights,
        eps_qs=eps_qs,
        add_k_diverse=add_k_diverse,
        include_hv=use_hv
    )
    advanced_idxs: Dict[str,int] = dict(zip(cand_df["label"], cand_df["index"]))

    legacy_idxs = legacy_best_indices(
        obj_df=obj_df,
        senses=senses,
        avg_pair=("Release NSE","Storage NSE"),
        all_objectives_for_legacy=(all_objectives_for_legacy or list(objectives)),
        bounds=(bounds or OBJ_FILTER_BOUNDS),
    )

    candidate_indices: Dict[str,int] = {**legacy_idxs, **advanced_idxs}

    if save_candidates_csv:
        outdir = Path(OUTPUT_DIR) / "focal_picks"
        outdir.mkdir(parents=True, exist_ok=True)
        tag = f"{policy}_{reservoir}_seed{SEED}_nfe{NFE}"

        cand_df.assign(reservoir=reservoir, policy=policy).to_csv(
            outdir / f"candidates_{tag}.csv", index=False
        )

        rows = []
        nd = normalize_objectives(obj_df, list(objectives), senses, bounds=(bounds or OBJ_FILTER_BOUNDS))
        for lab, idx in legacy_idxs.items():
            row = {"label": lab, "index": idx, "reservoir": reservoir, "policy": policy}
            for c in objectives:
                row[c] = float(obj_df.loc[idx, c])
                row[f"{c}__norm"] = float(nd.loc[idx, f"{c}__norm"])
            rows.append(row)
        legacy_df = pd.DataFrame(
            rows,
            columns=["label","index","reservoir","policy"] + [*objectives, *(f"{c}__norm" for c in objectives)]
        )
        merged_df = pd.concat(
            [cand_df.assign(reservoir=reservoir, policy=policy), legacy_df],
            ignore_index=True
        )
        merged_df.to_csv(outdir / f"candidates_all_{tag}.csv", index=False)

    if pick not in candidate_indices:
        fallback_idx = advanced_idxs.get("Compromise L2 (Euclidean)", legacy_idxs["Best Average NSE"])
        params_vec = var_df.loc[int(fallback_idx)].values.astype(float)
    else:
        params_vec = var_df.loc[int(candidate_indices[pick])].values.astype(float)

    return obj_df, var_df, params_vec, candidate_indices

def build_candidate_thetas(
    var_df: pd.DataFrame,
    cand_idxs: Dict[str, int],
    which: List[str] | None = None
) -> Dict[str, np.ndarray]:
    labels = which or list(cand_idxs.keys())
    out = {}
    for lab in labels:
        if lab not in cand_idxs:
            raise KeyError(f"Candidate label not in cand_idxs: {lab}")
        idx = int(cand_idxs[lab])
        out[lab] = var_df.loc[idx].values.astype(float)
    return out

# ---------- convenience: load saved candidates ----------
def load_candidates_csv(path: str | Path, label_col: str = "label", index_col: str = "index") -> Dict[str, int]:
    df = pd.read_csv(path)
    if label_col not in df.columns or index_col not in df.columns:
        raise ValueError(f"CSV must have columns [{label_col}, {index_col}]")
    df = df.drop_duplicates(subset=[label_col], keep="first")
    return {str(r[label_col]): int(r[index_col]) for _, r in df.iterrows()}
