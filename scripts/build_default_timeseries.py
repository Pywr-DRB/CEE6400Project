#!/usr/bin/env python3
# scripts/build_default_timeseries.py
# Create default Pywr-DRB time series (per-reservoir CSVs) without requiring observed data.

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import os

import numpy as np
import pandas as pd

# ---------------------- Paths & config ----------------------------------------
HERE = Path(__file__).parent.resolve()
# Put outputs at project root (one level above scripts/)
ROOT = HERE.parent
OUTPUTS_DIR   = Path(os.environ.get("DRB_OUTPUT_DIR", (ROOT / "pywr_data")))
DEFAULT_SERIES_DIR = Path(os.environ.get("DRB_DEFAULT_SERIES", (OUTPUTS_DIR / "_default_series")))
CACHE_DIR = Path(os.environ.get("DRB_DEFAULT_CACHE", (OUTPUTS_DIR / "_pywr_default_cache")))

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def smart_read_csv(p: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

# ------------------------------ pywrdrb ---------------------------------------
try:
    import pywrdrb  # top-level package (your editable install)
    from pywrdrb import Model, ModelBuilder, OutputRecorder
except Exception as e:
    print("[ERROR] Could not import pywrdrb. Is your env activated?", file=sys.stderr)
    print(f"Details: {e}", file=sys.stderr)
    sys.exit(1)

for sym in ("Model", "ModelBuilder", "OutputRecorder", "Data"):
    if not hasattr(pywrdrb, sym):
        print(
            f"[WARN] pywrdrb.{sym} not found. If your fork exposes it elsewhere, "
            f"adjust imports (e.g., from pywrdrb.postprocessing.data import Data).",
            file=sys.stderr,
        )

# ----------------------- Default run build & load ------------------------------
def save_default_pywr_run(
    start_date: str = "1983-10-01",
    end_date: str   = "2023-12-31",
    inflow_type: str = "pub_nhmv10_BC_withObsScaled",
    outdir: Path = CACHE_DIR,
    overwrite: bool = False,
) -> Path:
    """
    Build & run the default Pywr-DRB model once and cache HDF5.
    Returns the HDF5 path.
    """
    ensure_dir(outdir)
    tag = f"{start_date}_{end_date}_{inflow_type}"
    h5_path = outdir / f"output_default_{tag}.hdf5"
    model_json = outdir / f"model_default_{tag}.json"

    if h5_path.exists() and not overwrite:
        print(f"[default] Reusing cached HDF5: {h5_path}")
        return h5_path

    print(f"[default] Building model: {start_date} → {end_date} | inflow={inflow_type}")
    mb = ModelBuilder(start_date=start_date, end_date=end_date, inflow_type=inflow_type)
    mb.make_model()
    mb.write_model(str(model_json))

    print(f"[default] Running model → {h5_path.name}")
    m = Model.load(str(model_json))
    rec = OutputRecorder(m, str(h5_path))
    _ = m.run()

    if not h5_path.exists():
        raise RuntimeError("Default run finished but HDF5 not found. Check pywrdrb setup.")
    print(f"[default] Saved HDF5: {h5_path}")
    return h5_path


def list_reservoirs_from_h5(h5_path: Path) -> List[str]:
    """
    Inspect the cached HDF5 and return a sorted list of reservoir names
    by union of columns in res_storage and reservoir_downstream_gage.
    """
    dataD = pywrdrb.Data(
        print_status=False,
        results_sets=["res_storage", "reservoir_downstream_gage"],
        output_filenames=[str(h5_path)],
    )
    dataD.load_output()
    key = h5_path.stem

    names = set()
    try:
        df_sto = dataD.res_storage[key][0]
        names.update([c for c in df_sto.columns if isinstance(c, str)])
    except Exception:
        pass
    try:
        df_rel = dataD.reservoir_downstream_gage[key][0]
        names.update([c for c in df_rel.columns if isinstance(c, str)])
    except Exception:
        pass

    names = sorted([n for n in names if n and n.lower() != "date"])
    if not names:
        raise RuntimeError("No reservoir columns found in default HDF5 outputs.")
    return names


def load_default_series_for_reservoir(
    h5_path: Path, reservoir: str, return_mrf: bool = True
) -> Tuple[Optional[pd.Series], Optional[pd.Series], Optional[pd.Series]]:
    """
    Extract default Release (MGD) and Storage (MG) for one reservoir from cached HDF5.
    Optionally returns MRF contribution (MGD) if available; may be None.
    """
    dataD = pywrdrb.Data(
        print_status=False,
        results_sets=["res_storage", "reservoir_downstream_gage"] + (["lower_basin_mrf_contributions"] if return_mrf else []),
        output_filenames=[str(h5_path)],
    )
    dataD.load_output()
    key = h5_path.stem

    rel = None
    sto = None
    mrf = None

    try:
        rel = dataD.reservoir_downstream_gage[key][0][reservoir].astype(float)
        rel.name = "Default Release (MGD)"
    except Exception:
        pass

    try:
        sto = dataD.res_storage[key][0][reservoir].astype(float)
        sto.name = "Default Storage (MG)"
    except Exception:
        pass

    if return_mrf:
        try:
            mrf = dataD.lower_basin_mrf_contributions[key][0][reservoir].astype(float)
            mrf.name = "MRF Contribution (MGD)"
        except Exception:
            mrf = None

    return rel, sto, mrf


def export_default_per_reservoir(
    h5_path: Path,
    outdir: Path,
    save_mrf: bool = True,
) -> None:
    """
    Discover reservoirs from the HDF5, then export default series for each.
    Writes:
      outputs/_default_series/{res}_default_release.csv  (date,value)
      outputs/_default_series/{res}_default_storage.csv  (date,value)
      (optional) outputs/_default_series/{res}_default_mrf.csv (date,value)
    """
    ensure_dir(outdir)
    reservoirs = list_reservoirs_from_h5(h5_path)

    for res in reservoirs:
        rel, sto, mrf = load_default_series_for_reservoir(h5_path, res, return_mrf=save_mrf)

        if rel is not None and not rel.empty:
            df = pd.DataFrame({"date": rel.index.astype("datetime64[ns]").strftime("%Y-%m-%d"), "value": rel.values})
            p = outdir / f"{res}_default_release.csv"
            df.to_csv(p, index=False)
            print(f"[default] wrote {p}  (n={len(df)})")
        else:
            print(f"[default] no default RELEASE for {res} in HDF5")

        if sto is not None and not sto.empty:
            df = pd.DataFrame({"date": sto.index.astype("datetime64[ns]").strftime("%Y-%m-%d"), "value": sto.values})
            p = outdir / f"{res}_default_storage.csv"
            df.to_csv(p, index=False)
            print(f"[default] wrote {p}  (n={len(df)})")
        else:
            print(f"[default] no default STORAGE for {res} in HDF5")

        if save_mrf and (mrf is not None) and not mrf.empty:
            df = pd.DataFrame({"date": mrf.index.astype("datetime64[ns]").strftime("%Y-%m-%d"), "value": mrf.values})
            p = outdir / f"{res}_default_mrf.csv"
            df.to_csv(p, index=False)
            print(f"[default] wrote {p}  (n={len(df)})")
        elif save_mrf:
            print(f"[default] no MRF for {res} (not LB or not recorded)")

# ----------------------------------- CLI --------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Run default Pywr-DRB and export per-reservoir default time series (CSV). No observed data needed."
    )
    ap.add_argument("--start", default="1983-10-01", help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end",   default="2023-12-31", help="End date (YYYY-MM-DD)")
    ap.add_argument("--inflow-type", default="pub_nhmv10_BC_withObsScaled", help="Pywr-DRB inflow_type")
    ap.add_argument("--outdir-default", default=str(DEFAULT_SERIES_DIR), help="Output dir for per-reservoir default CSVs")
    ap.add_argument("--cache-dir", default=str(CACHE_DIR), help="Cache dir for model JSON / HDF5")
    ap.add_argument("--overwrite", action="store_true", help="Force re-run of default model")
    ap.add_argument("--no-mrf", action="store_true", help="Do not attempt to export MRF contributions")
    args = ap.parse_args()

    outdir    = Path(args.outdir_default).resolve()
    cache     = Path(args.cache_dir).resolve()
    ensure_dir(outdir); ensure_dir(cache)

    # 1) Run (or reuse) default model and get HDF5
    h5 = save_default_pywr_run(
        start_date=args.start, end_date=args.end,
        inflow_type=args.inflow_type, outdir=cache,
        overwrite=args.overwrite,
    )

    # 2) Export per-reservoir default release/storage (+ optional MRF) directly from HDF5
    export_default_per_reservoir(h5, outdir, save_mrf=(not args.no_mrf))

    print("\n✅ Done. Your Dash app can now read default series from:")
    print(f"  {outdir}/<reservoir>_default_{{release,storage}}.csv")
    if not args.no_mrf:
        print(f"  (If present) {outdir}/<reservoir>_default_mrf.csv")

if __name__ == "__main__":
    main()
