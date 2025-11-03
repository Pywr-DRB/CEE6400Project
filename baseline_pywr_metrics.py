# baseline_pywr_metrics.py
from pathlib import Path
import numpy as np
import pandas as pd

# ---- Your packages / config ----
import pywrdrb
from methods.load.observations import get_observational_training_data
from methods.metrics.objectives import ObjectiveCalculator
from methods.config import (
    PROCESSED_DATA_DIR, FIG_DIR as CFG_FIG_DIR,
    RELEASE_METRICS, STORAGE_METRICS, EPSILONS,
    reservoir_capacity, INERTIA_BY_RESERVOIR, release_max_by_reservoir
)

# ------------------- user inputs -------------------
RES        = "prompton"                  # reservoir key
VAL_START  = "1983-10-01"                # validation window
VAL_END    = "2023-12-31"
PYWR_START = VAL_START                   # Pywr-DRB model run window
PYWR_END   = VAL_END
INFLOW     = "pub_nhmv10_BC_withObsScaled"      # model inflow choice
#"pub_nhmv10_BC_withObsScaled"
#"nwmv21_withObsScaled"
# ------------------- helpers -------------------
def to_path(p): return p if isinstance(p, Path) else Path(p)

def run_default_starfit(res, pywr_start, pywr_end, inflow_type, work_dir):
    """
    Build & run a default-STARFIT Pywr-DRB model and return Series:
    - R: downstream gage flow (release+spill) [name='default_release']
    - S: storage time series [name='default_storage']
    """
    work_dir   = to_path(work_dir); work_dir.mkdir(parents=True, exist_ok=True)
    tmp_models = (Path(pywrdrb.__file__).resolve().parent.parent / "_tmp_models")
    tmp_models.mkdir(parents=True, exist_ok=True)

    tag = f"Default_STARFIT_{res}"
    mb = pywrdrb.ModelBuilder(start_date=pywr_start, end_date=pywr_end, inflow_type=inflow_type)
    mb.make_model()
    model_json = tmp_models / f"model_{tag}.json"
    mb.write_model(str(model_json))

    h5 = work_dir / f"output_{tag}.hdf5"
    model = pywrdrb.Model.load(str(model_json))
    rec   = pywrdrb.OutputRecorder(model, str(h5))
    _    = model.run()

    dataP = pywrdrb.Data(
        print_status=False,
        results_sets=["res_storage","reservoir_downstream_gage"],
        output_filenames=[str(h5)]
    )
    dataP.load_output(); key = h5.stem

    # downstream gage already includes spill; rename for clarity
    R = dataP.reservoir_downstream_gage[key][0][res].astype(float).rename("pywr_release")
    S = dataP.res_storage[key][0][res].astype(float).rename("pywr_storage")
    return R, S, h5

def align_series(*series, start=None, end=None):
    """Intersect indices and optional slice to [start:end]."""
    idx = series[0].index
    for s in series[1:]:
        idx = idx.intersection(s.index)
    if start is not None: idx = idx[idx >= pd.to_datetime(start)]
    if end   is not None: idx = idx[idx <= pd.to_datetime(end)]
    return [s.loc[idx] for s in series]

# ------------------- 1) run Pywr-DRB baseline -------------------
R_def_full, S_def_full, _h5 = run_default_starfit(
    RES, PYWR_START, PYWR_END, INFLOW, work_dir=(to_path(CFG_FIG_DIR) / "_starfit_default")
)

# ------------------- 2) load observations for the same reservoir -------------------
# Note: inflow_type here is only for loading inflow; ObjectiveCalculator uses release/storage.
inflow_obs_df, release_obs_df, storage_obs_df = get_observational_training_data(
    reservoir_name=RES, data_dir=PROCESSED_DATA_DIR, as_numpy=False, inflow_type="inflow_pub"
)

# Rename for clarity
R_obs_full = release_obs_df.squeeze().astype(float).rename("obs_release")
S_obs_full = storage_obs_df.squeeze().astype(float).rename("obs_storage")

# ------------------- 3) align on dates -------------------
R_obs, S_obs, R_def, S_def = align_series(
    R_obs_full, S_obs_full, R_def_full, S_def_full, start=VAL_START, end=VAL_END
)

# quick sanity checks
assert len(R_obs) == len(R_def) == len(S_obs) == len(S_def) and len(R_obs) > 0, "Date alignment failed."

# ------------------- 4) wire up ObjectiveCalculator exactly like optimization -------------------
cap   = reservoir_capacity[RES]
R_MAX = release_max_by_reservoir[RES]
iset  = INERTIA_BY_RESERVOIR[RES]

release_obj = ObjectiveCalculator(
    metrics=RELEASE_METRICS,
    inertia_tau=iset["release"]["tau"],
    inertia_scale_release=iset["release"]["scale"],
    inertia_release_scale_value=(R_MAX if iset["release"]["scale"] == "value" else None),
)

storage_obj = ObjectiveCalculator(
    metrics=STORAGE_METRICS,
    capacity_mg=reservoir_capacity[RES],
    inertia_tau=iset["storage"]["tau"],
    inertia_scale_storage=iset["storage"]["scale"],
    inertia_storage_scale_value=iset["storage"]["scale_value"],
)

# ------------------- 5) compute metrics -------------------
release_vals = release_obj.calculate(obs=R_obs.values.astype(np.float64),
                                     sim=R_def.values.astype(np.float64))
storage_vals = storage_obj.calculate(obs=S_obs.values.astype(np.float64),
                                     sim=S_def.values.astype(np.float64))

# build tidy dataframe to match MOEA metric order
metric_names = RELEASE_METRICS + STORAGE_METRICS
metric_vals  = release_vals + storage_vals
baseline_df  = pd.DataFrame({"metric": metric_names, "pywr_baseline": metric_vals})

# Optional: include simple reference stats for transparency
summary = pd.DataFrame({
    "series": ["obs_release","pywr_release","obs_storage","pywr_storage"],
    "start":  [R_obs.index.min()]*2 + [S_obs.index.min()]*2,
    "end":    [R_obs.index.max()]*2 + [S_obs.index.max()]*2,
    "n_days": [len(R_obs), len(R_def), len(S_obs), len(S_def)],
})

# ------------------- 6) print/save -------------------
pd.set_option("display.precision", 6)
print("\n=== Pywr-DRB Baseline Objective Values (same calculators as MOEA) ===")
#add reservoir name to header
print(f"--- Reservoir: {RES} ---")
print(baseline_df.to_string(index=False))

print("\n--- Alignment summary ---")
print(summary.to_string(index=False))

# Save if you want to join later with .set/.runtime analysis
out_dir = to_path(CFG_FIG_DIR) / f"_starfit_default_{INFLOW}"
out_dir.mkdir(parents=True, exist_ok=True)
baseline_df.to_csv(out_dir / f"baseline_objectives_{RES}_{VAL_START}_to_{VAL_END}.csv", index=False)
