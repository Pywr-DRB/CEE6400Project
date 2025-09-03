#!/usr/bin/env python3
import os
import sys
import glob
import math
import pandas as pd

# =========================
# Configuration / Metadata
# =========================

variable_names = {
    "STARFIT": [
        "NORhi_mu", "NORhi_min", "NORhi_max", "NORhi_alpha", "NORhi_beta",
        "NORlo_mu", "NORlo_min", "NORlo_max", "NORlo_alpha", "NORlo_beta",
        "Release_alpha1", "Release_alpha2", "Release_beta1", "Release_beta2",
        "Release_c", "Release_p1", "Release_p2"
    ],
    "RBF": [
        "rbf1_center_storage", "rbf1_center_inflow", "rbf1_center_doy",
        "rbf1_scale_storage",  "rbf1_scale_inflow",  "rbf1_scale_doy", "rbf1_weight",
        "rbf2_center_storage", "rbf2_center_inflow", "rbf2_center_doy",
        "rbf2_scale_storage",  "rbf2_scale_inflow",  "rbf2_scale_doy", "rbf2_weight"
    ],
    "PiecewiseLinear": [
        "storage_x1", "storage_x2", "storage_theta1", "storage_theta2", "storage_theta3",
        "inflow_x1",  "inflow_x2",  "inflow_theta1",  "inflow_theta2",  "inflow_theta3",
        "season_x1",  "season_x2",  "season_theta1",  "season_theta2",  "season_theta3"
    ]
}

objective_labels = ["Release_NSE", "Release_q20_Abs_PBias", "Release_q80_Abs_PBias", "Storage_NSE"]

# Reservoir meta appended to DRB CSVs (adjust as needed)
reservoir_data = {
    'prompton': {
        'GRanD_CAP_MG': None, 'GRanD_MEANFLOW_MGD': None,
        'Adjusted_CAP_MG': 27956.02, 'Adjusted_MEANFLOW_MGD': 83.49189416,
        'Max_release': 231.6065144, 'Release_max': 1.774, 'Release_min': -0.804,
        'I_min': 0.0, 'I_max': 900.585
    },
    'beltzvilleCombined': {
        'GRanD_CAP_MG': 48317.0588, 'GRanD_MEANFLOW_MGD': 116.5417199,
        'Adjusted_CAP_MG': 48317.0588, 'Adjusted_MEANFLOW_MGD': 116.5417199,
        'Max_release': 969.5, 'Release_max': 1.878, 'Release_min': -0.807,
        'I_min': 0.0, 'I_max': 1483.45
    },
    'fewalter': {
        'GRanD_CAP_MG': None, 'GRanD_MEANFLOW_MGD': None,
        'Adjusted_CAP_MG': 35800.0, 'Adjusted_MEANFLOW_MGD': 137.23,
        'Max_release': 1292.6, 'Release_max': 1.774, 'Release_min': -0.804,
        'I_min': 0.0, 'I_max': 2168.7
    }
}

# Counts must match how MOEA 5.0 writes lines: [DVs | Objectives | Constraints]
policy_structure = {
    "STARFIT":        {"n_dvs": 17, "n_objs": 4, "n_constraints": 1},
    "RBF":            {"n_dvs": 14, "n_objs": 4, "n_constraints": 0},
    "PiecewiseLinear":{"n_dvs": 15, "n_objs": 4, "n_constraints": 0},
}

# Toggle whether to process per-seed refs in addition to borg.ref
PROCESS_SEED_REFS = False

# Where to start searching
BASE_DIR = "outputs"


# =========================
# Helpers
# =========================

def parse_info_from_path(filepath: str):
    """
    Expect path like: outputs/Policy_<POLICY>/runtime/<reservoir>/[borg.ref|seedN.ref]
    """
    parts = filepath.split(os.sep)
    policy = None
    reservoir = None
    for p in parts:
        if p.startswith("Policy_"):
            policy = p.replace("Policy_", "")
            break
    if policy is None:
        return None, None
    # reservoir folder is the immediate parent
    if len(parts) >= 2:
        reservoir = parts[-2]
    return policy, reservoir


def read_numeric_rows(ref_path: str):
    """Read whitespace-delimited numeric rows; skip comments (#) and blanks."""
    rows = []
    with open(ref_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # tolerate multiple spaces / tabs
            parts = line.split()
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                raise ValueError(f"[ERROR] Non-numeric token in {ref_path}: {parts}")
    if not rows:
        raise ValueError(f"[ERROR] No numeric rows found in {ref_path}")
    return rows


def safe_minmax(df: pd.DataFrame) -> pd.DataFrame:
    """(x - min) / (max - min), but avoid division-by-zero by replacing 0 range with 1."""
    mins = df.min()
    maxs = df.max()
    rng = maxs - mins
    rng = rng.replace(0, 1.0)
    return (df - mins) / rng


def dataframe_from_ref(ref_path: str, policy: str, reservoir: str) -> pd.DataFrame:
    if policy not in policy_structure:
        raise KeyError(f"[ERROR] Unknown policy in path: {policy}")
    if reservoir not in reservoir_data:
        raise KeyError(f"[ERROR] Unknown reservoir in path: {reservoir}")

    rows = read_numeric_rows(ref_path)
    df = pd.DataFrame(rows)
    n_cols = df.shape[1]

    spec = policy_structure[policy]
    n_dvs = spec["n_dvs"]
    n_objs = spec["n_objs"]
    n_cons = spec["n_constraints"]

    expected_cols = n_dvs + n_objs + n_cons
    if n_cols != expected_cols:
        raise ValueError(
            f"[ERROR] Column mismatch in {ref_path} ({policy}/{reservoir}). "
            f"Got {n_cols}, expected {expected_cols} = DVs({n_dvs}) + Objs({n_objs}) + Cons({n_cons})."
        )

    headers = (
        variable_names[policy] +
        objective_labels[:n_objs] +
        (["constraint1"] if n_cons == 1 else [])
    )
    df.columns = headers
    return df


def score_and_label(df: pd.DataFrame, n_objs: int) -> pd.DataFrame:
    # Add a numeric policy_id first
    df = df.copy()
    df["policy_id"] = list(range(len(df)))

    # Composite score: NSE terms higher is better; bias terms lower (so multiply by -1 first)
    obj_cols = objective_labels[:n_objs]
    df_norm = df[obj_cols].copy()
    for col in ("Release_q20_Abs_PBias", "Release_q80_Abs_PBias"):
        if col in df_norm.columns:
            df_norm[col] = -df_norm[col]

    df_norm = safe_minmax(df_norm)
    df["composite_score"] = df_norm.mean(axis=1)

    # Pick best indices (only if present)
    def idx_best(col, kind="max"):
        if col not in df.columns:
            return None
        if kind == "max":
            return df[col].idxmax()
        return df[col].idxmin()

    picks = {
        idx_best("Release_NSE", "max"): "Best Release NSE",
        idx_best("Storage_NSE", "max"): "Best Storage NSE",
        idx_best("Release_q20_Abs_PBias", "min"): "Best q20 Bias",
        idx_best("Release_q80_Abs_PBias", "min"): "Best q80 Bias",
        idx_best("composite_score", "max"): "Best Overall",
    }

    for idx, label in picks.items():
        if idx is not None and not (isinstance(idx, float) and math.isnan(idx)):
            df.loc[idx, "policy_id"] = label

    return df


def write_outputs(df: pd.DataFrame, policy: str, reservoir: str, suffix: str = ""):
    """
    suffix: "", or e.g. "_seed3"
    Writes <policy>_<reservoir><suffix>_renamed.csv and _drb.csv
    """
    spec = policy_structure[policy]
    n_objs = spec["n_objs"]
    n_cons = spec["n_constraints"]

    headers = (
        variable_names[policy] +
        objective_labels[:n_objs] +
        (["constraint1"] if n_cons == 1 else [])
    )

    # renamed CSV (DVs + objectives [+ constraint] + policy_id)
    renamed_cols = headers + ["policy_id"]
    renamed_path = f"{policy}_{reservoir}{suffix}_renamed.csv"
    df[renamed_cols].to_csv(renamed_path, index=False)
    print(f"  - wrote {renamed_path}")

    # DRB CSV (DVs + policy_id + reservoir metadata)
    drb_df = df[variable_names[policy] + ["policy_id"]].copy()
    drb_df["reservoir"] = reservoir
    meta = reservoir_data[reservoir]
    for k, v in meta.items():
        drb_df[k] = v

    final_cols = ["reservoir", "policy_id"] + variable_names[policy] + list(meta.keys())
    drb_df = drb_df[final_cols]

    drb_path = f"{policy}_{reservoir}{suffix}_drb.csv"
    drb_df.to_csv(drb_path, index=False)
    print(f"  - wrote {drb_path}")

    return drb_df


# =========================
# Main
# =========================

if __name__ == "__main__":
    base_dir = BASE_DIR
    print(f"Searching for refs under: {base_dir}")

    # Storage for aggregated DRB CSVs per policy
    all_drb_data = { "STARFIT": [], "RBF": [], "PiecewiseLinear": [] }

    found_any = False

    for root, _, _ in os.walk(base_dir):
        # process global borg.ref if present
        borg_path = os.path.join(root, "borg.ref")
        targets = []
        if os.path.isfile(borg_path):
            targets.append(("borg", borg_path))

        # optionally process seed*.ref in the same folder
        if PROCESS_SEED_REFS:
            for p in glob.glob(os.path.join(root, "seed*.ref")):
                name = os.path.splitext(os.path.basename(p))[0]  # seedN
                targets.append((name, p))

        for label, ref_path in targets:
            policy, reservoir = parse_info_from_path(ref_path)
            if not policy or not reservoir:
                continue

            try:
                df = dataframe_from_ref(ref_path, policy, reservoir)
            except Exception as e:
                print(e)
                continue

            found_any = True
            df = score_and_label(df, policy_structure[policy]["n_objs"])

            # suffix is "" for borg, or "_seedN" for seed refs
            suffix = "" if label == "borg" else f"_{label}"
            drb_df = write_outputs(df, policy, reservoir, suffix=suffix)

            # only aggregate global borg refs (to avoid duplicate seed rows)
            if label == "borg":
                all_drb_data[policy].append(drb_df)

    if not found_any:
        print("No refs found (borg.ref or seed*.ref).")
        sys.exit(0)

    # Write aggregated DRB CSV per policy across reservoirs
    for policy, dfs in all_drb_data.items():
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            out_path = f"{policy}_all_reservoirs_drb.csv"
            combined.to_csv(out_path, index=False)
            print(f"[AGG] wrote {out_path}")

    print("\nDone.\n")
