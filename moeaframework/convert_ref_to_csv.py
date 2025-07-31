import os
import pandas as pd

# === Header definitions ===
variable_names = {
    "STARFIT": [
        "NORhi_mu", "NORhi_min", "NORhi_max", "NORhi_alpha", "NORhi_beta",
        "NORlo_mu", "NORlo_min", "NORlo_max", "NORlo_alpha", "NORlo_beta",
        "Release_alpha1", "Release_alpha2", "Release_beta1", "Release_beta2",
        "Release_c", "Release_p1", "Release_p2"
    ],
    "RBF": [
        "rbf1_center_inflow", "rbf1_center_storage", "rbf1_center_doy",
        "rbf1_scale_inflow", "rbf1_scale_storage", "rbf1_scale_doy", "rbf1_weight",
        "rbf2_center_inflow", "rbf2_center_storage", "rbf2_center_doy",
        "rbf2_scale_inflow", "rbf2_scale_storage", "rbf2_scale_doy", "rbf2_weight"
    ],
    "PiecewiseLinear": [
        "storage_x1", "storage_x2", "storage_theta1", "storage_theta2", "storage_theta3",
        "inflow_x1", "inflow_x2", "inflow_theta1", "inflow_theta2", "inflow_theta3",
        "season_x1", "season_x2", "season_theta1", "season_theta2", "season_theta3"
    ]
}

objective_labels = [
    "Release_NSE", "Release_q20_Abs_PBias", "Release_q80_Abs_PBias", "Storage_NSE"
]

reservoir_data = {
    'prompton': {'GRanD_CAP_MG': None, 'GRanD_MEANFLOW_MGD': None,
                 'Adjusted_CAP_MG': 27956.02, 'Adjusted_MEANFLOW_MGD': 83.49189416,
                 'Max_release': 231.6065144,'Release_max': 1.774, 'Release_min': -0.804},
    'beltzvilleCombined': {'GRanD_CAP_MG': 48317.0588, 'GRanD_MEANFLOW_MGD': 116.5417199,
                 'Adjusted_CAP_MG': 48317.0588, 'Adjusted_MEANFLOW_MGD': 116.5417199,
                 'Max_release': 969.5, 'Release_max': 1.878, 'Release_min': -0.807},
    'fewalter': {'GRanD_CAP_MG': None, 'GRanD_MEANFLOW_MGD': None,
                 'Adjusted_CAP_MG': 35800.0, 'Adjusted_MEANFLOW_MGD': 137.23,
                 'Max_release': 1292.6, 'Release_max': 1.774, 'Release_min': -0.804}
}

policy_structure = {
    "STARFIT": {"n_dvs": 17, "n_objs": 4, "n_constraints": 1},
    "RBF": {"n_dvs": 14, "n_objs": 4, "n_constraints": 0},
    "PiecewiseLinear": {"n_dvs": 15, "n_objs": 4, "n_constraints": 0},
}

def parse_info_from_path(filepath):
    parts = filepath.split(os.sep)
    try:
        policy_folder = next(p for p in parts if p.startswith("Policy_"))
        policy = policy_folder.replace("Policy_", "")
        reservoir = parts[-2]
        return policy, reservoir
    except StopIteration:
        return None, None

def convert_ref_file_to_csv(filepath):
    policy, reservoir = parse_info_from_path(filepath)
    if policy not in policy_structure or reservoir not in reservoir_data:
        print(f"Skipping {filepath} (could not parse policy/reservoir)")
        return

    print(f"Processing: {policy}, {reservoir}")
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if not line.startswith("#") and line.strip()]

    n_dvs = policy_structure[policy]["n_dvs"]
    n_objs = policy_structure[policy]["n_objs"]
    n_constraints = policy_structure[policy]["n_constraints"]

    data = [list(map(float, line.split())) for line in lines]
    df = pd.DataFrame(data)

    headers = (
        variable_names[policy] +
        objective_labels[:n_objs] +
        (["constraint1"] if n_constraints == 1 else [])
    )
    df.columns = headers

    # === Assign and label policy_id for both outputs ===
    df["policy_id"] = list(range(len(df)))

    # Compute composite score
    df_norm = df[objective_labels[:n_objs]].copy()
    if "Release_q20_Abs_PBias" in df_norm.columns:
        df_norm["Release_q20_Abs_PBias"] *= -1
    if "Release_q80_Abs_PBias" in df_norm.columns:
        df_norm["Release_q80_Abs_PBias"] *= -1
    df_norm = (df_norm - df_norm.min()) / (df_norm.max() - df_norm.min())
    df["composite_score"] = df_norm.mean(axis=1)

    # Identify best indices
    idx_release = df["Release_NSE"].idxmax()
    idx_storage = df["Storage_NSE"].idxmax()
    idx_q20 = df["Release_q20_Abs_PBias"].idxmin()
    idx_q80 = df["Release_q80_Abs_PBias"].idxmin()
    idx_composite = df["composite_score"].idxmax()

    label_map = {
        idx_release: "Best Release NSE",
        idx_storage: "Best Storage NSE",
        idx_q20: "Best q20 Bias",
        idx_q80: "Best q80 Bias",
        idx_composite: "Best Overall"
    }

    for idx, label in label_map.items():
        df.loc[idx, "policy_id"] = label

    # === Save renamed CSV with policy_id ===
    renamed_cols = headers + ["policy_id"]
    df[renamed_cols].to_csv(f"{policy}_{reservoir}_renamed.csv", index=False)
    print(f"Saved renamed CSV to {policy}_{reservoir}_renamed.csv")

    # === Save DRB CSV ===
    drb_df = df[variable_names[policy] + ["policy_id"]].copy()
    drb_df["reservoir"] = reservoir

    meta = reservoir_data[reservoir]
    for field in meta:
        drb_df[field] = meta[field]

    final_cols = ["reservoir", "policy_id"] + variable_names[policy] + list(meta.keys())
    drb_df = drb_df[final_cols]
    drb_df.to_csv(f"{policy}_{reservoir}_drb.csv", index=False)
    print(f"Saved DRB-formatted CSV to {policy}_{reservoir}_drb.csv")

    # Append to master DRB data store
    all_drb_data[policy].append(drb_df)

# === Dictionary to store DRB DataFrames for each policy ===
all_drb_data = {
    "STARFIT": [],
    "RBF": [],
    "PWL": []
}



# === Execution ===
if __name__ == "__main__":
    base_dir = "outputs"
    print(f"Searching for borg.ref files in: {base_dir}")
    
    any_found = False
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file == "borg.ref":
                any_found = True
                filepath = os.path.join(root, file)
                print(f"\nFound file: {filepath}")
                convert_ref_file_to_csv(filepath)
    
    if not any_found:
        print("No borg.ref files found in the outputs directory.")
    else:
        print("\nFinished converting all borg.ref files.\n")
    
    # === Save aggregated CSV per policy ===
    for policy, dfs in all_drb_data.items():
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            combined_df.to_csv(f"{policy}_all_reservoirs_drb.csv", index=False)
            print(f" Saved aggregated DRB CSV for {policy}: {policy}_all_reservoirs_drb.csv")
