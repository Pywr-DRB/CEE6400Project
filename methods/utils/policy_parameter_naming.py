import pandas as pd
import re




# ---------------- config & project imports ----------------
from methods.config import (
n_rbfs, n_rbf_inputs, n_segments, n_pwl_inputs
)

def safe_name(label: str) -> str:
    """Turn any label into a filesystem-friendly token."""
    s = re.sub(r'[^A-Za-z0-9._-]+', '_', str(label))
    s = re.sub(r'_+', '_', s).strip('_')
    return s if s else "pick"

def get_param_names_for_policy(policy: str):
    """Return ordered parameter names matching the CSV var order for each policy."""
    policy = str(policy).upper()
    if policy == "STARFIT":
        return [
            "NORhi_mu", "NORhi_min", "NORhi_max", "NORhi_alpha", "NORhi_beta",
            "NORlo_mu", "NORlo_min", "NORlo_max", "NORlo_alpha", "NORlo_beta",
            "Release_alpha1", "Release_alpha2", "Release_beta1", "Release_beta2",
            "Release_c", "Release_p1", "Release_p2",
        ]
    if policy == "RBF":
        labels = ["storage", "inflow", "doy"][:n_rbf_inputs]
        names = []
        # weights
        for i in range(1, n_rbfs + 1):
            names.append(f"w{i}")
        # centers
        for i in range(1, n_rbfs + 1):
            for v in labels:
                names.append(f"c{i}_{v}")
        # radii/scales
        for i in range(1, n_rbfs + 1):
            for v in labels:
                names.append(f"r{i}_{v}")
        return names
    if policy == "PWL":
        names = []
        block_labels = ["storage", "inflow", "day"][:n_pwl_inputs]
        for lab in block_labels:
            for k in range(1, n_segments):
                names.append(f"{lab}_x{k}")
            for k in range(1, n_segments + 1):
                names.append(f"{lab}_theta{k}")
        return names
    raise ValueError(f"Unknown policy '{policy}'")

def rename_vars_with_param_names(var_df: pd.DataFrame, policy_type: str) -> pd.DataFrame:
    """Rename var1..varN to policy parameter names. Leaves extra columns as var* if any."""
    if var_df is None or var_df.empty:
        return var_df
    out = var_df.copy()
    # Identify var* columns in order
    var_cols = [c for c in out.columns if c.lower().startswith("var")]
    # Keep order as in CSV (var1, var2, ...)
    var_cols_sorted = sorted(var_cols, key=lambda x: int(re.sub(r'[^0-9]', '', x) or 0))
    names = get_param_names_for_policy(policy_type)
    k = min(len(names), len(var_cols_sorted))
    # Map first k vars to names; keep any extras as-is
    rename_map = {var_cols_sorted[i]: names[i] for i in range(k)}
    out.rename(columns=rename_map, inplace=True)
    return out

def print_params_flat(policy_type: str, params_1d):
    """Flat index → name → value (works for all policies)."""
    names = get_param_names_for_policy(policy_type)
    assert len(params_1d) == len(names), f"Length mismatch: got {len(params_1d)} values, expected {len(names)}"
    print(f"\n--- Parameters ({policy_type}) ---")
    for i, (n, v) in enumerate(zip(names, params_1d)):
        print(f"[{i:02d}] {n:16s} = {float(v): .6f}")

def print_params_pretty(policy_type: str, params_1d):
    """Grouped printing for RBF/PWL; STARFIT remains flat."""
    policy = policy_type.upper()
    names = get_param_names_for_policy(policy)
    assert len(params_1d) == len(names), f"Length mismatch: got {len(params_1d)} values, expected {len(names)}"

    if policy == "STARFIT":
        print_params_flat(policy, params_1d)
        return

    if policy == "RBF":
        print(f"\n--- Parameters (RBF) n_rbfs={n_rbfs}, n_inputs={n_rbf_inputs} ---")
        idx = 0
        print("Weights:")
        for i in range(1, n_rbfs + 1):
            print(f"  w{i} = {float(params_1d[idx]): .6f}")
            idx += 1
        print("Centers c[i, var]:")
        for i in range(1, n_rbfs + 1):
            row = []
            for var in ["storage", "inflow", "doy"][:n_rbf_inputs]:
                row.append(float(params_1d[idx])); idx += 1
            print(f"  c{i} = {row}")
        print("Scales r[i, var]:")
        for i in range(1, n_rbfs + 1):
            row = []
            for var in ["storage", "inflow", "doy"][:n_rbf_inputs]:
                row.append(float(params_1d[idx])); idx += 1
            print(f"  r{i} = {row}")
        return

    if policy == "PWL":
        print(f"\n--- Parameters (PWL) n_segments={n_segments}, n_inputs={n_pwl_inputs} ---")
        per_block = 2 * n_segments - 1
        blocks = ["storage", "inflow", "day"][:n_pwl_inputs]
        for b, lab in enumerate(blocks):
            block = params_1d[b*per_block:(b+1)*per_block]
            xs     = block[:n_segments-1]
            thetas = block[n_segments-1:]
            print(f"{lab.capitalize()} block:")
            for i, x in enumerate(xs, start=1):
                print(f"  x{i}     = {float(x): .6f}")
            for i, th in enumerate(thetas, start=1):
                print(f"  theta{i} = {float(th): .6f}")
        return

    print_params_flat(policy, params_1d)

def has_solutions(solution_objs, reservoir_name: str, policy_type: str) -> bool:
    return (
        reservoir_name in solution_objs and
        policy_type in solution_objs[reservoir_name] and
        solution_objs[reservoir_name][policy_type] is not None and
        len(solution_objs[reservoir_name][policy_type]) > 0
    )

def reservoir_has_any(solution_objs, reservoir_name: str) -> bool:
    d = solution_objs.get(reservoir_name, {})
    return any((df is not None) and (len(df) > 0) for df in d.values())

def _params_for_row(var_df: pd.DataFrame, row_idx):
    """Row-safe accessor: prefer .loc by index label, fall back to .iloc if needed."""
    try:
        return var_df.loc[row_idx].values
    except Exception:
        return var_df.iloc[int(row_idx)].values