# methods/plotting/styles.py
policy_type_colors = {'PWL': 'blue', 'RBF': 'orange', 'STARFIT': 'green', 'Baseline': 'black'}


ADVANCED_COLORS = {
    "Best Average (All Objectives)": "#1f77b4",  # <-- add this line
    "Compromise L2 (Euclidean)": "#7b6cff",
    "Tchebycheff L∞": "#c266ff",
    "Manhattan L1": "#ff66c4",
    "Knee (max curvature)": "#ff914d",
    "ε-constraint Release NSE ≥ Q50": "#00c2a8",
    "ε-constraint Release NSE ≥ Q80": "#008eaa",
    "Diverse #1 (FPS)": "#8bd3dd",
    "Diverse #2 (FPS)": "#a0e7e5",
    "Max HV Contribution": "#ffd166",
}