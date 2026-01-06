import pandas as pd

from src.simulate import run_experiment
from src.analysis import run_ab_analysis

# Functions to run A/A test

# One A/A validation test
def run_aa_once(n_users: int, cfg):
    df = run_experiment(n_users=n_users, cfg=cfg)
    res = run_ab_analysis(df, outcome_col="signed_up")
    return {
        "p_value": res["test"]["p_value"],
        "ci_low": res["lift_ci"]["ci_low"],
        "ci_high": res["lift_ci"]["ci_high"],
        "reject": res["test"]["p_value"] < 0.05,
        "ci_excludes_zero": (res["lift_ci"]["ci_low"] > 0) or (res["lift_ci"]["ci_high"] < 0),
        "srm_passes": res["srm"]["passes"],
    }

# Run many times
def run_aa_simulation(
    n_sims: int,
    n_users: int,
    cfg
):
    rows = []
    for i in range(n_sims):
        # change seed each time
        cfg_i = cfg.__class__(**{**cfg.__dict__, "seed": cfg.seed + i})
        rows.append(run_aa_once(n_users, cfg_i))
    return pd.DataFrame(rows)