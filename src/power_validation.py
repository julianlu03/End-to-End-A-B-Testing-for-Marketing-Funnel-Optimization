import numpy as np
import pandas as pd

from src.simulate import run_experiment
from src.analysis import run_ab_analysis
from src.config import ExperimentConfig

def run_one_sim(n_users: int, cfg: ExperimentConfig, true_lift: float, alpha: float = 0.05):
    df = run_experiment(n_users=n_users, cfg=cfg)
    res = run_ab_analysis(df, outcome_col="signed_up")

    lift = res["lift_ci"]["lift"]
    ci_low = res["lift_ci"]["ci_low"]
    ci_high = res["lift_ci"]["ci_high"]
    pval = res["test"]["p_value"]
    reject = pval < alpha  # one-sided since your test_lift default is "greater"
    covers = (ci_low <= true_lift) and (true_lift <= ci_high)

    return {
        "n_users": n_users,
        "lift_hat": lift,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_width": ci_high - ci_low,
        "p_value": pval,
        "reject": reject,
        "covers": covers,
        "srm_passes": res["srm"]["passes"],
    }


def run_phase5(
    n_sims: int,
    n_users_grid: list[int],
    cfg_base: ExperimentConfig,
    true_lift: float,
    alpha: float = 0.05,
):
    """
    Runs repeated simulated experiments across multiple sample sizes.
    Returns:
      - raw_df: one row per simulated experiment
      - summary_df: aggregated metrics by sample size
    """
    rows = []
    for n_users in n_users_grid:
        for i in range(n_sims):
            # vary seed each run for independence
            cfg_i = cfg_base.__class__(**{**cfg_base.__dict__, "seed": cfg_base.seed + (10_000 * n_users) + i})
            rows.append(run_one_sim(n_users=n_users, cfg=cfg_i, true_lift=true_lift, alpha=alpha))

    raw_df = pd.DataFrame(rows)

    summary_df = (
        raw_df.groupby("n_users")
        .agg(
            power=("reject", "mean"),
            mean_lift=("lift_hat", "mean"),
            std_lift=("lift_hat", "std"),
            mean_ci_width=("ci_width", "mean"),
            coverage=("covers", "mean"),
            srm_pass_rate=("srm_passes", "mean"),
        )
        .reset_index()
    )

    return raw_df, summary_df
