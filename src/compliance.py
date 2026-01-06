from src.config import ExperimentConfig
from src.simulate import run_experiment
from src.analysis import run_ab_analysis
import pandas as pd

def run_once(compliance):
    cfg = ExperimentConfig(
        baseline_signup_cvr=0.12,
        treatment_abs_lift=0.012,
        compliance_rate=compliance,
        seed=123
    )
    df = run_experiment(n_users=200_000, cfg=cfg)
    res = run_ab_analysis(df, outcome_col="signed_up")

    exposure_rate = df.loc[df["variant"]=="B", "exposed_B"].mean()
    predicted_lift = exposure_rate * cfg.treatment_abs_lift

    return {
        "compliance": compliance,
        "exposure_rate_realized": exposure_rate,
        "lift_hat": res["lift_ci"]["lift"],
        "predicted_lift": predicted_lift,
        "ci_low": res["lift_ci"]["ci_low"],
        "ci_high": res["lift_ci"]["ci_high"],
        "p_value": res["test"]["p_value"]
    }


def power_at_compliance(compliance, n_sims=300, n_users=50_000):
    rejects = []
    for i in range(n_sims):
        cfg = ExperimentConfig(
            baseline_signup_cvr=0.12,
            treatment_abs_lift=0.012,
            compliance_rate=compliance,
            seed=1000 + i
        )
        df = run_experiment(n_users=n_users, cfg=cfg)
        res = run_ab_analysis(df, outcome_col="signed_up")
        rejects.append(res["test"]["p_value"] < 0.05)
    return sum(rejects) / len(rejects)

rows = []
for c in [1.0, 0.9, 0.8, 0.7, 0.6]:
    rows.append({"compliance": c, "power": power_at_compliance(c)})

pd.DataFrame(rows)