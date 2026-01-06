import numpy as np
import pandas as pd
from src.assignment import assign_variants
from src.config import ExperimentConfig


def generate_users(n_users: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    user_id = np.arange(1, n_users + 1)
    is_new = rng.binomial(1, 0.70, size=n_users)
    device = rng.choice(["mobile", "desktop"], p=[0.65, 0.35], size=n_users)
    channel = rng.choice(["search", "social"], p=[0.55, 0.45], size=n_users)
    
    users = pd.DataFrame({
        "user_id": user_id,
        "is_new": is_new,
        "device": device,
        "channel": channel,
    })
    return users

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def simulate_signup(users: pd.DataFrame, cfg: ExperimentConfig, seed: int = 42) -> pd.DataFrame:
    df = users.copy()
    rng = np.random.default_rng(seed)

    # Assignment indicator (from variant column)
    df["assigned_B"] = (df["variant"] == "B").astype(int)

    # Non-compliance: only some assigned-to-B actually see B
    df["exposed_B"] = (
        (df["assigned_B"] == 1)
        & (rng.uniform(size=len(df)) < cfg.compliance_rate)
    ).astype(int)

    # ----- Build baseline signup probability (p_signup_base) -----
    # Start from baseline CVR as an intercept in log-odds space
    base_logit = np.log(cfg.baseline_signup_cvr / (1 - cfg.baseline_signup_cvr))

    # Covariate effects (tweak as you like)
    logit = (
        base_logit
        + 0.30 * (df["is_new"].values == 1).astype(float)     # new users slightly more likely
        - 0.20 * (df["device"].values == "mobile").astype(float)  # mobile slightly less likely
        + 0.10 * (df["channel"].values == "search").astype(float) # search slightly higher intent
    )

    df["p_signup_base"] = sigmoid(logit)

    # --- Add Novelty decay: lift shrinks over time (first days stronger, later days weaker) ---
    rng_time = np.random.default_rng(seed + 999)
    
    # assign each user a day in the experiment window (0..D-1)
    D = getattr(cfg, "experiment_days", 28)
    df["day"] = rng_time.integers(0, D, size=len(df))
    
    # exponential decay: lift_day = lift0 * exp(-k * day)
    k = getattr(cfg, "novelty_decay_k", 0.08)  # ~half-life around 8-9 days if k≈0.08
    lift_day = cfg.treatment_abs_lift * np.exp(-k * df["day"].values)
    
    # apply day-specific lift only if actually exposed to B
    df["p_signup"] = np.clip(
        df["p_signup_base"].values + lift_day * df["exposed_B"].values,
        0.0, 1.0
    )

    # ----- Apply treatment effect only to exposed users -----
    df["p_signup"] = np.clip(
        df["p_signup_base"].values + cfg.treatment_abs_lift * df["exposed_B"].values,
        0.0, 1.0
    )

    df["signed_up"] = rng.binomial(1, df["p_signup"].values)
    return df


def simulate_purchase(df, purchase_given_signup=0.20, seed=42):
    rng = np.random.default_rng(seed + 1)
    out = df.copy()
    p_purchase = purchase_given_signup
    
    p_purchase_user = np.clip(
        p_purchase
        + 0.03 * (out["is_new"].values == 0).astype(float)
        - 0.02 * (out["channel"].values == "social").astype(float),
        0.0, 1.0
    )
    out["p_purchase_given_signup"] = p_purchase_user
    out["purchased"] = 0
    mask = out["signed_up"].values == 1
    out.loc[mask, "purchased"] = rng.binomial(1, out.loc[mask, "p_purchase_given_signup"].values)
    return out

def run_experiment(n_users: int, cfg: ExperimentConfig):
    users = generate_users(n_users=n_users, seed=cfg.seed)
    users = assign_variants(users, split=cfg.split)
    df = simulate_signup(
        users,
        cfg=cfg,
        seed=cfg.seed
    )
    df = simulate_purchase(df, purchase_given_signup=cfg.purchase_given_signup, seed=cfg.seed)
    return df
