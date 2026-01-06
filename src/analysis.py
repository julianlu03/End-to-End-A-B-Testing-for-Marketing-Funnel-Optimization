from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats

@dataclass(frozen=True)
class GroupStats:
    n_A: int    # users in A
    n_B: int    # users in B
    conv_A: int # number of users in A who converted
    conv_B: int # number of users in B who converted
    p_A: float  # signup rate in A
    p_B: float  # signup rate in B



def compute_group_stats(
    df: pd.DataFrame,
    group_col: str = "variant",
    outcome_col: str = "signed_up",
    group_A: str = "A",
    group_B: str = "B",
) -> GroupStats:
    """
    Compute sample sizes and conversions per arm.
    Assumes outcome is binary 0/1.
    """
    if group_col not in df.columns or outcome_col not in df.columns:
        raise KeyError(f"Expected columns '{group_col}' and '{outcome_col}' in df.")

    dA = df[df[group_col] == group_A]
    dB = df[df[group_col] == group_B]

    n_A = int(len(dA))
    n_B = int(len(dB))
    if n_A == 0 or n_B == 0:
        raise ValueError("One of the groups has 0 rows. Check your group labels.")

    conv_A = int(dA[outcome_col].sum())
    conv_B = int(dB[outcome_col].sum())

    p_A = conv_A / n_A
    p_B = conv_B / n_B

    return GroupStats(n_A=n_A, n_B=n_B, conv_A=conv_A, conv_B=conv_B, p_A=p_A, p_B=p_B)

def estimate_lift(stats_: GroupStats, alpha: float = 0.05) -> dict:
    """
    Difference in proportions (B - A) with Wald CI (normal approximation).
    Returns dict with lift, SE, CI.
    """
    lift = stats_.p_B - stats_.p_A # absolute lift calculation

    # Standard error for difference in independent proportions
    se = np.sqrt(
        stats_.p_A * (1 - stats_.p_A) / stats_.n_A
        + stats_.p_B * (1 - stats_.p_B) / stats_.n_B
    )

    zcrit = stats.norm.ppf(1 - alpha / 2)  # 1.96 for 95% CI
    ci_low = lift - zcrit * se # CI calculations
    ci_high = lift + zcrit * se

    return {
        "lift": lift,
        "se": se,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "alpha": alpha,
        "conf_level": 1 - alpha,
    }

def test_lift(stats_: GroupStats, alternative: str = "greater") -> dict:
    """
    Hypothesis test for difference in proportions using a z-test.
    H0: p_B - p_A = 0
    alternative:
      - 'greater' (p_B > p_A)  [one-sided]
      - 'less'
      - 'two-sided'
    Uses pooled SE under H0 (standard two-proportion z-test).
    """
    lift = stats_.p_B - stats_.p_A
    # Pooled conversion rate for Hypothesis testing
    pooled = (stats_.conv_A + stats_.conv_B) / (stats_.n_A + stats_.n_B)
    se0 = np.sqrt(pooled * (1 - pooled) * (1 / stats_.n_A + 1 / stats_.n_B)) # Standard error for pooled (assuming p_A = p_B)

    if se0 == 0:
        raise ValueError("Pooled standard error is 0. Check if outcomes are constant.")

    z = lift / se0

    if alternative == "greater":
        pval = 1 - stats.norm.cdf(z)
    elif alternative == "less":
        pval = stats.norm.cdf(z)
    elif alternative in ("two-sided", "two_sided", "twosided"):
        pval = 2 * (1 - stats.norm.cdf(abs(z)))
    else:
        raise ValueError("alternative must be 'greater', 'less', or 'two-sided'.")

    return {
        "z": z,
        "p_value": pval,
        "alternative": alternative,
        "pooled_rate": pooled,
        "se0": se0,
        "lift": lift,
    }

def check_srm(
    df: pd.DataFrame,
    group_col: str = "variant",
    expected_split: tuple[float, float] = (0.5, 0.5),
    group_labels: tuple[str, str] = ("A", "B"),
) -> dict:
    """
    Sample Ratio Mismatch (SRM) check via chi-square goodness-of-fit.
    H0: observed counts match expected split.
    """
    gA, gB = group_labels
    n_A = int((df[group_col] == gA).sum())
    n_B = int((df[group_col] == gB).sum())
    n_total = n_A + n_B

    if n_total == 0:
        raise ValueError("No rows found for SRM check.")

    exp_A = n_total * expected_split[0]
    exp_B = n_total * expected_split[1]

    chi2, pval = stats.chisquare(f_obs=[n_A, n_B], f_exp=[exp_A, exp_B])

    return {
        "n_A": n_A,
        "n_B": n_B,
        "expected_A": exp_A,
        "expected_B": exp_B,
        "chi2": chi2,
        "p_value": pval,
        "passes": pval >= 0.05,  # conventional threshold
    }


def run_ab_analysis(df: pd.DataFrame, outcome_col: str = "signed_up") -> dict:
    """
    Convenience wrapper: stats + lift + test + SRM.
    """
    s = compute_group_stats(df, outcome_col=outcome_col)
    lift = estimate_lift(s, alpha=0.05)
    test = test_lift(s, alternative="greater")
    srm = check_srm(df)

    return {
        "group_stats": s,
        "lift_ci": lift,
        "test": test,
        "srm": srm,
    }