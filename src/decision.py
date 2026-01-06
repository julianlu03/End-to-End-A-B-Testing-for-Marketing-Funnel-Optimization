import pandas as pd
from src.analysis import run_ab_analysis

def _fmt_pct(x: float, digits: int = 2) -> str:
    return f"{100*x:.{digits}f}%"

def _fmt_pp(x: float, digits: int = 3) -> str:
    # x is in probability units; convert to percentage points
    return f"{100*x:.{digits}f} pp"

def decision_clean(df, practical_lift: float = 0.005):
    # Run analyses
    signup = run_ab_analysis(df, outcome_col="signed_up")
    purchase = run_ab_analysis(df, outcome_col="purchased")

    df_signed = df[df["signed_up"] == 1]
    p_given_s = run_ab_analysis(df_signed, outcome_col="purchased")

    # Extract stats
    s = signup["group_stats"]
    p = purchase["group_stats"]
    q = p_given_s["group_stats"]

    # Validity (SRM only meaningful on full randomized population)
    srm_ok = signup["srm"]["passes"]

    # Primary criteria
    signup_ci_low = signup["lift_ci"]["ci_low"]
    signup_ci_high = signup["lift_ci"]["ci_high"]
    primary_stat_ok = signup_ci_low > 0.0
    primary_practical_ok = signup_ci_low > practical_lift
    primary_ok = srm_ok and primary_stat_ok and primary_practical_ok

    # Guardrail: purchase|signup — don’t SRM-check this subset
    guard_ci_low = p_given_s["lift_ci"]["ci_low"]
    guard_ci_high = p_given_s["lift_ci"]["ci_high"]

    # A simple guardrail rule (you can tighten later):
    # fail only if we can confidently say it's worse (CI entirely below 0)
    guardrail_ok = not (guard_ci_high < 0.0)

    # Overall decision (revenue/bounce not available)
    ship = primary_ok and guardrail_ok

    out = {
        "decision": "SHIP" if ship else "DO NOT SHIP",
        "notes": [
            "Revenue check not available (need order_value + cost model).",
            "Bounce rate not available (need session/event data).",
            "SRM on purchase|signup subset is not applicable (conditioning on post-treatment variable).",
        ],

        "validity": {
            "srm_passes": srm_ok,
            "n_A": signup["srm"]["n_A"],
            "n_B": signup["srm"]["n_B"],
            "srm_p_value": signup["srm"]["p_value"],
        },

        "primary_signup_cvr": {
            "A_rate": _fmt_pct(s.p_A),
            "B_rate": _fmt_pct(s.p_B),
            "lift": _fmt_pp(signup["lift_ci"]["lift"]),
            "ci_95": f"[{_fmt_pp(signup_ci_low)}, {_fmt_pp(signup_ci_high)}]",
            "p_value_one_sided": signup["test"]["p_value"],
            "passes_stat_sig": primary_stat_ok,
            "passes_practical": primary_practical_ok,
            "practical_threshold": _fmt_pp(practical_lift),
        },

        "secondary_purchase_per_exposure": {
            "A_rate": _fmt_pct(p.p_A),
            "B_rate": _fmt_pct(p.p_B),
            "lift": _fmt_pp(purchase["lift_ci"]["lift"]),
            "ci_95": f"[{_fmt_pp(purchase['lift_ci']['ci_low'])}, {_fmt_pp(purchase['lift_ci']['ci_high'])}]",
            "p_value_one_sided": purchase["test"]["p_value"],
        },

        "guardrail_purchase_given_signup": {
            "A_rate": _fmt_pct(q.p_A),
            "B_rate": _fmt_pct(q.p_B),
            "lift": _fmt_pp(p_given_s["lift_ci"]["lift"]),
            "ci_95": f"[{_fmt_pp(guard_ci_low)}, {_fmt_pp(guard_ci_high)}]",
            "guardrail_ok": guardrail_ok,
        },
    }

    return out

# Readable version
def print_decision(summary: dict):
    print(f"\nDECISION: {summary['decision']}")
    print("\nVALIDITY")
    v = summary["validity"]
    print(f"  SRM passes: {v['srm_passes']}  (p={v['srm_p_value']:.4f}, nA={v['n_A']}, nB={v['n_B']})")

    print("\nPRIMARY (Signup CVR)")
    m = summary["primary_signup_cvr"]
    print(f"  A: {m['A_rate']}  |  B: {m['B_rate']}")
    print(f"  Lift: {m['lift']}  95% CI: {m['ci_95']}")
    print(f"  p(one-sided): {m['p_value_one_sided']:.3e}")
    print(f"  Stat sig: {m['passes_stat_sig']} | Practical: {m['passes_practical']} (threshold {m['practical_threshold']})")

    print("\nSECONDARY (Purchase per exposure)")
    s = summary["secondary_purchase_per_exposure"]
    print(f"  A: {s['A_rate']}  |  B: {s['B_rate']}")
    print(f"  Lift: {s['lift']}  95% CI: {s['ci_95']}")
    print(f"  p(one-sided): {s['p_value_one_sided']:.3e}")

    print("\nGUARDRAIL (Purchase | Signup)")
    g = summary["guardrail_purchase_given_signup"]
    print(f"  A: {g['A_rate']}  |  B: {g['B_rate']}")
    print(f"  Lift: {g['lift']}  95% CI: {g['ci_95']}")
    print(f"  Guardrail OK: {g['guardrail_ok']}")

    print("\nNOTES")
    for note in summary["notes"]:
        print(f"  - {note}")