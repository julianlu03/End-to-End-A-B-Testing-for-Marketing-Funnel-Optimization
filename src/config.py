from dataclasses import dataclass

@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    split: float = 0.5
    traffic_per_day: int = 10_000
    baseline_signup_cvr: float = 0.12
    treatment_abs_lift: float = 0.012
    purchase_given_signup: float = 0.20
    practical_abs_lift: float = 0.005
    compliance_rate: float = 1.0 
    experiment_days: int = 28      
    novelty_decay_k: float = 0.0  # Test for novelty effects later 