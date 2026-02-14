# A/B Testing for Marketing Funnel Optimization

This repository demonstrates an end-to-end A/B testing workflow designed for marketing analytics and data science applications, with a focus on conversion rate optimization in a user acquisition funnel.

The project simulates, analyzes, and stress-tests an A/B experiment evaluating whether adding urgency-based copy (“Limited time offer”) to a landing page increases signup conversion, while monitoring downstream effects on purchase behavior.

---

## Landing Page Variants

<p align="center">
  <img src="images/landing_page_example.png" width="500">
</p>

---

## Business Problem

Marketing teams frequently test landing page copy to improve conversion rates, but reliable decision-making requires more than a single p-value. This project addresses the following questions:

- Does the treatment meaningfully increase signup conversion?
- Is the observed lift statistically and practically significant?
- Is the experiment sufficiently powered to detect real effects?
- How robust are conclusions under realistic conditions such as imperfect exposure or novelty decay?

---

## Experiment Design

- **Unit of randomization:** User  
- **Variants:**
  - Control (A): Standard landing page
  - Treatment (B): Landing page with “Limited time offer”
- **Primary metric:** Signup conversion rate  
- **Secondary metrics:** Purchase per exposure  
- **Guardrail metrics:** Purchase given signup  
- **Decision threshold:**
  - Statistical significance at α = 0.05 (one-sided)
  - Practical significance ≥ +0.5 percentage points

**User funnel:**  
Ad → Landing Page → Signup → Purchase

---

## Methodology

### Data Simulation
- User-level simulation with realistic covariates:
  - Device type (mobile / desktop)
  - Acquisition channel (search / social)
  - New vs returning users
- Signup probabilities generated using a **logistic model** to capture heterogeneity
- Purchases simulated conditional on signup

### Statistical Analysis
- Difference in proportions estimator for treatment effect
- One-sided two-sample z-test using pooled standard error under the null
- Wald confidence intervals using unpooled standard errors
- Sample Ratio Mismatch (SRM) detection for experiment validity

### Validation & Robustness
- **A/A tests** to validate false positive rate and CI calibration
- **Power analysis** across sample sizes via repeated simulation
- **Non-compliance modeling** to simulate partial treatment exposure
- **Novelty decay modeling** to assess time-varying treatment effects

---

## Key Results (Simulated)

- Treatment shows a statistically and practically significant lift in signup conversion
- Purchase per exposure increases due to higher signup volume
- No statistically meaningful degradation in purchase quality conditional on signup
- Power analysis confirms the experiment is sufficiently sized
- Robustness checks show expected attenuation under non-compliance and novelty effects

---

## Project Structure

```
├── notebook/
│   └── ab_testing_analysis.ipynb
├── src/
│   ├── simulate.py
│   ├── assignment.py
│   ├── analysis.py
│   ├── AATest.py
│   └── config.py
├── images/
│   └── landing_page_example.png
└── README.md
```

---

## Skills Demonstrated

- A/B test design and causal reasoning
- Statistical inference for binary outcomes
- Experiment validation (A/A testing, SRM detection)
- Power analysis and uncertainty quantification
- Marketing funnel analysis
- Robustness testing under real-world constraints
- Modular Python experimentation pipelines

---

## Notes

All data in this project is **simulated** for demonstration purposes. The goal is to illustrate best practices in experimentation and decision-making rather than to optimize a specific real-world product.
