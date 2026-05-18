# Step 04 - Deep survival next step

Recommended first deep-learning extension for this project:

1. Use CoxPH as the clinical baseline.
2. Use Random Survival Forest / Gradient Boosting Survival as ML baselines.
3. Add DeepSurv or Cox-Time only after the tabular feature pipeline is stable.

For this dataset, DeepSurv is the most direct deep-learning candidate because the
current outcome is single-risk right-censored 90-day mortality and the available
features are structured tabular variables. DeepHit/MTLR can be considered if you
want to discretize time into intervals such as 0-30, 31-60, and 61-90 days.

Explainability to report:

- Cox: hazard ratio and confidence interval.
- ML survival models: permutation importance.
- DeepSurv/Cox-Time: permutation importance or SHAP on predicted 90-day risk.
