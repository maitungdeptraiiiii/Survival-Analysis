# Report Outline

## 1. Research question

This project studies 90-day mortality among patients with heart failure using survival analysis.

Main questions:

- Which clinical and demographic factors are associated with 90-day mortality?
- Does using more variables improve survival prediction compared with the baseline Cox model?
- Can machine learning survival models improve discrimination over Cox proportional hazards?

## 2. Dataset and cohort

Use `outputs/step01_dataset_overview.json` to report:

- Number of patients/admissions
- Number of 90-day deaths
- 90-day mortality rate
- Follow-up definition and censoring at 90 days

Use `outputs/step01_missing_values.csv` to describe missingness.

## 3. Exploratory survival analysis

Use `outputs/step01_mortality_by_group.csv` to summarize mortality by:

- Age group
- Gender
- CCI group
- Admission type
- Insurance
- Race
- Marital status
- Individual Charlson comorbidities

## 4. Classical survival analysis

Use the original notebook plus `outputs/step02_logrank_extended.csv`.

Recommended plots/tables:

- Overall Kaplan-Meier curve
- Kaplan-Meier curves by age group, CCI group, admission type, and selected comorbidities
- Log-rank test table

## 5. Cox model comparison

Use `outputs/step02_cox_model_comparison.csv`.

Compare:

- Baseline Cox: age, gender, CCI
- Demographic Cox: baseline plus race, insurance, admission type, marital status
- Comorbidity-split Cox: age, gender, individual comorbidities
- Full penalized Cox: all encoded features

Report:

- Hazard ratio
- 95% confidence interval
- p-value
- Concordance index
- Partial AIC

Use `outputs/step02_ph_assumption_tests.csv` to discuss proportional hazards assumption.

## 6. Machine learning survival models

Use `outputs/step03_ml_model_comparison.csv` if Step 03 runs successfully.

Compare:

- Coxnet Survival
- Random Survival Forest
- Gradient Boosting Survival as an optional extension if runtime allows

Primary metric:

- Test concordance index

Use `outputs/step03_permutation_importance.csv` for explainability.

## 7. Risk stratification and calibration

Use Step 05 outputs:

- `outputs/step05_risk_group_summary.csv`
- `outputs/step05_risk_group_logrank.csv`
- `outputs/step05_calibration_90d.csv`
- `outputs/plots/step05_forest_plot_top20_cox.png`
- `outputs/plots/step05_km_by_predicted_risk_group.png`
- `outputs/plots/step05_calibration_90d.png`

Report:

- Mortality rate in low, medium, and high predicted-risk groups.
- Log-rank test across predicted-risk groups.
- Whether predicted 90-day risks are well calibrated against observed Kaplan-Meier risk.

## 8. Deep learning extension

Based on the deep survival analysis review, the most suitable next models are:

- DeepSurv: neural-network extension of CoxPH for tabular data.
- Cox-Time: allows time-varying effects.
- DeepHit or MTLR: useful if survival time is discretized into intervals.

For the current dataset, DeepSurv is the most practical first deep learning extension.

## 9. Conclusion

Summarize:

- Important risk factors.
- Whether expanded variables improve Cox performance.
- Whether ML survival models improve concordance.
- Explainability findings.
- Limitations: single cohort, limited variables, no ICU/lab/vital-sign features yet.

## 10. Future work

- Add ICU variables if available.
- Add laboratory values and vital signs.
- Evaluate time-dependent AUC and Brier score.
- Add DeepSurv/Cox-Time implementation.
- Add SHAP or other model explanation for predicted 90-day risk.
