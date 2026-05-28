# Final Project Summary

## Cohort

- Rows / ICU stays: 65,366
- Subjects: 65366
- 90-day deaths: 11,906
- 90-day mortality rate: 0.1821
- Longitudinal summary features detected: 168

## Outcomes Available

- 7-day mortality: 4,865 events (0.0744)
- 28-day mortality: 8,796 events (0.1346)
- 90-day mortality: 11,906 events (0.1821)

## Model Comparison

- random_survival_forest: C-index 0.8175 (400 features, 30000 rows)
- extra_survival_trees: C-index 0.8107 (400 features, 30000 rows)
- gradient_boosting_survival: C-index 0.7219 (400 features, 30000 rows)
- coxnet: C-index 0.7181 (400 features, 30000 rows)

## Feature Set Comparison

- vitals_labs_24h: C-index 0.8196 (398 features)
- full_all_features: C-index 0.8175 (400 features)
- labs_24h: C-index 0.8162 (270 features)
- vitals_24h: C-index 0.7632 (190 features)
- demographics_icu: C-index 0.7421 (62 features)

## Fixed-Horizon AUC

- 7 days: AUC 0.8873 (550 events in holdout)
- 14 days: AUC 0.8723 (788 events in holdout)
- 30 days: AUC 0.8580 (1021 events in holdout)
- 60 days: AUC 0.8424 (1227 events in holdout)
- 90 days: AUC 0.8334 (1344 events in holdout)

## Holdout Risk Groups

- low: 2500 rows, 46 events, event rate 0.0184
- medium: 2500 rows, 321 events, event rate 0.1284
- high: 2500 rows, 977 events, event rate 0.3908

## Top RSF Permutation Importance

- anchor_age: 0.02305
- lactate_last_24h: 0.00716
- bun_mean_24h: 0.00664
- spo2_last_24h: 0.00634
- platelet_min_24h: 0.00425
- ph_last_24h: 0.00423
- bicarbonate_mean_24h: 0.00380
- wbc_min_24h: 0.00326
- po2_slope_24h: 0.00299
- resp_rate_mean_24h: 0.00296

## Current Interpretation

This project uses ICU stays from MIMIC-IV with longitudinal vital/lab summaries from the first 24 ICU hours. Random Survival Forest is the strongest current model and is used for holdout risk stratification, calibration, and fixed-horizon early mortality AUC.
