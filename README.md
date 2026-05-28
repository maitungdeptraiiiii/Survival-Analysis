# Survival Analysis - ICU Longitudinal Mortality Prediction

Dự án này xây dựng pipeline dự đoán tử vong sớm / tử vong 90 ngày của bệnh nhân ICU từ data MIMIC-IV.
Pipeline tạo cohort ICU, trích xuất vital signs và lab tests trong giai đoạn đầu ICU, biến đổi dữ liệu dọc thành các đặc trưng tổng hợp, sau đó đánh giá bằng Cox model, các mô hình ML survival, risk stratification, calibration và fixed-horizon AUC.

## Cấu trúc thư mục và file

```text
Survival-Analysis-main/
  README.md                              # tài liệu hướng dẫn chạy project
  requirements.txt                       # danh sách thư viện Python cần cài đặt
  build_icu_longitudinal_cohort.py       # tạo cohort ICU từ MIMIC-IV và trích xuất vital/lab 24h đầu ICU
  enhanced_survival_pipeline.py          # chạy EDA, Cox, ML survival, risk groups, calibration và AUC
  cohort_icu_longitudinal_90d.csv        # cohort đầu ra sau khi chạy build_icu_longitudinal_cohort.py
  mimic-iv-3.1/                          # thư mục raw data MIMIC-IV
    hosp/                                # dữ liệu cấp bệnh viện: admission, patient, lab, thuốc, ICD
      admissions.csv                     # thông tin từng lần nhập viện, thời gian vào/ra viện, tử vong trong viện
      patients.csv                       # thông tin bệnh nhân: tuổi anchor, giới, ngày tử vong
      labevents.csv                      # kết quả xét nghiệm lab theo thời gian
      ...
    icu/                                 # dữ liệu ICU: ICU stay, charting, input/output, procedure ICU
      icustays.csv                       # thông tin từng ICU stay: stay_id, careunit, intime, outtime, LOS
      chartevents.csv                    # vital signs và các quan sát lâm sàng trong ICU theo thời gian
      ...
  outputs/                               # thư mục kết quả sinh ra khi chạy builder/pipeline
    cache_icu_vitals_24h.csv             # cache summary vital signs 24h để lần sau không quét lại chartevents
    cache_icu_labs_24h.csv               # cache summary lab 24h để lần sau không quét lại labevents
    step01_dataset_overview.json         # tổng quan cohort, số mẫu, số event, tỷ lệ tử vong
    step02_cox_model_comparison.csv      # kết quả so sánh các Cox models
    step03_ml_model_comparison.csv       # kết quả so sánh các ML survival models
    step04_binary_auc_by_horizon.csv     # AUC tại các mốc 7/14/30/60/90 ngày
    final_project_summary.md             # file tổng hợp kết quả cuối cùng
    plots/                               # các hình forest plot, Kaplan-Meier, calibration, AUC
```

## Cài đặt

```powershell
pip install -r requirements.txt
```

## Build cohort ICU longitudinal từ MIMIC-IV
```powershell
python build_icu_longitudinal_cohort.py
```

## Enhanced survival pipeline (Pipeline chính)

Sau khi tạo cohort ICU, chạy pipeline chính bằng:

```powershell
python enhanced_survival_pipeline.py
```

## Chạy từng bước

### Step 01 - Tổng quan cohort
```powershell
python enhanced_survival_pipeline.py --step 01
```

Output:

- `outputs/step01_dataset_overview.json`

Mục tiêu: chứng minh dữ liệu đủ lớn, ghi nhận số mẫu, số bệnh nhân, số ICU stays, tỷ lệ tử vong 90 ngày và các nhóm đặc trưng ICU/longitudinal được phát hiện.

Nếu dữ liệu có thêm cột ICU như `stay_id`, `intime`, `outtime`, `first_careunit`, `last_careunit`, pipeline sẽ tự ghi nhận số ICU stays. Nếu có `intime/outtime`, pipeline tự tạo `icu_los_hours` và `icu_los_group`.

### Step 02 - Survival analysis mở rộng

```powershell
python enhanced_survival_pipeline.py --step 02
```

Output:

- `outputs/step02_logrank_extended.csv`
- `outputs/step02_cox_model_comparison.csv`
- `outputs/step02_ph_assumption_tests.csv`

Model Cox được so sánh:

- `cox_demographic_baseline`: age + gender
- `cox_age_group_demographic`: age group + gender
- `cox_demographics`: baseline + admission type + insurance + race + marital status
- `cox_icu_longitudinal_if_available`: ICU / longitudinal features nếu có trong dữ liệu
- `cox_full_penalized`: tất cả biến đã encode, có penalizer để giảm overfitting

### Step 03 - Machine Learning survival models

```powershell
python enhanced_survival_pipeline.py --step 03
```

Output:

- `outputs/step03_ml_model_comparison.csv`
- `outputs/step03_permutation_importance.csv`
- `outputs/step03_rsf_permutation_importance.csv`
- `outputs/step03_feature_set_comparison.csv`

Model ML:

- Coxnet Survival
- Gradient Boosting Survival
- Random Survival Forest
- Extra Survival Trees

### Step 04 - Visualization, risk groups, calibration

```powershell
python enhanced_survival_pipeline.py --step 04
```

Output:

- `outputs/step04_top20_cox_hazard_ratios.csv`
- `outputs/step04_risk_group_summary.csv`
- `outputs/step04_risk_group_logrank.csv`
- `outputs/step04_calibration_90d.csv`
- `outputs/step04_binary_auc_by_horizon.csv`
- `outputs/plots/step04_forest_plot_top20_cox.png`
- `outputs/plots/step04_km_by_predicted_risk_group.png`
- `outputs/plots/step04_calibration_90d.png`
- `outputs/plots/step04_binary_auc_by_horizon.png`

Mục tiêu:

- Vẽ forest plot cho các hazard ratio nổi bật.
- Chia bệnh nhân thành nhóm nguy cơ thấp / trung bình / cao theo Cox full penalized.
- Vẽ Kaplan-Meier theo nhóm nguy cơ dự đoán.
- Kiểm tra calibration của predicted 90-day risk.
- Tính fixed-horizon binary ROC AUC tại 7, 14, 30, 60, và 90 ngày cho early mortality.

### Step 05 - Final project summary

```powershell
python enhanced_survival_pipeline.py --step 05
```

Output:

- `outputs/final_project_summary.md`

File này tổng hợp cohort size, 7/28/90-day outcomes, model comparison, feature-set comparison, fixed-horizon AUC, risk groups, và top RSF permutation importance.
