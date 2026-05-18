# Survival Analysis - 90-day Mortality in Heart Failure

Project này phân tích sống còn cho bệnh nhân heart failure với outcome là tử vong trong 90 ngày.

## Dữ liệu hiện có

- File chính: `cohort_hf_survival_90d.csv`
- Duration column: `time_to_event_90d`
- Event column: `death_90d`
- Notebook baseline: `survival_analysis_90d_hf.ipynb`
- Pipeline mở rộng: `enhanced_survival_pipeline.py`

## Cài đặt

```powershell
python -m pip install -r requirements.txt
```

Nếu muốn chạy thêm Machine Learning survival models:

```powershell
python -m pip install -r requirements-ml.txt
```

## Chạy từng bước

### Step 01 - EDA và mortality theo nhóm

```powershell
python enhanced_survival_pipeline.py --step 01
```

Output:

- `outputs/step01_dataset_overview.json`
- `outputs/step01_missing_values.csv`
- `outputs/step01_mortality_by_group.csv`

Mục tiêu: chứng minh dữ liệu đủ lớn, kiểm tra missing values, và xem tử vong 90 ngày thay đổi theo age, gender, CCI, admission type, insurance, race, marital status, từng bệnh nền.

### Step 02 - Survival analysis mở rộng

```powershell
python enhanced_survival_pipeline.py --step 02
```

Output:

- `outputs/step02_logrank_extended.csv`
- `outputs/step02_cox_model_comparison.csv`
- `outputs/step02_ph_assumption_tests.csv`

Model Cox được so sánh:

- `cox_baseline_3vars`: age + gender + CCI
- `cox_demographics`: baseline + admission type + insurance + race + marital status
- `cox_comorbidity_split`: age + gender + từng bệnh nền Charlson tách riêng
- `cox_full_penalized`: tất cả biến đã encode, có penalizer để giảm overfitting

### Step 03 - Machine Learning survival models

```powershell
python enhanced_survival_pipeline.py --step 03
```

Output nếu cài đủ `scikit-survival`:

- `outputs/step03_ml_model_comparison.csv`
- `outputs/step03_permutation_importance.csv`

Model ML:

- Coxnet Survival
- Random Survival Forest

Gradient Boosting Survival được ghi chú là hướng mở rộng tùy chọn vì chạy quá chậm trên full dataset trong môi trường local hiện tại.

Nếu thiếu package, pipeline sẽ tạo file `outputs/step03_ml_survival_SKIPPED.txt` và các bước khác vẫn chạy bình thường.

### Step 04 - Deep survival và explainability

```powershell
python enhanced_survival_pipeline.py --step 04
```

Output:

- `outputs/step04_deep_survival_next_steps.md`

Hướng phát triển đề xuất:

- DeepSurv hoặc Cox-Time cho dữ liệu tabular survival.
- DeepHit/MTLR nếu chia thời gian thành các khoảng 0-30, 31-60, 61-90 ngày.
- Explainability: hazard ratio cho Cox, permutation importance cho ML, SHAP/permutation importance cho deep survival.

### Step 05 - Visualization, risk groups, calibration

```powershell
python enhanced_survival_pipeline.py --step 05
```

Output:

- `outputs/step05_top20_cox_hazard_ratios.csv`
- `outputs/step05_risk_group_summary.csv`
- `outputs/step05_risk_group_logrank.csv`
- `outputs/step05_calibration_90d.csv`
- `outputs/plots/step05_forest_plot_top20_cox.png`
- `outputs/plots/step05_km_by_predicted_risk_group.png`
- `outputs/plots/step05_calibration_90d.png`

Mục tiêu:

- Vẽ forest plot cho các hazard ratio nổi bật.
- Chia bệnh nhân thành nhóm nguy cơ thấp / trung bình / cao theo Cox full penalized.
- Vẽ Kaplan-Meier theo nhóm nguy cơ dự đoán.
- Kiểm tra calibration của predicted 90-day risk.

Time-dependent AUC cũng được thử trong step này. Nếu không tính được do phân bố censoring ở mốc 90 ngày, pipeline sẽ tạo file `outputs/step05_time_dependent_auc_SKIPPED.txt`.

## Chạy toàn bộ

```powershell
python enhanced_survival_pipeline.py --step all
```

Hoặc:

```powershell
python a.py
```

## Gợi ý viết báo cáo

1. Giới thiệu bài toán: dự đoán/nghiên cứu yếu tố nguy cơ tử vong 90 ngày ở bệnh nhân heart failure.
2. Mô tả cohort: số mẫu, số event, event rate, missing values.
3. Baseline survival analysis: Kaplan-Meier, log-rank, Cox với age/gender/CCI.
4. Mở rộng biến: demographics, admission information, từng bệnh nền Charlson.
5. So sánh mô hình: Cox baseline vs Cox mở rộng vs ML survival.
6. Explainability: hazard ratio và feature importance.
7. Risk stratification: low / medium / high risk survival curves.
8. Calibration: predicted vs observed 90-day mortality.
9. Hướng phát triển deep learning: DeepSurv/Cox-Time theo bài review deep learning survival analysis.
