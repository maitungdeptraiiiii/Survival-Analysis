from __future__ import annotations

import argparse
import json
import logging
import os
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning


# Các hằng số cấu hình cho pipeline
DATA_FILE = Path("cohort_icu_longitudinal_90d.csv")
OUTPUT_DIR = Path("outputs")
DURATION_COL = "time_to_event_90d"
EVENT_COL = "death_90d"
RANDOM_STATE = 42
MAX_ML_ROWS = 30000
MAX_PH_TEST_FEATURES = 80

LOGGER = logging.getLogger("survival_pipeline")

# Bộ biến baseline lâm sàng tối thiểu cho mô hình Cox ICU đơn giản nhất.
BASE_FEATURES = ["anchor_age", "gender_male"]

# Các biến phân loại cấp bệnh viện, được one-hot encode cho Cox/ML.
CATEGORICAL_FEATURES = ["admission_type", "insurance", "race_simple", "marital_status"]

# Các biến phân loại dẫn xuất từ biến liên tục hoặc có thể có nhóm tuổi, được one-hot encode cho Cox/ML.
DERIVED_CATEGORICAL_FEATURES = ["age_group"]

# Các biến mô tả ICU stay được thêm bởi build_icu_longitudinal_cohort.py.
ICU_CATEGORICAL_FEATURES = ["first_careunit", "last_careunit"]

ICU_NUMERIC_FEATURES = ["icu_los_hours", "icu_los_days", "los_icu"]

# Các hậu tố dùng để tự động nhận diện feature summary theo cửa sổ thời gian.
LONGITUDINAL_SUMMARY_KEYWORDS = ("_first", "_last", "_min", "_max", "_mean", "_median", "_std", "_slope", "_delta", "_count")

# Các tiền tố biến lâm sàng được tạo bởi bộ build cohort ICU longitudinal.
LONGITUDINAL_CLINICAL_PREFIXES = ("heart_rate", "sbp", "dbp", "mbp", "resp_rate", "temperature", "spo2", "glucose", "creatinine", "bun", "wbc", "hemoglobin", "platelet", "sodium", "potassium", "chloride", "bicarbonate", "lactate", "ph", "po2", "pco2")

# Nhóm đặc trưng dùng để so sánh kiểu ablation ở Bước 03.
VITAL_PREFIXES = ("heart_rate", "sbp", "dbp", "mbp", "resp_rate", "temperature", "spo2", "glucose")

LAB_PREFIXES = ("creatinine", "bun", "wbc", "hemoglobin", "platelet", "sodium", "potassium", "chloride", "bicarbonate", "lactate", "ph", "po2", "pco2")


# def ensure_output_dir() -> None:
#     # Tạo thư mục output nếu chưa tồn tại, để lưu tất cả kết quả của pipeline.
#     OUTPUT_DIR.mkdir(exist_ok=True)

def setup_logging(verbose: bool = False) -> None:
    # Cấu hình log terminal và ẩn các warning nhiễu về convergence/performance.
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        force=True,
    )
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=PerformanceWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"lifelines\..*")
    warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"matplotlib\..*")
    try:
        from lifelines.exceptions import ConvergenceWarning

        warnings.filterwarnings("ignore", category=ConvergenceWarning)
    except Exception:
        pass


def default_data_file() -> Path:
    # Trả về cohort ICU longitudinal mặc định của pipeline.
    return DATA_FILE


def log_section(title: str) -> None:
    # In tiêu đề section rõ ràng trên terminal log.
    LOGGER.info("")
    LOGGER.info("%s:", title)


def log_item(message: str, *args: object) -> None:
    # In chi tiết tiến trình, chỉ hiện khi chạy với --verbose.
    LOGGER.debug("  " + message, *args)


def log_progress(message: str, *args: object) -> None:
    # In dòng tiến trình chính trong chế độ log mặc định.
    LOGGER.info("  " + message, *args)


def log_done(message: str, *args: object) -> None:
    # In một dòng hoàn tất có thụt lề.
    LOGGER.info("  " + message, *args)


def log_skip(message: str, *args: object) -> None:
    # In một dòng bỏ qua cho các bước tuỳ chọn.
    LOGGER.info("  - " + message, *args)


def load_data(path: Path = DATA_FILE) -> pd.DataFrame:
    # Load cohort CSV và kiểm tra các cột tối thiểu cần cho survival analysis.
    log_progress("Loading dataset")
    df = pd.read_csv(path)
    required = {DURATION_COL, EVENT_COL, "anchor_age", "gender"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    log_progress("Loaded %s rows x %s columns", len(df), df.shape[1])
    return df


def simplify_race(value: object) -> str:
    # Gộp nhãn race chi tiết của MIMIC thành các nhóm lớn dễ báo cáo.
    text = str(value).upper()
    if text in {"NAN", "NONE", ""}:
        return "UNKNOWN"
    if "WHITE" in text:
        return "WHITE"
    if "BLACK" in text or "AFRICAN" in text:
        return "BLACK"
    if "HISPANIC" in text or "LATINO" in text:
        return "HISPANIC/LATINO"
    if "ASIAN" in text:
        return "ASIAN"
    if "UNKNOWN" in text or "UNABLE" in text or "DECLINED" in text:
        return "UNKNOWN"
    return "OTHER"


def add_analysis_features(df: pd.DataFrame) -> pd.DataFrame:
    # Thêm các biến dẫn xuất dùng cho EDA, Cox và ML survival models.
    log_item("Adding derived analysis features")
    out = df.copy()
    out["gender_male"] = out["gender"].map({"F": 0, "M": 1})
    out["age_group"] = pd.cut(
        out["anchor_age"],
        bins=[0, 65, 80, np.inf],
        labels=["<65", "65-79", ">=80"],
        right=False,
    )
    if "race" in out.columns:
        out["race_simple"] = out["race"].map(simplify_race)
    for col in ["marital_status", "insurance", "admission_type"]:
        if col in out.columns:
            out[col] = out[col].fillna("UNKNOWN")
    if "intime" in out.columns and "outtime" in out.columns and "icu_los_hours" not in out.columns:
        intime = pd.to_datetime(out["intime"], errors="coerce")
        outtime = pd.to_datetime(out["outtime"], errors="coerce")
        out["icu_los_hours"] = (outtime - intime).dt.total_seconds() / 3600
    if "icu_los_hours" in out.columns and "icu_los_group" not in out.columns:
        out["icu_los_group"] = pd.cut(
            out["icu_los_hours"],
            bins=[-np.inf, 24, 72, 168, np.inf],
            labels=["<24h", "1-3d", "3-7d", ">=7d"],
            right=False,
        )
    for col in ICU_CATEGORICAL_FEATURES:
        if col in out.columns:
            out[col] = out[col].fillna("UNKNOWN")
    log_item(
        "Detected ICU features: categorical=%s numeric=%s longitudinal=%s",
        detected_icu_features(out)["categorical"],
        detected_icu_features(out)["numeric"],
        len(detected_icu_features(out)["longitudinal"]),
    )
    return out


def existing_columns(df: pd.DataFrame, cols: Iterable[str]) -> list[str]:
    # Chỉ trả về các cột yêu cầu thật sự tồn tại trong dataframe.
    return [c for c in cols if c in df.columns]


def detect_longitudinal_summary_features(df: pd.DataFrame) -> list[str]:
    # Nhận diện cột summary vital/lab bằng tiền tố lâm sàng và hậu tố thống kê.
    features = []
    for col in df.columns:
        lower = col.lower()
        if not any(lower.startswith(prefix) for prefix in LONGITUDINAL_CLINICAL_PREFIXES):
            continue
        if not any(keyword in lower for keyword in LONGITUDINAL_SUMMARY_KEYWORDS):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            features.append(col)
    return features


def detected_icu_features(df: pd.DataFrame) -> dict[str, list[str]]:
    # Tập hợp các feature ICU dạng phân loại, số và longitudinal đang có.
    categorical = existing_columns(df, ICU_CATEGORICAL_FEATURES)
    if "icu_los_group" in df.columns:
        categorical.append("icu_los_group")
    numeric = existing_columns(df, ICU_NUMERIC_FEATURES)
    longitudinal = detect_longitudinal_summary_features(df)
    return {
        "categorical": categorical,
        "numeric": numeric,
        "longitudinal": longitudinal,
    }


def mortality_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    # Tóm tắt số ca tử vong và tỷ lệ tử vong 90 ngày theo một biến nhóm.
    table = (
        df.groupby(group_col, dropna=False, observed=False)[EVENT_COL]
        .agg(n="count", events="sum", event_rate="mean")
        .reset_index()
        .sort_values(["event_rate", "n"], ascending=[False, False])
    )
    table.insert(0, "variable", group_col)
    return table


def save_step_01_eda(df: pd.DataFrame) -> None:
    # Bước 01: xuất tổng quan cohort ICU dùng cho phân tích.
    log_section("Step 01 - EDA and mortality summaries")
    # ensure_output_dir()
    icu_features = detected_icu_features(df)
    overview = {
        "n_rows": int(len(df)),
        "n_subjects": int(df["subject_id"].nunique()) if "subject_id" in df else None,
        "n_admissions": int(df["hadm_id"].nunique()) if "hadm_id" in df else None,
        "n_icu_stays": int(df["stay_id"].nunique()) if "stay_id" in df else None,
        "n_columns": int(df.shape[1]),
        "n_events_90d": int(df[EVENT_COL].sum()),
        "event_rate_90d": float(df[EVENT_COL].mean()),
        "median_followup_days": float(df[DURATION_COL].median()),
        "detected_icu_categorical_features": icu_features["categorical"],
        "detected_icu_numeric_features": icu_features["numeric"],
        "detected_longitudinal_summary_features": icu_features["longitudinal"],
    }
    (OUTPUT_DIR / "step01_dataset_overview.json").write_text(
        json.dumps(overview, indent=2), encoding="utf-8"
    )
    log_done(
        "Overview: %s events among %s rows",
        overview["n_events_90d"],
        overview["n_rows"],
    )


def build_model_matrix(
    df: pd.DataFrame,
    include_derived_groups: bool = False,
    include_icu: bool = True,
    include_longitudinal: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    # Tạo bảng modeling dạng số với imputation, cờ missing và one-hot encoding.
    icu_features = detected_icu_features(df)
    numeric_binary = [
        c
        for c in ["anchor_age", "gender_male"]
        if c in df.columns
    ]
    if include_icu:
        numeric_binary += [c for c in icu_features["numeric"] if c not in numeric_binary]
    if include_longitudinal:
        numeric_binary += [c for c in icu_features["longitudinal"] if c not in numeric_binary]

    categorical_features = existing_columns(df, CATEGORICAL_FEATURES)
    if include_derived_groups:
        categorical_features += existing_columns(df, DERIVED_CATEGORICAL_FEATURES)
    if include_icu:
        categorical_features += [
            c for c in icu_features["categorical"] if c not in categorical_features
        ]

    model_df = df[[DURATION_COL, EVENT_COL] + numeric_binary + categorical_features].copy()
    model_df = model_df.dropna(subset=[DURATION_COL, EVENT_COL])
    for col in categorical_features:
        model_df[col] = model_df[col].astype("object").fillna("UNKNOWN")
    missing_indicators = {}
    for col in numeric_binary:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        if model_df[col].isna().any():
            missing_indicators[f"{col}_missing"] = model_df[col].isna().astype(float)
            median = model_df[col].median()
            if pd.isna(median):
                median = 0.0
            model_df[col] = model_df[col].fillna(median)
    if missing_indicators:
        model_df = pd.concat(
            [model_df, pd.DataFrame(missing_indicators, index=model_df.index)],
            axis=1,
        )
    model_df = pd.get_dummies(
        model_df,
        columns=categorical_features,
        drop_first=True,
        dtype=float,
    )

    feature_cols = [c for c in model_df.columns if c not in {DURATION_COL, EVENT_COL}]
    low_variance = [c for c in feature_cols if model_df[c].nunique(dropna=True) <= 1]
    if low_variance:
        model_df = model_df.drop(columns=low_variance)
        feature_cols = [c for c in feature_cols if c not in low_variance]
    return model_df, feature_cols


def standardize_for_cox(fit_df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    # Chuẩn hoá các cột liên tục để Cox ổn định hơn và tránh overflow trong lifelines.
    out = fit_df.copy()
    for col in cols:
        values = out[col].astype(float).replace([np.inf, -np.inf], np.nan)
        if values.isna().any():
            median = values.median()
            if pd.isna(median):
                median = 0.0
            values = values.fillna(median)
        unique_values = values.nunique(dropna=True)
        if unique_values <= 2:
            out[col] = values
            continue
        std = values.std(ddof=0)
        if pd.isna(std) or std <= 1e-12:
            out[col] = values
            continue
        out[col] = (values - values.mean()) / std
    return out


def cox_penalizer_candidates(n_features: int) -> list[float]:
    # Tạo danh sách penalizer tăng dần để Cox nhiều biến không bị singular matrix.
    if n_features > 100:
        candidates = [0.1, 0.5, 1.0, 5.0]
    elif n_features > 10:
        candidates = [0.01, 0.1, 0.5, 1.0]
    else:
        candidates = [0.0, 0.01, 0.1]
    return list(dict.fromkeys(candidates))


def feature_matches_prefix(feature: str, prefixes: Iterable[str]) -> bool:
    # Kiểm tra một feature của mô hình có thuộc nhóm biến lâm sàng nào không.
    base = feature.removesuffix("_missing")
    return any(base.startswith(prefix) for prefix in prefixes)


def feature_set_columns(feature_cols: list[str]) -> dict[str, list[str]]:
    # Định nghĩa các tập feature để so sánh demographics, vitals, labs và full model.
    demographic_icu = [
        c
        for c in feature_cols
        if c in {"anchor_age", "gender_male", "icu_los_hours", "icu_los_days"}
        or c.startswith("admission_type_")
        or c.startswith("insurance_")
        or c.startswith("race_simple_")
        or c.startswith("marital_status_")
        or c.startswith("first_careunit_")
        or c.startswith("last_careunit_")
        or c.startswith("icu_los_group_")
    ]
    vital_cols = [c for c in feature_cols if feature_matches_prefix(c, VITAL_PREFIXES)]
    lab_cols = [c for c in feature_cols if feature_matches_prefix(c, LAB_PREFIXES)]
    return {
        "demographics_icu": demographic_icu,
        "vitals_24h": demographic_icu + vital_cols,
        "labs_24h": demographic_icu + lab_cols,
        "vitals_labs_24h": demographic_icu + vital_cols + lab_cols,
        "full_all_features": feature_cols,
    }


def cox_summary_frame(model, model_name: str) -> pd.DataFrame:
    # Chuyển summary của Cox model đã fit thành bảng output thống nhất.
    out = model.summary.reset_index().rename(columns={"covariate": "term"})
    out.insert(0, "model", model_name)
    out["concordance"] = model.concordance_index_
    out["partial_AIC"] = model.AIC_partial_
    cols = [
        "model",
        "term",
        "coef",
        "exp(coef)",
        "exp(coef) lower 95%",
        "exp(coef) upper 95%",
        "p",
        "concordance",
        "partial_AIC",
    ]
    return out[cols]


def save_step_02_cox(df: pd.DataFrame) -> None:
    # Bước 02: chạy log-rank, nhiều cấu hình Cox và kiểm định PH assumption.
    from lifelines import CoxPHFitter
    from lifelines.statistics import multivariate_logrank_test, proportional_hazard_test

    # ensure_output_dir()
    log_section("Step 02 - Log-rank tests and Cox model comparison")

    logrank_rows = []
    logrank_cols = [
        "gender",
        "age_group",
        "admission_type",
        "insurance",
        "race_simple",
        "marital_status",
        "icu_los_group",
        "first_careunit",
        "last_careunit",
    ]
    for col in logrank_cols:
        if col not in df.columns:
            continue
        tmp = df[[DURATION_COL, EVENT_COL, col]].dropna()
        if tmp[col].nunique() < 2:
            continue
        result = multivariate_logrank_test(tmp[DURATION_COL], tmp[col], tmp[EVENT_COL])
        logrank_rows.append(
            {
                "variable": col,
                "n_groups": int(tmp[col].nunique()),
                "test_statistic": float(result.test_statistic),
                "p": float(result.p_value),
            }
    )
    pd.DataFrame(logrank_rows).to_csv(OUTPUT_DIR / "step02_logrank_extended.csv", index=False)

    model_df, feature_cols = build_model_matrix(df)
    granular_df, granular_feature_cols = build_model_matrix(
        df,
        include_derived_groups=True,
        include_icu=True,
        include_longitudinal=True,
    )
    model_specs = {
        "cox_demographic_baseline": [c for c in BASE_FEATURES if c in model_df.columns],
        "cox_age_group_demographic": [
            c
            for c in granular_feature_cols
            if c == "gender_male" or c.startswith("age_group_")
        ],
        "cox_demographics": [
            c
            for c in feature_cols
            if c in BASE_FEATURES
            or c.startswith("admission_type_")
            or c.startswith("insurance_")
            or c.startswith("race_simple_")
            or c.startswith("marital_status_")
        ],
        "cox_icu_longitudinal_if_available": [
            c
            for c in feature_cols
            if c in detected_icu_features(df)["numeric"]
            or c in detected_icu_features(df)["longitudinal"]
            or c.startswith("first_careunit_")
            or c.startswith("last_careunit_")
            or c.startswith("icu_los_group_")
        ],
        "cox_full_penalized": feature_cols,
    }

    summaries = []
    ph_rows = []
    failed_model_rows = []
    for name, cols in model_specs.items():
        if not cols:
            log_skip("Skipping %s because no matching features are available", name)
            continue
        log_item("Fitting %-34s %3s features", name, len(cols))
        source_df = granular_df if name == "cox_age_group_demographic" else model_df
        fit_df = source_df[[DURATION_COL, EVENT_COL] + cols].copy()
        fit_df = standardize_for_cox(fit_df, cols)
        cph = None
        last_error = ""
        used_penalizer = None
        for penalizer in cox_penalizer_candidates(len(cols)):
            try:
                candidate = CoxPHFitter(penalizer=penalizer)
                candidate.fit(fit_df, duration_col=DURATION_COL, event_col=EVENT_COL)
                cph = candidate
                used_penalizer = penalizer
                break
            except Exception as exc:
                last_error = str(exc)
        if cph is None:
            LOGGER.warning("  ! %s failed: %s", name, last_error)
            failed_model_rows.append(
                {
                    "model": name,
                    "n_features": len(cols),
                    "status": "failed",
                    "error": last_error,
                }
            )
            continue
        if used_penalizer is not None and used_penalizer > 0:
            log_item("Used Cox penalizer %.2g for %s", used_penalizer, name)
        summaries.append(cox_summary_frame(cph, name))

        if len(cols) > MAX_PH_TEST_FEATURES:
            log_item(
                "PH test skipped for %s because it has %s features",
                name,
                len(cols),
            )
            continue
        try:
            ph = proportional_hazard_test(cph, fit_df, time_transform="rank").summary
            ph = ph.reset_index().rename(columns={"index": "term"})
            ph.insert(0, "model", name)
            ph_rows.append(ph)
        except Exception as exc:
            log_item("PH test skipped for %s: %s", name, exc)

    if summaries:
        pd.concat(summaries, ignore_index=True).to_csv(
            OUTPUT_DIR / "step02_cox_model_comparison.csv", index=False
        )
    if ph_rows:
        pd.concat(ph_rows, ignore_index=True).to_csv(
            OUTPUT_DIR / "step02_ph_assumption_tests.csv", index=False
        )
    if failed_model_rows:
        pd.DataFrame(failed_model_rows).to_csv(
            OUTPUT_DIR / "step02_cox_model_failures.csv", index=False
        )
    log_done("%s Cox model summaries", len(summaries))
    if ph_rows:
        log_item("Saved step02_ph_assumption_tests.csv")
    if failed_model_rows:
        log_item("Saved step02_cox_model_failures.csv")


def save_step_03_ml_survival(df: pd.DataFrame) -> None:
    # Bước 03: so sánh ML survival models và xuất feature-set/importance.
    try:
        from sklearn.inspection import permutation_importance
        from sklearn.model_selection import train_test_split
        from sksurv.ensemble import ExtraSurvivalTrees
        from sksurv.ensemble import GradientBoostingSurvivalAnalysis
        from sksurv.ensemble import RandomSurvivalForest
        from sksurv.linear_model import CoxnetSurvivalAnalysis
        from sksurv.util import Surv
    except ImportError as exc:
        msg = (
            "Step 03 requires scikit-survival and scikit-learn. "
            "Install dependencies from requirements.txt, then run this step again. "
            f"Import error: {exc}"
        )
        (OUTPUT_DIR / "step03_ml_survival_SKIPPED.txt").write_text(msg, encoding="utf-8")
        LOGGER.warning("  ! %s", msg)
        return

    # ensure_output_dir()
    log_section("Step 03 - ML survival model comparison")
    model_df, feature_cols = build_model_matrix(df)
    X = model_df[feature_cols].astype(float)
    y_source = model_df
    if len(X) > MAX_ML_ROWS:
        sampled = X.sample(n=MAX_ML_ROWS, random_state=RANDOM_STATE).index
        X = X.loc[sampled]
        y_source = model_df.loc[sampled]
        log_item("Sampled %s rows for ML runtime control", MAX_ML_ROWS)
    y = Surv.from_dataframe(EVENT_COL, DURATION_COL, y_source)
    log_progress("ML matrix: %s rows x %s features", X.shape[0], X.shape[1])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE
    )

    models = {
        "coxnet": CoxnetSurvivalAnalysis(l1_ratio=0.5, alpha_min_ratio=0.01),
        "gradient_boosting_survival": GradientBoostingSurvivalAnalysis(
            n_estimators=40,
            learning_rate=0.05,
            max_depth=2,
            random_state=RANDOM_STATE,
        ),
        "random_survival_forest": RandomSurvivalForest(
            n_estimators=40,
            min_samples_split=20,
            min_samples_leaf=10,
            n_jobs=1,
            random_state=RANDOM_STATE,
        ),
        "extra_survival_trees": ExtraSurvivalTrees(
            n_estimators=40,
            min_samples_split=20,
            min_samples_leaf=10,
            n_jobs=1,
            random_state=RANDOM_STATE,
        ),
    }

    metrics = []
    importances = []
    rsf_importances = []
    for name, model in models.items():
        log_progress("Fitting model: %s", name)
        try:
            model.fit(X_train, y_train)
            cindex = float(model.score(X_test, y_test))
            log_progress("C-index = %.4f", cindex)
        except Exception as exc:
            LOGGER.exception("%s failed", name)
            metrics.append(
                {
                    "model": name,
                    "test_concordance_index": np.nan,
                    "status": "failed",
                    "error": str(exc),
                    "n_rows_used": int(len(X)),
                    "n_features": int(len(feature_cols)),
                }
            )
            continue
        metrics.append(
            {
                "model": name,
                "test_concordance_index": cindex,
                "status": "ok",
                "error": "",
                "n_rows_used": int(len(X)),
                "n_features": int(len(feature_cols)),
            }
        )
        pd.DataFrame(metrics).sort_values(
            "test_concordance_index", ascending=False
        ).to_csv(OUTPUT_DIR / "step03_ml_model_comparison.csv", index=False)

        if name == "random_survival_forest":
            log_item("Computing lightweight permutation importance for %s", name)
            if len(X_test) > 500:
                rng = np.random.default_rng(RANDOM_STATE)
                perm_idx = np.sort(rng.choice(len(X_test), size=500, replace=False))
                X_perm = X_test.iloc[perm_idx]
                y_perm = y_test[perm_idx]
            else:
                X_perm = X_test
                y_perm = y_test
            perm = permutation_importance(
                model,
                X_perm,
                y_perm,
                n_repeats=2,
                random_state=RANDOM_STATE,
                n_jobs=1,
            )
            for feature, mean, std in zip(
                feature_cols, perm.importances_mean, perm.importances_std
            ):
                rsf_importances.append(
                    {
                        "model": name,
                        "feature": feature,
                        "importance_mean": float(mean),
                        "importance_std": float(std),
                    }
                )
            pd.DataFrame(rsf_importances).sort_values(
                ["model", "importance_mean"], ascending=[True, False]
            ).to_csv(OUTPUT_DIR / "step03_rsf_permutation_importance.csv", index=False)
            continue

        if name != "coxnet":
            continue

        log_item("Computing permutation importance for %s", name)
        if len(X_test) > 1000:
            rng = np.random.default_rng(RANDOM_STATE)
            perm_idx = np.sort(rng.choice(len(X_test), size=1000, replace=False))
            X_perm = X_test.iloc[perm_idx]
            y_perm = y_test[perm_idx]
        else:
            X_perm = X_test
            y_perm = y_test

        perm = permutation_importance(
            model,
            X_perm,
            y_perm,
            n_repeats=3,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        for feature, mean, std in zip(feature_cols, perm.importances_mean, perm.importances_std):
            importances.append(
                {
                    "model": name,
                    "feature": feature,
                    "importance_mean": float(mean),
                    "importance_std": float(std),
                }
            )
        pd.DataFrame(importances).sort_values(
            ["model", "importance_mean"], ascending=[True, False]
        ).to_csv(OUTPUT_DIR / "step03_permutation_importance.csv", index=False)

    pd.DataFrame(metrics).sort_values("test_concordance_index", ascending=False).to_csv(
        OUTPUT_DIR / "step03_ml_model_comparison.csv", index=False
    )
    pd.DataFrame(importances).sort_values(
        ["model", "importance_mean"], ascending=[True, False]
    ).to_csv(OUTPUT_DIR / "step03_permutation_importance.csv", index=False)
    if rsf_importances:
        pd.DataFrame(rsf_importances).sort_values(
            ["model", "importance_mean"], ascending=[True, False]
        ).to_csv(OUTPUT_DIR / "step03_rsf_permutation_importance.csv", index=False)

    feature_set_rows = []
    for set_name, cols in feature_set_columns(feature_cols).items():
        cols = [c for c in dict.fromkeys(cols) if c in feature_cols]
        if not cols:
            continue
        log_item("Feature-set comparison: %-20s %3s features", set_name, len(cols))
        model = RandomSurvivalForest(
            n_estimators=40,
            min_samples_split=20,
            min_samples_leaf=10,
            n_jobs=1,
            random_state=RANDOM_STATE,
        )
        try:
            model.fit(X_train[cols], y_train)
            cindex = float(model.score(X_test[cols], y_test))
            status = "ok"
            error = ""
        except Exception as exc:
            cindex = np.nan
            status = "failed"
            error = str(exc)
        feature_set_rows.append(
            {
                "feature_set": set_name,
                "model": "random_survival_forest",
                "test_concordance_index": cindex,
                "n_features": len(cols),
                "n_rows_used": int(len(X)),
                "status": status,
                "error": error,
            }
        )
    if feature_set_rows:
        pd.DataFrame(feature_set_rows).sort_values(
            "test_concordance_index", ascending=False
        ).to_csv(OUTPUT_DIR / "step03_feature_set_comparison.csv", index=False)

def save_step_04_advanced_evaluation(df: pd.DataFrame) -> None:
    # Bước 04: tạo plot, nhóm nguy cơ holdout, calibration và fixed-horizon AUC.
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test

    log_section("Step 04 - Plots, risk groups, calibration, fixed-horizon AUC")
    OUTPUT_DIR.mkdir(exist_ok=True)
    plots_dir = OUTPUT_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)
    mpl_cache_dir = OUTPUT_DIR / ".matplotlib"
    mpl_cache_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir.resolve()))
    os.environ.setdefault("XDG_CACHE_HOME", str(mpl_cache_dir.resolve()))

    import matplotlib.pyplot as plt

    model_df, feature_cols = build_model_matrix(df)
    log_progress("Fitting full penalized Cox with %s features", len(feature_cols))
    cox_fit_df = standardize_for_cox(
        model_df[[DURATION_COL, EVENT_COL] + feature_cols],
        feature_cols,
    )
    cph = None
    last_error = ""
    used_penalizer = None
    for penalizer in cox_penalizer_candidates(len(feature_cols)):
        try:
            candidate = CoxPHFitter(penalizer=penalizer)
            candidate.fit(
                cox_fit_df,
                duration_col=DURATION_COL,
                event_col=EVENT_COL,
            )
            cph = candidate
            used_penalizer = penalizer
            break
        except Exception as exc:
            last_error = str(exc)

    if cph is None:
        msg = (
            "Full penalized Cox failed in Step 04 because of collinearity/singular "
            f"matrix. Forest plot skipped. Error: {last_error}"
        )
        (OUTPUT_DIR / "step04_cox_forest_plot_SKIPPED.txt").write_text(
            msg, encoding="utf-8"
        )
        LOGGER.warning("  ! %s", msg)
    else:
        if used_penalizer is not None and used_penalizer > 0:
            log_progress("Used Cox penalizer %.2g for Step 04 forest plot", used_penalizer)
        summary = cph.summary.reset_index().rename(columns={"covariate": "term"}).copy()
        summary["abs_log_hr"] = summary["coef"].abs()
        top_terms = summary.sort_values("abs_log_hr", ascending=False).head(20).copy()
        top_terms.to_csv(OUTPUT_DIR / "step04_top20_cox_hazard_ratios.csv", index=False)
        log_item("Saved step04_top20_cox_hazard_ratios.csv")

        forest = top_terms.sort_values("exp(coef)")
        fig_height = max(6, 0.35 * len(forest))
        plt.figure(figsize=(10, fig_height))
        y = np.arange(len(forest))
        plt.errorbar(
            forest["exp(coef)"],
            y,
            xerr=[
                forest["exp(coef)"] - forest["exp(coef) lower 95%"],
                forest["exp(coef) upper 95%"] - forest["exp(coef)"],
            ],
            fmt="o",
            color="#1f77b4",
            ecolor="#7f7f7f",
            capsize=3,
        )
        plt.axvline(1.0, color="#d62728", linestyle="--", linewidth=1)
        plt.yticks(y, forest["term"])
        plt.xscale("log")
        plt.xlabel("Hazard ratio, log scale")
        plt.title("Top Cox Hazard Ratios")
        plt.tight_layout()
        plt.savefig(plots_dir / "step04_forest_plot_top20_cox.png", dpi=180)
        plt.close()
        log_item("Saved plots/step04_forest_plot_top20_cox.png")

    try:
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score
        from sksurv.ensemble import RandomSurvivalForest
        from sksurv.util import Surv
    except ImportError as exc:
        msg = (
            "Fixed-horizon AUC requires scikit-survival and scikit-learn. "
            f"Install requirements.txt. Import error: {exc}"
        )
        (OUTPUT_DIR / "step04_binary_auc_SKIPPED.txt").write_text(
            msg, encoding="utf-8"
        )
        LOGGER.warning("  ! %s", msg)
        return

    eval_df = model_df
    if len(eval_df) > MAX_ML_ROWS:
        sampled = eval_df.sample(n=MAX_ML_ROWS, random_state=RANDOM_STATE).index
        eval_df = eval_df.loc[sampled].copy()
        log_progress("Sampled %s rows for Step 04 RSF runtime control", MAX_ML_ROWS)

    X_full = eval_df[feature_cols].astype(float)
    y_full = Surv.from_dataframe(EVENT_COL, DURATION_COL, eval_df)
    (
        X_train_eval,
        X_test_eval,
        y_train_eval,
        y_test_eval,
        idx_train_eval,
        idx_test_eval,
    ) = train_test_split(
        X_full,
        y_full,
        eval_df.index,
        test_size=0.25,
        random_state=RANDOM_STATE,
    )
    rsf_eval = RandomSurvivalForest(
        n_estimators=50,
        min_samples_split=20,
        min_samples_leaf=10,
        n_jobs=1,
        random_state=RANDOM_STATE,
    )
    log_progress("Fitting Random Survival Forest for holdout risk groups/calibration")
    rsf_eval.fit(X_train_eval, y_train_eval)
    holdout_risk_scores = rsf_eval.predict(X_test_eval)
    risk = pd.Series(holdout_risk_scores, index=idx_test_eval, dtype=float)
    risk_df = eval_df.loc[idx_test_eval, [DURATION_COL, EVENT_COL]].copy()
    risk_df["risk_score"] = risk.values
    risk_df["risk_group"] = pd.qcut(
        risk_df["risk_score"],
        q=3,
        labels=["low", "medium", "high"],
        duplicates="drop",
    )
    risk_summary = (
        risk_df.groupby("risk_group", observed=False)
        .agg(
            n=(EVENT_COL, "count"),
            events=(EVENT_COL, "sum"),
            event_rate=(EVENT_COL, "mean"),
            median_risk_score=("risk_score", "median"),
        )
        .reset_index()
    )
    risk_summary.to_csv(OUTPUT_DIR / "step04_risk_group_summary.csv", index=False)
    log_item("Saved step04_risk_group_summary.csv")

    logrank = multivariate_logrank_test(
        risk_df[DURATION_COL], risk_df["risk_group"], risk_df[EVENT_COL]
    )
    pd.DataFrame(
        [
            {
                "comparison": "random_survival_forest risk groups",
                "test_statistic": float(logrank.test_statistic),
                "p": float(logrank.p_value),
            }
        ]
    ).to_csv(OUTPUT_DIR / "step04_risk_group_logrank.csv", index=False)

    plt.figure(figsize=(8, 6))
    kmf = KaplanMeierFitter()
    for group in ["low", "medium", "high"]:
        mask = risk_df["risk_group"] == group
        kmf.fit(
            risk_df.loc[mask, DURATION_COL],
            event_observed=risk_df.loc[mask, EVENT_COL],
            label=f"{group} risk",
        )
        kmf.plot_survival_function(ci_show=False)
    plt.title("Kaplan-Meier Curves by Predicted Risk Group")
    plt.xlabel("Days")
    plt.ylabel("Survival probability")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "step04_km_by_predicted_risk_group.png", dpi=180)
    plt.close()
    log_item("Saved plots/step04_km_by_predicted_risk_group.png")

    calibration_df = risk_df.copy()
    survival_functions = rsf_eval.predict_survival_function(X_test_eval)
    calibration_df["predicted_90d_risk"] = [
        1.0 - float(fn(90.0)) for fn in survival_functions
    ]
    calibration_df["risk_decile"] = pd.qcut(
        calibration_df["predicted_90d_risk"],
        q=10,
        labels=False,
        duplicates="drop",
    )
    calibration_rows = []
    for decile, group in calibration_df.groupby("risk_decile", observed=False):
        kmf.fit(group[DURATION_COL], event_observed=group[EVENT_COL])
        observed_risk = 1.0 - float(kmf.predict(90))
        calibration_rows.append(
            {
                "risk_decile": int(decile),
                "n": int(len(group)),
                "mean_predicted_90d_risk": float(group["predicted_90d_risk"].mean()),
                "observed_90d_risk_km": observed_risk,
            }
        )
    calibration_table = pd.DataFrame(calibration_rows)
    calibration_table.to_csv(OUTPUT_DIR / "step04_calibration_90d.csv", index=False)
    log_item("Saved step04_calibration_90d.csv")

    plt.figure(figsize=(6, 6))
    plt.plot(
        calibration_table["mean_predicted_90d_risk"],
        calibration_table["observed_90d_risk_km"],
        marker="o",
    )
    lim = max(
        calibration_table["mean_predicted_90d_risk"].max(),
        calibration_table["observed_90d_risk_km"].max(),
    )
    plt.plot([0, lim], [0, lim], linestyle="--", color="#7f7f7f")
    plt.xlabel("Mean predicted 90-day risk")
    plt.ylabel("Observed 90-day risk")
    plt.title("90-day Calibration Plot")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "step04_calibration_90d.png", dpi=180)
    plt.close()
    log_item("Saved plots/step04_calibration_90d.png")

    binary_auc_rows = []
    for horizon in [7.0, 14.0, 30.0, 60.0, 90.0]:
        y_binary = (y_test_eval[EVENT_COL]) & (
            y_test_eval[DURATION_COL].astype(float) <= horizon
        )
        n_events = int(y_binary.sum())
        n_nonevents = int((~y_binary).sum())
        if n_events == 0 or n_nonevents == 0:
            continue
        binary_auc_rows.append(
            {
                "model": "random_survival_forest",
                "horizon_days": horizon,
                "binary_roc_auc": float(roc_auc_score(y_binary, holdout_risk_scores)),
                "n_events_by_horizon": n_events,
                "n_nonevents_by_horizon": n_nonevents,
                "note": (
                    "Fixed-horizon discrimination metric for early mortality. "
                    "Used instead of IPCW time-dependent AUC because this dataset "
                    "is administratively censored at 90 days."
                ),
            }
        )
    if not binary_auc_rows:
        msg = (
            "Fixed-horizon AUC could not be computed because no horizon had both "
            "events and non-events in the test split."
        )
        (OUTPUT_DIR / "step04_binary_auc_SKIPPED.txt").write_text(msg, encoding="utf-8")
        LOGGER.warning("  ! %s", msg)
        return

    binary_auc_table = pd.DataFrame(binary_auc_rows).sort_values("horizon_days")
    binary_auc_table.to_csv(OUTPUT_DIR / "step04_binary_auc_by_horizon.csv", index=False)

    plt.figure(figsize=(7, 5))
    plt.plot(
        binary_auc_table["horizon_days"],
        binary_auc_table["binary_roc_auc"],
        marker="o",
    )
    plt.ylim(0.5, 1.0)
    plt.xlabel("Days")
    plt.ylabel("Binary ROC AUC")
    plt.title("Random Survival Forest Fixed-horizon AUC")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "step04_binary_auc_by_horizon.png", dpi=180)
    plt.close()

    stale_files = [
        OUTPUT_DIR / "step04_time_dependent_auc_SKIPPED.txt",
        OUTPUT_DIR / "step04_time_dependent_auc_candidate_errors.csv",
        OUTPUT_DIR / "step04_time_dependent_auc_skipped_times.csv",
        OUTPUT_DIR / "step04_time_dependent_auc.csv",
        plots_dir / "step04_time_dependent_auc.png",
    ]
    for stale_file in stale_files:
        if stale_file.exists():
            stale_file.unlink()

def save_final_project_summary(df: pd.DataFrame) -> None:
    # Bước 05: tổng hợp kết quả chính về cohort, model, AUC, risk group và importance.
    log_section("Step 05 - Final project summary")
    # ensure_output_dir()

    overview_path = OUTPUT_DIR / "step01_dataset_overview.json"
    ml_path = OUTPUT_DIR / "step03_ml_model_comparison.csv"
    feature_set_path = OUTPUT_DIR / "step03_feature_set_comparison.csv"
    auc_path = OUTPUT_DIR / "step04_binary_auc_by_horizon.csv"
    risk_path = OUTPUT_DIR / "step04_risk_group_summary.csv"
    importance_path = OUTPUT_DIR / "step03_rsf_permutation_importance.csv"

    overview = json.loads(overview_path.read_text(encoding="utf-8")) if overview_path.exists() else {}
    n_subjects = df["subject_id"].nunique() if "subject_id" in df.columns else "NA"
    lines = [
        "# Final Project Summary",
        "",
        "## Cohort",
        "",
        f"- Rows / ICU stays: {int(overview.get('n_rows', len(df))):,}",
        f"- Subjects: {overview.get('n_subjects', n_subjects)}",
        f"- 90-day deaths: {int(overview.get('n_events_90d', int(df[EVENT_COL].sum()))):,}",
        f"- 90-day mortality rate: {float(overview.get('event_rate_90d', float(df[EVENT_COL].mean()))):.4f}",
        f"- Longitudinal summary features detected: {len(overview.get('detected_longitudinal_summary_features', []))}",
        "",
        "## Outcomes Available",
        "",
    ]

    for horizon in [7, 28, 90]:
        event_col = f"death_{horizon}d"
        time_col = f"time_to_event_{horizon}d"
        if event_col in df.columns and time_col in df.columns:
            lines.append(
                f"- {horizon}-day mortality: {int(df[event_col].sum()):,} events "
                f"({float(df[event_col].mean()):.4f})"
            )
    lines.append("")

    if ml_path.exists():
        ml = pd.read_csv(ml_path)
        lines += ["## Model Comparison", ""]
        for _, row in ml.sort_values("test_concordance_index", ascending=False).iterrows():
            lines.append(
                f"- {row['model']}: C-index {row['test_concordance_index']:.4f} "
                f"({int(row['n_features'])} features, {int(row['n_rows_used'])} rows)"
            )
        lines.append("")

    if feature_set_path.exists():
        fs = pd.read_csv(feature_set_path)
        lines += ["## Feature Set Comparison", ""]
        for _, row in fs.sort_values("test_concordance_index", ascending=False).iterrows():
            lines.append(
                f"- {row['feature_set']}: C-index {row['test_concordance_index']:.4f} "
                f"({int(row['n_features'])} features)"
            )
        lines.append("")

    if auc_path.exists():
        auc = pd.read_csv(auc_path)
        lines += ["## Fixed-Horizon AUC", ""]
        for _, row in auc.iterrows():
            lines.append(
                f"- {int(row['horizon_days'])} days: AUC {row['binary_roc_auc']:.4f} "
                f"({int(row['n_events_by_horizon'])} events in holdout)"
            )
        lines.append("")

    if risk_path.exists():
        risk = pd.read_csv(risk_path)
        lines += ["## Holdout Risk Groups", ""]
        for _, row in risk.iterrows():
            lines.append(
                f"- {row['risk_group']}: {int(row['n'])} rows, "
                f"{int(row['events'])} events, event rate {row['event_rate']:.4f}"
            )
        lines.append("")

    if importance_path.exists():
        imp = pd.read_csv(importance_path).head(10)
        lines += ["## Top RSF Permutation Importance", ""]
        for _, row in imp.iterrows():
            lines.append(f"- {row['feature']}: {row['importance_mean']:.5f}")
        lines.append("")

    lines += [
        "## Current Interpretation",
        "",
        "This project uses ICU stays from MIMIC-IV with longitudinal vital/lab summaries "
        "from the first 24 ICU hours. Random Survival Forest is the strongest current "
        "model and is used for holdout risk stratification, calibration, and fixed-horizon "
        "early mortality AUC.",
        "",
    ]
    (OUTPUT_DIR / "final_project_summary.md").write_text("\n".join(lines), encoding="utf-8")
    log_done("Saved final_project_summary.md")


def parse_steps(raw_steps: Iterable[str]) -> set[str]:
    # Chuẩn hoá tham số step từ CLI như 1, 01, step01 hoặc all.
    if not raw_steps:
        return {"01", "02", "03", "04", "05"}
    out = set()
    for step in raw_steps:
        normalized = step.lower().replace("step", "").strip()
        if normalized in {"1", "01"}:
            out.add("01")
        elif normalized in {"2", "02"}:
            out.add("02")
        elif normalized in {"3", "03"}:
            out.add("03")
        elif normalized in {"4", "04"}:
            out.add("04")
        elif normalized in {"5", "05"}:
            out.add("05")
        elif normalized == "all":
            out.update({"01", "02", "03", "04", "05"})
        else:
            raise ValueError(f"Unknown step: {step}")
    return out


def main() -> None:
    # Điểm vào dòng lệnh để chạy các bước phân tích được chọn.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step",
        nargs="*",
        default=["all"],
        help="Steps to run: 01 02 03 04 05 or all",
    )
    parser.add_argument(
        "--data-file",
        default=None,
        help=(
            "Input cohort CSV. If omitted, uses cohort_icu_longitudinal_90d.csv "
            "from the current directory."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show debug-level terminal logs",
    )
    args = parser.parse_args()
    setup_logging(args.verbose)
    steps = parse_steps(args.step)

    data_file = Path(args.data_file) if args.data_file else default_data_file()
    log_item("Selected data file: %s", data_file)
    df = add_analysis_features(load_data(data_file))

    if "01" in steps:
        save_step_01_eda(df)
    if "02" in steps:
        save_step_02_cox(df)
    if "03" in steps:
        save_step_03_ml_survival(df)
    if "04" in steps:
        save_step_04_advanced_evaluation(df)
    if "05" in steps:
        save_final_project_summary(df)

    log_item("Outputs saved to %s", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
