from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DATA_FILE = Path("cohort_hf_survival_90d.csv")
OUTPUT_DIR = Path("outputs")
DURATION_COL = "time_to_event_90d"
EVENT_COL = "death_90d"
RANDOM_STATE = 42

BASE_FEATURES = ["anchor_age", "gender_male", "cci_without_hf"]

COMORBIDITY_FEATURES = [
    "myocardial_infarct",
    "peripheral_vascular_disease",
    "cerebrovascular_disease",
    "dementia",
    "chronic_pulmonary_disease",
    "rheumatic_disease",
    "peptic_ulcer_disease",
    "mild_liver_disease",
    "diabetes_without_cc",
    "diabetes_with_cc",
    "paraplegia",
    "renal_disease",
    "malignant_cancer",
    "severe_liver_disease",
    "metastatic_solid_tumor",
    "aids",
]

CATEGORICAL_FEATURES = [
    "admission_type",
    "insurance",
    "race_simple",
    "marital_status",
]


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)


def load_data(path: Path = DATA_FILE) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {DURATION_COL, EVENT_COL, "anchor_age", "gender", "cci_without_hf"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df


def simplify_race(value: object) -> str:
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
    out = df.copy()
    out["gender_male"] = out["gender"].map({"F": 0, "M": 1})
    out["age_group"] = pd.cut(
        out["anchor_age"],
        bins=[0, 65, 80, np.inf],
        labels=["<65", "65-79", ">=80"],
        right=False,
    )
    out["cci_group"] = pd.cut(
        out["cci_without_hf"],
        bins=[-np.inf, 3, 6, np.inf],
        labels=["0-2", "3-5", ">=6"],
        right=False,
    )
    out["race_simple"] = out["race"].map(simplify_race)
    out["marital_status"] = out["marital_status"].fillna("UNKNOWN")
    out["insurance"] = out["insurance"].fillna("UNKNOWN")
    out["admission_type"] = out["admission_type"].fillna("UNKNOWN")
    return out


def mortality_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    table = (
        df.groupby(group_col, dropna=False, observed=False)[EVENT_COL]
        .agg(n="count", events="sum", event_rate="mean")
        .reset_index()
        .sort_values(["event_rate", "n"], ascending=[False, False])
    )
    table.insert(0, "variable", group_col)
    return table


def save_step_01_eda(df: pd.DataFrame) -> None:
    ensure_output_dir()
    overview = {
        "n_rows": int(len(df)),
        "n_subjects": int(df["subject_id"].nunique()) if "subject_id" in df else None,
        "n_admissions": int(df["hadm_id"].nunique()) if "hadm_id" in df else None,
        "n_columns": int(df.shape[1]),
        "n_events_90d": int(df[EVENT_COL].sum()),
        "event_rate_90d": float(df[EVENT_COL].mean()),
        "median_followup_days": float(df[DURATION_COL].median()),
    }
    (OUTPUT_DIR / "step01_dataset_overview.json").write_text(
        json.dumps(overview, indent=2), encoding="utf-8"
    )

    missing = (
        df.isna()
        .sum()
        .rename("missing")
        .reset_index()
        .rename(columns={"index": "column"})
    )
    missing["missing_rate"] = missing["missing"] / len(df)
    missing.sort_values("missing_rate", ascending=False).to_csv(
        OUTPUT_DIR / "step01_missing_values.csv", index=False
    )

    group_cols = [
        "gender",
        "age_group",
        "cci_group",
        "admission_type",
        "insurance",
        "race_simple",
        "marital_status",
    ] + [c for c in COMORBIDITY_FEATURES if c in df.columns]
    tables = [mortality_table(df, col) for col in group_cols if col in df.columns]
    pd.concat(tables, ignore_index=True).to_csv(
        OUTPUT_DIR / "step01_mortality_by_group.csv", index=False
    )


def build_model_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    numeric_binary = [
        c
        for c in ["anchor_age", "gender_male", "cci_without_hf"] + COMORBIDITY_FEATURES
        if c in df.columns
    ]
    model_df = df[[DURATION_COL, EVENT_COL] + numeric_binary + CATEGORICAL_FEATURES].copy()
    model_df = pd.get_dummies(
        model_df,
        columns=CATEGORICAL_FEATURES,
        drop_first=True,
        dtype=float,
    )
    model_df = model_df.dropna()

    feature_cols = [c for c in model_df.columns if c not in {DURATION_COL, EVENT_COL}]
    low_variance = [c for c in feature_cols if model_df[c].nunique(dropna=True) <= 1]
    if low_variance:
        model_df = model_df.drop(columns=low_variance)
        feature_cols = [c for c in feature_cols if c not in low_variance]
    return model_df, feature_cols


def cox_summary_frame(model, model_name: str) -> pd.DataFrame:
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
    from lifelines import CoxPHFitter
    from lifelines.statistics import multivariate_logrank_test, proportional_hazard_test

    ensure_output_dir()

    logrank_rows = []
    for col in ["gender", "age_group", "cci_group", "admission_type", "insurance", "race_simple"]:
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
    model_specs = {
        "cox_baseline_3vars": [c for c in BASE_FEATURES if c in model_df.columns],
        "cox_demographics": [
            c
            for c in feature_cols
            if c in BASE_FEATURES
            or c.startswith("admission_type_")
            or c.startswith("insurance_")
            or c.startswith("race_simple_")
            or c.startswith("marital_status_")
        ],
        "cox_comorbidity_split": [
            c
            for c in ["anchor_age", "gender_male"] + COMORBIDITY_FEATURES
            if c in model_df.columns
        ],
        "cox_full_penalized": feature_cols,
    }

    summaries = []
    ph_rows = []
    for name, cols in model_specs.items():
        if not cols:
            continue
        fit_df = model_df[[DURATION_COL, EVENT_COL] + cols].copy()
        penalizer = 0.01 if len(cols) > 10 else 0.0
        cph = CoxPHFitter(penalizer=penalizer)
        cph.fit(fit_df, duration_col=DURATION_COL, event_col=EVENT_COL)
        summaries.append(cox_summary_frame(cph, name))

        ph = proportional_hazard_test(cph, fit_df, time_transform="rank").summary
        ph = ph.reset_index().rename(columns={"index": "term"})
        ph.insert(0, "model", name)
        ph_rows.append(ph)

    pd.concat(summaries, ignore_index=True).to_csv(
        OUTPUT_DIR / "step02_cox_model_comparison.csv", index=False
    )
    pd.concat(ph_rows, ignore_index=True).to_csv(
        OUTPUT_DIR / "step02_ph_assumption_tests.csv", index=False
    )


def save_step_03_ml_survival(df: pd.DataFrame) -> None:
    try:
        from sklearn.inspection import permutation_importance
        from sklearn.model_selection import train_test_split
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
        print(msg)
        return

    ensure_output_dir()
    model_df, feature_cols = build_model_matrix(df)
    X = model_df[feature_cols].astype(float)
    y = Surv.from_dataframe(EVENT_COL, DURATION_COL, model_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE
    )

    models = {
        "coxnet": CoxnetSurvivalAnalysis(l1_ratio=0.5, alpha_min_ratio=0.01),
        "random_survival_forest": RandomSurvivalForest(
            n_estimators=50,
            min_samples_split=20,
            min_samples_leaf=10,
            n_jobs=1,
            random_state=RANDOM_STATE,
        ),
    }

    metrics = []
    importances = []
    importance_notes = [
        {
            "model": "gradient_boosting_survival",
            "note": (
                "Not run in the default Step 03 because it was too slow on the "
                "full dataset in the local environment. Keep it as an optional "
                "extension after the report baseline is stable."
            ),
        }
    ]
    for name, model in models.items():
        model.fit(X_train, y_train)
        cindex = float(model.score(X_test, y_test))
        metrics.append({"model": name, "test_concordance_index": cindex})
        pd.DataFrame(metrics).sort_values(
            "test_concordance_index", ascending=False
        ).to_csv(OUTPUT_DIR / "step03_ml_model_comparison.csv", index=False)

        if name != "coxnet":
            importance_notes.append(
                {
                    "model": name,
                    "note": (
                        "Permutation importance skipped in the default pipeline "
                        "because it is slow and disk-intensive on this dataset. "
                        "Use Cox hazard ratios and Coxnet permutation importance "
                        "for the first explainability report."
                    ),
                }
            )
            continue

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
    if importance_notes:
        pd.DataFrame(importance_notes).to_csv(
            OUTPUT_DIR / "step03_importance_notes.csv", index=False
        )


def save_step_04_deep_survival_notes() -> None:
    ensure_output_dir()
    text = """# Step 04 - Deep survival next step

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
"""
    (OUTPUT_DIR / "step04_deep_survival_next_steps.md").write_text(text, encoding="utf-8")


def save_step_05_advanced_evaluation(df: pd.DataFrame) -> None:
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test

    ensure_output_dir()
    plots_dir = OUTPUT_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)

    import matplotlib.pyplot as plt

    model_df, feature_cols = build_model_matrix(df)
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(
        model_df[[DURATION_COL, EVENT_COL] + feature_cols],
        duration_col=DURATION_COL,
        event_col=EVENT_COL,
    )

    summary = cph.summary.reset_index().rename(columns={"covariate": "term"}).copy()
    summary["abs_log_hr"] = summary["coef"].abs()
    top_terms = summary.sort_values("abs_log_hr", ascending=False).head(20).copy()
    top_terms.to_csv(OUTPUT_DIR / "step05_top20_cox_hazard_ratios.csv", index=False)

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
    plt.savefig(plots_dir / "step05_forest_plot_top20_cox.png", dpi=180)
    plt.close()

    risk = cph.predict_partial_hazard(model_df[feature_cols]).astype(float)
    risk_df = model_df[[DURATION_COL, EVENT_COL]].copy()
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
    risk_summary.to_csv(OUTPUT_DIR / "step05_risk_group_summary.csv", index=False)

    logrank = multivariate_logrank_test(
        risk_df[DURATION_COL], risk_df["risk_group"], risk_df[EVENT_COL]
    )
    pd.DataFrame(
        [
            {
                "comparison": "cox_full_penalized risk groups",
                "test_statistic": float(logrank.test_statistic),
                "p": float(logrank.p_value),
            }
        ]
    ).to_csv(OUTPUT_DIR / "step05_risk_group_logrank.csv", index=False)

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
    plt.savefig(plots_dir / "step05_km_by_predicted_risk_group.png", dpi=180)
    plt.close()

    survival_at_90 = cph.predict_survival_function(model_df[feature_cols], times=[90]).T
    calibration_df = risk_df.copy()
    calibration_df["predicted_90d_risk"] = 1.0 - survival_at_90.iloc[:, 0].values
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
    calibration_table.to_csv(OUTPUT_DIR / "step05_calibration_90d.csv", index=False)

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
    plt.savefig(plots_dir / "step05_calibration_90d.png", dpi=180)
    plt.close()

    try:
        from sklearn.model_selection import train_test_split
        from sksurv.ensemble import RandomSurvivalForest
        from sksurv.metrics import cumulative_dynamic_auc
        from sksurv.util import Surv
    except ImportError as exc:
        msg = (
            "Time-dependent AUC requires scikit-survival. "
            f"Install requirements-ml.txt. Import error: {exc}"
        )
        (OUTPUT_DIR / "step05_time_dependent_auc_SKIPPED.txt").write_text(
            msg, encoding="utf-8"
        )
        return

    X = model_df[feature_cols].astype(float)
    y_surv = Surv.from_dataframe(EVENT_COL, DURATION_COL, model_df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_surv, test_size=0.25, random_state=RANDOM_STATE
    )
    rsf = RandomSurvivalForest(
        n_estimators=50,
        min_samples_split=20,
        min_samples_leaf=10,
        n_jobs=1,
        random_state=RANDOM_STATE,
    )
    rsf.fit(X_train, y_train)
    risk_scores = rsf.predict(X_test)
    times = np.array([30.0, 60.0, 75.0])
    try:
        auc, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_scores, times)
    except ValueError as exc:
        msg = (
            "Time-dependent AUC could not be computed with the current censoring "
            f"distribution. Error: {exc}"
        )
        (OUTPUT_DIR / "step05_time_dependent_auc_SKIPPED.txt").write_text(
            msg, encoding="utf-8"
        )
        return
    pd.DataFrame(
        {
            "model": "random_survival_forest",
            "time_days": times,
            "time_dependent_auc": auc,
            "mean_auc": float(mean_auc),
        }
    ).to_csv(OUTPUT_DIR / "step05_time_dependent_auc.csv", index=False)

    plt.figure(figsize=(7, 5))
    plt.plot(times, auc, marker="o")
    plt.ylim(0.5, 1.0)
    plt.xlabel("Days")
    plt.ylabel("Time-dependent AUC")
    plt.title("Random Survival Forest Time-dependent AUC")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "step05_time_dependent_auc.png", dpi=180)
    plt.close()


def parse_steps(raw_steps: Iterable[str]) -> set[str]:
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step",
        nargs="*",
        default=["all"],
        help="Steps to run: 01 02 03 04 05 or all",
    )
    args = parser.parse_args()
    steps = parse_steps(args.step)

    df = add_analysis_features(load_data())

    if "01" in steps:
        save_step_01_eda(df)
    if "02" in steps:
        save_step_02_cox(df)
    if "03" in steps:
        save_step_03_ml_survival(df)
    if "04" in steps:
        save_step_04_deep_survival_notes()
    if "05" in steps:
        save_step_05_advanced_evaluation(df)

    print(f"Done. Outputs saved to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
