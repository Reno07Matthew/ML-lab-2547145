#!/usr/bin/env python3
"""Reproduce EDA, leakage-safe model comparison, and untouched-test evaluation."""

from __future__ import annotations

import argparse
import json
import os
import re
import warnings
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
MPL_CACHE = PROJECT / ".cache" / "matplotlib"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import joblib
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


RANDOM_STATE = 42
TARGET = "_MICHD"
WEIGHT = "_LLCPWT"
ID_COLUMNS = ["_STATE", "SEQNO"]
LIFESTYLE = [
    "_AGE80", "_SEX", "GENHLTH", "PHYSHLTH", "MENTHLTH", "EXERANY2",
    "_BMI5", "_SMOKER3", "DRNKANY6", "_TOTINDA", "_EDUCAG", "_INCOMG1",
]
EXTENDED_ADDITIONS = ["BPHIGH6", "cholesterol_status", "DIABETE4", "CHCKDNY2", "DIFFWALK"]
FEATURE_CONFIGS = {"lifestyle_only": LIFESTYLE, "extended_low_cost": LIFESTYLE + EXTENDED_ADDITIONS}
NUMERIC = ["_AGE80", "PHYSHLTH", "MENTHLTH", "_BMI5"]
CATEGORICAL = [c for c in FEATURE_CONFIGS["extended_low_cost"] if c not in NUMERIC]
RAW_COLUMNS = list(dict.fromkeys(ID_COLUMNS + [TARGET, WEIGHT, "CHOLCHK3", "TOLDHI3"] + LIFESTYLE + [c for c in EXTENDED_ADDITIONS if c != "cholesterol_status"]))
FORBIDDEN = {"CVDINFR4", "CVDCRHD4", TARGET, *[f"HASYMP{i}" for i in range(1, 7)], WEIGHT}
CSV_ONLY_UNDOCUMENTED = ["BIRTHSEX", "CELSXBRT", "LNDSXBRT", "RCSGEND1", "RCSXBRTH", "TRNSGNDR"]
CURRENT_ONLY_MISSING_FROM_CSV = ["RCSBORG1"]
TREATMENTS = ["none", "class_weight", "smotenc_0.10", "smotenc_0.25"]
MODEL_NAMES = ["logistic_regression", "random_forest", "catboost", "extra_trees"]
HYPERPARAMETERS = {
    "logistic_regression": [{"C": 0.1}, {"C": 1.0}],
    "random_forest": [
        {"n_estimators": 160, "max_depth": 12, "min_samples_leaf": 4},
        {"n_estimators": 240, "max_depth": None, "min_samples_leaf": 3},
    ],
    "catboost": [
        {"iterations": 160, "depth": 5, "learning_rate": 0.08},
        {"iterations": 240, "depth": 7, "learning_rate": 0.06},
    ],
    "extra_trees": [
        {"n_estimators": 160, "max_depth": 14, "min_samples_leaf": 3},
        {"n_estimators": 240, "max_depth": None, "min_samples_leaf": 2},
    ],
}
VALUE_LABELS = {
    "_SEX": {1: "Male", 2: "Female"},
    "GENHLTH": {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"},
    "EXERANY2": {1: "Yes", 2: "No"},
    "_SMOKER3": {1: "Every day", 2: "Some days", 3: "Former", 4: "Never"},
    "BPHIGH6": {1: "Yes", 2: "Pregnancy only", 3: "No", 4: "Borderline/elevated"},
    "DIABETE4": {1: "Yes", 2: "Pregnancy only", 3: "No", 4: "Pre-diabetes/borderline"},
    "CHCKDNY2": {1: "Yes", 2: "No"},
    "DIFFWALK": {1: "Yes", 2: "No"},
}


def derive_cholesterol_status(frame: pd.DataFrame) -> pd.Series:
    """Apply the official CHOLCHK3 -> TOLDHI3 skip logic without imputation."""
    checked = frame["CHOLCHK3"].isin([2, 3, 4, 5, 6, 8])
    result = pd.Series("missing", index=frame.index, dtype="object")
    result[frame["CHOLCHK3"].eq(1)] = "never_checked"
    result[frame["CHOLCHK3"].eq(7) | (checked & frame["TOLDHI3"].eq(7))] = "unknown_not_sure"
    result[frame["CHOLCHK3"].eq(9) | (checked & frame["TOLDHI3"].eq(9))] = "refused"
    result[checked & frame["TOLDHI3"].eq(1)] = "high"
    result[checked & frame["TOLDHI3"].eq(2)] = "not_high"
    return result


def load_cohort(data_path: Path) -> pd.DataFrame:
    header = pd.read_csv(data_path, nrows=0).columns.tolist()
    missing = sorted(set(RAW_COLUMNS) - set(header))
    if missing:
        raise ValueError(f"Required documented columns are absent: {missing}")
    assert not set(FEATURE_CONFIGS["extended_low_cost"]) & FORBIDDEN
    frame = pd.read_csv(data_path, usecols=RAW_COLUMNS, low_memory=False)
    frame = frame[frame["_AGE80"].between(18, 39) & frame[TARGET].isin([1, 2])].copy()
    frame["target"] = frame[TARGET].eq(1).astype("int8")
    frame["participant_id"] = (
        frame["_STATE"].astype(int).astype(str).str.zfill(2)
        + "-"
        + frame["SEQNO"].astype(str).str.extract(r"(\d{10})", expand=False)
    )
    if frame["participant_id"].isna().any() or frame["participant_id"].duplicated().any():
        raise ValueError("Composite participant IDs are missing or duplicated")

    invalid = {
        "GENHLTH": [7, 9], "PHYSHLTH": [77, 99], "MENTHLTH": [77, 99],
        "EXERANY2": [7, 9], "_SMOKER3": [9], "DRNKANY6": [7, 9],
        "_TOTINDA": [9], "_EDUCAG": [9], "_INCOMG1": [9], "BPHIGH6": [7, 9],
        "DIABETE4": [7, 9], "CHCKDNY2": [7, 9], "DIFFWALK": [7, 9],
    }
    for column, codes in invalid.items():
        frame[column] = frame[column].replace(codes, np.nan)
    frame["PHYSHLTH"] = frame["PHYSHLTH"].replace(88, 0)
    frame["MENTHLTH"] = frame["MENTHLTH"].replace(88, 0)
    frame["_BMI5"] = frame["_BMI5"] / 100
    frame["cholesterol_status"] = derive_cholesterol_status(frame)
    for column in CATEGORICAL:
        if column != "cholesterol_status":
            frame[column] = frame[column].map(lambda value: str(int(value)) if pd.notna(value) else np.nan)
    assert len(frame) == 99_114 and int(frame["target"].sum()) == 1_050
    return frame


def write_column_discrepancy(data_path: Path, report_dir: Path) -> None:
    csv_columns = pd.read_csv(data_path, nrows=0).columns.tolist()
    rows = [
        *({"column": c, "status": "CSV only; absent from current 345-variable layout", "used_for_modelling": False} for c in CSV_ONLY_UNDOCUMENTED),
        *({"column": c, "status": "Current documented layout only; absent from supplied CSV", "used_for_modelling": False} for c in CURRENT_ONLY_MISSING_FROM_CSV),
    ]
    pd.DataFrame(rows).to_csv(report_dir / "column_discrepancy.csv", index=False)
    explanation = f"""# 350-versus-345 column discrepancy

- Supplied `LLCP2023.csv`: {len(csv_columns)} columns.
- Current CDC February 2025 SAS Transport release: 345 variables.
- CSV-only fields: {', '.join(f'`{c}`' for c in CSV_ONLY_UNDOCUMENTED)}.
- Current-layout field absent from the CSV: `RCSBORG1` (Sex of child: boy/girl).
- Net difference: six legacy fields removed and one replacement field added, producing 350 - 6 + 1 = 345.

CDC states that the February 2025 file was modified to comply with Presidential executive orders and that removed questions can cause apparently inconsistent missing values. No field-by-field CDC release note was located. The six CSV-only names are sex-at-birth/gender-related items, so that dataset-level explanation is consistent with the observed difference; this is an inference, not a more specific CDC statement.

All modelling uses an explicit whitelist of requested variables that appear in the current documented layout. Every CSV-only field, `RCSBORG1`, `_LLCPWT`, and all forbidden leakage fields are excluded from predictors.

Sources:
- https://www.cdc.gov/brfss/annual_data/annual_2023.html
- https://www.cdc.gov/brfss/annual_data/2023/llcp_varlayout_23_onecolumn.html
"""
    (report_dir / "column_discrepancy.md").write_text(explanation)


def write_model_dictionary(report_dir: Path) -> None:
    descriptions = {
        "_AGE80": "Imputed age, restricted to 18-39", "_SEX": "Calculated sex variable",
        "GENHLTH": "Self-rated general health", "PHYSHLTH": "Physical-health days not good in past 30 days",
        "MENTHLTH": "Mental-health days not good in past 30 days", "EXERANY2": "Any exercise in past 30 days",
        "_BMI5": "Computed BMI with two implied decimal places", "_SMOKER3": "Computed smoking status",
        "DRNKANY6": "Any alcohol in past 30 days", "_TOTINDA": "Computed leisure-time physical activity",
        "_EDUCAG": "Computed education category", "_INCOMG1": "Computed income category",
        "BPHIGH6": "History of high blood pressure", "cholesterol_status": "CHOLCHK3/TOLDHI3 skip-logic status",
        "DIABETE4": "Diabetes history", "CHCKDNY2": "Kidney-disease history", "DIFFWALK": "Serious difficulty walking/climbing stairs",
    }
    rows = [{
        "field": TARGET, "role": "target", "feature_config": "neither", "data_type": "binary",
        "description": "Previously reported CHD or MI", "cleaning": "1 -> positive; 2 -> negative; blank excluded",
    }]
    for feature in FEATURE_CONFIGS["extended_low_cost"]:
        cleaning = "Training-fold categorical missing category"
        if feature in NUMERIC: cleaning = "Training-fold median imputation and scaling"
        if feature in {"PHYSHLTH", "MENTHLTH"}: cleaning = "88 -> 0; 77/99 -> missing; then training-fold median imputation and scaling"
        if feature == "_BMI5": cleaning = "Divide by 100; training-fold median imputation and scaling"
        if feature == "cholesterol_status": cleaning = "Deterministic official skip logic; no ordinary imputation"
        rows.append({
            "field": feature, "role": "predictor", "feature_config": "both" if feature in LIFESTYLE else "extended_low_cost",
            "data_type": "numeric" if feature in NUMERIC else "categorical", "description": descriptions[feature], "cleaning": cleaning,
        })
    rows.append({
        "field": WEIGHT, "role": "EDA survey weight only", "feature_config": "neither", "data_type": "numeric",
        "description": "Final BRFSS survey weight", "cleaning": "Never supplied to a predictor pipeline",
    })
    pd.DataFrame(rows).to_csv(report_dir / "data_dictionary.csv", index=False)


def save_splits(frame: pd.DataFrame, output_dir: Path) -> dict[str, np.ndarray]:
    train_idx, held_idx = train_test_split(
        np.arange(len(frame)), test_size=0.30, stratify=frame["target"], random_state=RANDOM_STATE
    )
    validation_idx, test_idx = train_test_split(
        held_idx, test_size=0.50, stratify=frame.iloc[held_idx]["target"], random_state=RANDOM_STATE
    )
    splits = {"train": train_idx, "validation": validation_idx, "test": test_idx}
    split_dir = output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    combined = []
    for name, indices in splits.items():
        ids = frame.iloc[indices][["participant_id", "_STATE", "SEQNO", "target"]].copy()
        ids.insert(1, "split", name)
        ids.to_csv(split_dir / f"{name}_ids.csv", index=False)
        combined.append(ids)
    pd.concat(combined).to_csv(split_dir / "all_split_ids.csv", index=False)
    assert not set(frame.iloc[train_idx]["participant_id"]) & set(frame.iloc[test_idx]["participant_id"])
    return splits


def _weighted_prevalence(group: pd.DataFrame) -> float:
    return float(np.average(group["target"], weights=group[WEIGHT]))


def prevalence_table(frame: pd.DataFrame, column: str, labels: dict | None = None, order: list[str] | None = None) -> pd.DataFrame:
    temp = frame[[column, "target", WEIGHT]].copy()
    if labels:
        temp[column] = pd.to_numeric(temp[column], errors="coerce").map(labels)
        order = order or list(labels.values())
    if order:
        temp[column] = pd.Categorical(temp[column], categories=order, ordered=True)
    rows = []
    for value, group in temp.dropna(subset=[column]).groupby(column, observed=True):
        rows.append({
            "group": str(value), "n": len(group), "positive_n": int(group["target"].sum()),
            "unweighted_prevalence": group["target"].mean(), "weighted_prevalence": _weighted_prevalence(group),
        })
    return pd.DataFrame(rows)


def _plot_prevalence(ax, table: pd.DataFrame, title: str) -> None:
    x = np.arange(len(table))
    ax.bar(x - 0.18, 100 * table["unweighted_prevalence"], 0.36, label="Unweighted")
    ax.bar(x + 0.18, 100 * table["weighted_prevalence"], 0.36, label="Weighted")
    ax.set_xticks(x, table["group"], rotation=30, ha="right")
    ax.set_ylabel("Reported CHD/MI prevalence (%)")
    ax.set_title(title)
    ax.legend(fontsize=8)


def create_eda(frame: pd.DataFrame, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    unweighted = frame["target"].mean()
    weighted = _weighted_prevalence(frame)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(["Negative", "Positive"], [(frame.target == 0).sum(), frame.target.sum()], color=["#4C78A8", "#E45756"])
    axes[0].set_yscale("log"); axes[0].set_title("Analytic class counts (log scale)"); axes[0].set_ylabel("Participants")
    axes[1].bar(["Unweighted", "Survey-weighted"], [100 * unweighted, 100 * weighted], color=["#72B7B2", "#F2CF5B"])
    axes[1].set_title("Reported premature CHD/MI prevalence"); axes[1].set_ylabel("Percent")
    fig.tight_layout(); fig.savefig(output_dir / "01_class_prevalence.png", dpi=180); plt.close(fig)

    missing = frame[FEATURE_CONFIGS["extended_low_cost"]].isna().mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6)); ax.barh(missing.index[::-1], 100 * missing.values[::-1], color="#4C78A8")
    ax.set_xlabel("Missing after official-code cleaning (%)"); ax.set_title("Predictor missingness (cholesterol skip logic already resolved)")
    fig.tight_layout(); fig.savefig(output_dir / "02_missingness.png", dpi=180); plt.close(fig)

    age = frame.copy(); age["age_group"] = pd.cut(age["_AGE80"], [17, 24, 29, 34, 39], labels=["18-24", "25-29", "30-34", "35-39"])
    bmi = frame.copy(); bmi["bmi_group"] = pd.cut(bmi["_BMI5"], [0, 18.5, 25, 30, np.inf], labels=["<18.5", "18.5-24.9", "25.0-29.9", "30+"])
    physical = frame.copy(); physical["physical_days_group"] = pd.cut(physical["PHYSHLTH"], [-1, 0, 5, 13, 29, 30], labels=["0", "1-5", "6-13", "14-29", "30"])
    panels = [
        (prevalence_table(age, "age_group"), "Age group"),
        (prevalence_table(frame, "_SEX", VALUE_LABELS["_SEX"]), "Sex"),
        (prevalence_table(frame, "_SMOKER3", VALUE_LABELS["_SMOKER3"]), "Smoking status"),
        (prevalence_table(bmi, "bmi_group"), "BMI group"),
        (prevalence_table(frame, "EXERANY2", VALUE_LABELS["EXERANY2"]), "Exercise in past 30 days"),
        (prevalence_table(frame, "GENHLTH", VALUE_LABELS["GENHLTH"]), "General health"),
        (prevalence_table(physical, "physical_days_group"), "Physical-health days not good"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(17, 14))
    for ax, (table, title) in zip(axes.flat, panels): _plot_prevalence(ax, table, title)
    for ax in axes.flat[len(panels):]: ax.axis("off")
    fig.tight_layout(); fig.savefig(output_dir / "03_lifestyle_relationships.png", dpi=180); plt.close(fig)

    health_panels = [
        (prevalence_table(frame, "BPHIGH6", VALUE_LABELS["BPHIGH6"]), "Blood-pressure history"),
        (prevalence_table(frame, "cholesterol_status", order=["high", "not_high", "never_checked", "unknown_not_sure", "refused"]), "Cholesterol status"),
        (prevalence_table(frame, "DIABETE4", VALUE_LABELS["DIABETE4"]), "Diabetes history"),
        (prevalence_table(frame, "CHCKDNY2", VALUE_LABELS["CHCKDNY2"]), "Kidney-disease history"),
        (prevalence_table(frame, "DIFFWALK", VALUE_LABELS["DIFFWALK"]), "Difficulty walking"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for ax, (table, title) in zip(axes.flat, health_panels): _plot_prevalence(ax, table, title)
    for ax in axes.flat[len(health_panels):]: ax.axis("off")
    fig.tight_layout(); fig.savefig(output_dir / "04_health_history_relationships.png", dpi=180); plt.close(fig)

    summary = {
        "rows": len(frame), "positive_n": int(frame.target.sum()), "negative_n": int((1 - frame.target).sum()),
        "unweighted_positive_pct": 100 * unweighted, "survey_weighted_positive_pct": 100 * weighted,
        "cholesterol_status_counts": frame["cholesterol_status"].value_counts().to_dict(),
        "missing_pct": (100 * missing).round(4).to_dict(),
    }
    (output_dir / "eda_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def make_pipeline(config: str, model_name: str, params: dict, treatment: str, voting_weights: tuple[int, int, int] | None = None) -> Pipeline:
    features = FEATURE_CONFIGS[config]
    numeric = [c for c in NUMERIC if c in features]
    categorical = [c for c in features if c not in numeric]
    before_sampling = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), categorical),
    ], sparse_threshold=0)
    category_indices = list(range(len(numeric), len(numeric) + len(categorical)))
    sampler: str | SMOTENC = "passthrough"
    if treatment.startswith("smotenc"):
        sampler = SMOTENC(category_indices, sampling_strategy=float(treatment.rsplit("_", 1)[1]), random_state=RANDOM_STATE)
    after_sampling = ColumnTransformer([
        ("numeric", "passthrough", list(range(len(numeric)))),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), category_indices),
    ])
    estimator = make_estimator(model_name, params, treatment, voting_weights)
    return Pipeline([("before_sampling", before_sampling), ("sampler", sampler), ("after_sampling", after_sampling), ("model", estimator)])


def make_estimator(model_name: str, params: dict, treatment: str, voting_weights=None):
    weighted = treatment == "class_weight"
    if model_name == "dummy":
        return DummyClassifier(strategy="prior", random_state=RANDOM_STATE)
    if model_name == "logistic_regression":
        return LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced" if weighted else None, random_state=RANDOM_STATE, **params)
    if model_name == "random_forest":
        return RandomForestClassifier(class_weight="balanced" if weighted else None, n_jobs=2, random_state=RANDOM_STATE, **params)
    if model_name == "extra_trees":
        return ExtraTreesClassifier(class_weight="balanced" if weighted else None, n_jobs=2, random_state=RANDOM_STATE, **params)
    if model_name == "catboost":
        return CatBoostClassifier(
            loss_function="Logloss", verbose=False, allow_writing_files=False, random_seed=RANDOM_STATE,
            thread_count=2, auto_class_weights="Balanced" if weighted else None, **params,
        )
    if model_name == "soft_voting":
        selected = params
        return VotingClassifier(
            estimators=[
                ("lr", make_estimator("logistic_regression", selected["logistic_regression"], treatment)),
                ("et", make_estimator("extra_trees", selected["extra_trees"], treatment)),
                ("cb", make_estimator("catboost", selected["catboost"], treatment)),
            ], voting="soft", weights=voting_weights, n_jobs=1,
        )
    raise KeyError(model_name)


def select_threshold(y_true: pd.Series | np.ndarray, probability: np.ndarray) -> dict:
    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    f2 = np.divide(5 * precision[:-1] * recall[:-1], 4 * precision[:-1] + recall[:-1], out=np.zeros_like(thresholds), where=(4 * precision[:-1] + recall[:-1]) > 0)
    index = int(np.nanargmax(f2))
    return {"threshold": float(thresholds[index]), "f2": float(f2[index]), "precision": float(precision[index]), "recall": float(recall[index])}


def probability_metrics(y_true, probability, threshold: float) -> dict:
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, prediction), "precision": precision_score(y_true, prediction, zero_division=0),
        "recall": recall_score(y_true, prediction, zero_division=0), "specificity": tn / (tn + fp),
        "positive_class_f1": f1_score(y_true, prediction, zero_division=0),
        "macro_f1": f1_score(y_true, prediction, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, prediction, average="weighted", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, prediction), "roc_auc": roc_auc_score(y_true, probability),
        "pr_auc": average_precision_score(y_true, probability), "brier_score": brier_score_loss(y_true, probability),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def bootstrap_intervals(y_true, probability, threshold: float, iterations: int = 1000) -> dict:
    y = np.asarray(y_true); rng = np.random.default_rng(RANDOM_STATE); values = []
    for _ in range(iterations):
        indices = rng.integers(0, len(y), len(y)); sample_y = y[indices]; sample_p = probability[indices]
        if sample_y.min() == sample_y.max():
            continue
        prediction = sample_p >= threshold
        values.append((recall_score(sample_y, prediction), f1_score(sample_y, prediction, zero_division=0), average_precision_score(sample_y, sample_p)))
    array = np.asarray(values)
    result = {}
    for index, name in enumerate(["recall", "positive_class_f1", "pr_auc"]):
        result[f"{name}_ci_low"], result[f"{name}_ci_high"] = np.quantile(array[:, index], [0.025, 0.975])
    return result


def validation_score(y_true, probability) -> dict:
    selected = select_threshold(y_true, probability)
    return {**selected, "pr_auc": average_precision_score(y_true, probability), "roc_auc": roc_auc_score(y_true, probability), "brier_score": brier_score_loss(y_true, probability)}


def run_model_comparison(frame: pd.DataFrame, splits: dict[str, np.ndarray], output_dir: Path) -> dict:
    report_dir = output_dir / "reports"; model_dir = output_dir / "models"; figure_dir = output_dir / "figures" / "models"
    for directory in [report_dir, model_dir, figure_dir]: directory.mkdir(parents=True, exist_ok=True)
    train = frame.iloc[splits["train"]]; validation = frame.iloc[splits["validation"]]; test = frame.iloc[splits["test"]]
    y_train, y_validation = train.target, validation.target
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "precision", "recall", "f1", "f1_macro", "balanced_accuracy", "roc_auc", "average_precision"]

    hyper_rows = []; selected_params = {}
    for config in FEATURE_CONFIGS:
        selected_params[config] = {}
        for model_name in MODEL_NAMES:
            candidates = []
            for params in HYPERPARAMETERS[model_name]:
                pipeline = make_pipeline(config, model_name, params, "none")
                pipeline.fit(train[FEATURE_CONFIGS[config]], y_train)
                probability = pipeline.predict_proba(validation[FEATURE_CONFIGS[config]])[:, 1]
                row = {"feature_config": config, "model": model_name, "params": json.dumps(params, sort_keys=True), "validation_pr_auc": average_precision_score(y_validation, probability), "validation_roc_auc": roc_auc_score(y_validation, probability)}
                hyper_rows.append(row); candidates.append((row["validation_pr_auc"], params))
            selected_params[config][model_name] = max(candidates, key=lambda item: item[0])[1]
    pd.DataFrame(hyper_rows).to_csv(report_dir / "hyperparameter_selection.csv", index=False)

    cv_rows = []; validation_rows = []; validation_probabilities = {}
    for config in FEATURE_CONFIGS:
        for model_name in MODEL_NAMES:
            for treatment in TREATMENTS:
                pipeline = make_pipeline(config, model_name, selected_params[config][model_name], treatment)
                scores = cross_validate(pipeline, train[FEATURE_CONFIGS[config]], y_train, cv=cv, scoring=scoring, n_jobs=1, error_score="raise")
                cv_row = {"feature_config": config, "model": model_name, "imbalance_treatment": treatment}
                for metric in scoring:
                    key = f"test_{metric}"; cv_row[f"cv_{metric}_mean"] = scores[key].mean(); cv_row[f"cv_{metric}_std"] = scores[key].std(ddof=1)
                cv_rows.append(cv_row)
                pipeline.fit(train[FEATURE_CONFIGS[config]], y_train)
                probability = pipeline.predict_proba(validation[FEATURE_CONFIGS[config]])[:, 1]
                validation_probabilities[(config, model_name, treatment)] = probability
                validation_rows.append({"feature_config": config, "model": model_name, "imbalance_treatment": treatment, **validation_score(y_validation, probability)})

    weight_grid = [(1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2), (2, 2, 1), (1, 2, 2), (2, 1, 2)]
    voting_weights = {}
    for config in FEATURE_CONFIGS:
        for treatment in TREATMENTS:
            component_probs = [validation_probabilities[(config, name, treatment)] for name in ["logistic_regression", "extra_trees", "catboost"]]
            weights, probability, _ = max(
                ((weights, np.average(component_probs, axis=0, weights=weights), average_precision_score(y_validation, np.average(component_probs, axis=0, weights=weights))) for weights in weight_grid),
                key=lambda item: item[2],
            )
            voting_weights[(config, treatment)] = weights
            pipeline = make_pipeline(config, "soft_voting", selected_params[config], treatment, weights)
            scores = cross_validate(pipeline, train[FEATURE_CONFIGS[config]], y_train, cv=cv, scoring=scoring, n_jobs=1, error_score="raise")
            cv_row = {"feature_config": config, "model": "soft_voting", "imbalance_treatment": treatment, "voting_weights": str(weights)}
            for metric in scoring:
                key = f"test_{metric}"; cv_row[f"cv_{metric}_mean"] = scores[key].mean(); cv_row[f"cv_{metric}_std"] = scores[key].std(ddof=1)
            cv_rows.append(cv_row)
            validation_rows.append({"feature_config": config, "model": "soft_voting", "imbalance_treatment": treatment, "voting_weights": str(weights), **validation_score(y_validation, probability)})

    cv_table = pd.DataFrame(cv_rows); validation_table = pd.DataFrame(validation_rows)
    cv_table.to_csv(report_dir / "cross_validation_results.csv", index=False)
    validation_table.to_csv(report_dir / "validation_selection.csv", index=False)
    selected = validation_table.sort_values(["f2", "pr_auc"], ascending=False).groupby(["feature_config", "model"], as_index=False).first()
    selected.to_csv(report_dir / "selected_candidates.csv", index=False)
    best = selected[selected.model != "dummy"].sort_values(["f2", "pr_auc"], ascending=False).iloc[0].to_dict()

    combined = pd.concat([train, validation])
    y_combined = combined.target
    final_models = {}; test_probabilities = {}; comparison_rows = []
    for _, choice in selected.iterrows():
        config, model_name, treatment = choice.feature_config, choice.model, choice.imbalance_treatment
        weights = voting_weights.get((config, treatment))
        params = selected_params[config] if model_name == "soft_voting" else selected_params[config][model_name]
        pipeline = make_pipeline(config, model_name, params, treatment, weights)
        pipeline.fit(combined[FEATURE_CONFIGS[config]], y_combined)
        probability = pipeline.predict_proba(test[FEATURE_CONFIGS[config]])[:, 1]
        key = (config, model_name); final_models[key] = pipeline; test_probabilities[key] = probability
        joblib.dump(pipeline, model_dir / f"{config}__{model_name}.joblib", compress=3)
        for threshold_name, threshold in [("0.50", 0.5), ("validation_f2", float(choice.threshold))]:
            row = {"feature_config": config, "model": model_name, "imbalance_treatment": treatment, "threshold_name": threshold_name, "threshold": threshold, **probability_metrics(test.target, probability, threshold)}
            row.update(bootstrap_intervals(test.target, probability, threshold))
            comparison_rows.append(row)

    for config in FEATURE_CONFIGS:
        pipeline = make_pipeline(config, "dummy", {}, "none")
        pipeline.fit(combined[FEATURE_CONFIGS[config]], y_combined)
        probability = pipeline.predict_proba(test[FEATURE_CONFIGS[config]])[:, 1]
        key = (config, "dummy"); final_models[key] = pipeline; test_probabilities[key] = probability
        joblib.dump(pipeline, model_dir / f"{config}__dummy.joblib", compress=3)
        row = {"feature_config": config, "model": "dummy", "imbalance_treatment": "none", "threshold_name": "0.50", "threshold": 0.5, **probability_metrics(test.target, probability, 0.5)}
        row.update(bootstrap_intervals(test.target, probability, 0.5)); comparison_rows.append(row)

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(report_dir / "model_comparison.csv", index=False)
    _save_model_figures(test, test_probabilities, selected, figure_dir)

    no_treatment = cv_table[cv_table.imbalance_treatment.eq("none")]
    smote = cv_table[cv_table.imbalance_treatment.str.startswith("smotenc")].sort_values("cv_f1_mean", ascending=False).groupby(["feature_config", "model"], as_index=False).first()
    smote_comparison = smote.merge(no_treatment, on=["feature_config", "model"], suffixes=("_smote", "_none"))
    for metric in ["recall", "f1", "average_precision"]:
        smote_comparison[f"delta_cv_{metric}"] = smote_comparison[f"cv_{metric}_mean_smote"] - smote_comparison[f"cv_{metric}_mean_none"]
    smote_comparison["improved_recall_and_f1"] = (smote_comparison.delta_cv_recall > 0) & (smote_comparison.delta_cv_f1 > 0)
    smote_comparison.to_csv(report_dir / "smotenc_comparison.csv", index=False)

    split_summary = pd.DataFrame([
        {"split": name, "rows": len(indices), "positive_n": int(frame.iloc[indices].target.sum()), "negative_n": int((1 - frame.iloc[indices].target).sum())}
        for name, indices in splits.items()
    ])
    split_summary.to_csv(report_dir / "split_summary.csv", index=False)
    selection = {"best_model_selected_on_validation": best, "test_accessed_only_after_selection": True}
    (report_dir / "model_selection.json").write_text(json.dumps(selection, indent=2, default=str) + "\n")
    return selection


def _save_model_figures(test: pd.DataFrame, probabilities: dict, selected: pd.DataFrame, output_dir: Path) -> None:
    colors = plt.cm.tab10(np.linspace(0, 1, 6))
    for config in FEATURE_CONFIGS:
        fig_roc, ax_roc = plt.subplots(figsize=(7, 6)); fig_pr, ax_pr = plt.subplots(figsize=(7, 6)); fig_cal, ax_cal = plt.subplots(figsize=(7, 6))
        choices = selected[selected.feature_config.eq(config)].set_index("model")
        for color, ((cfg, model_name), probability) in zip(colors, [(key, value) for key, value in probabilities.items() if key[0] == config]):
            fpr, tpr, _ = roc_curve(test.target, probability); precision, recall, _ = precision_recall_curve(test.target, probability)
            ax_roc.plot(fpr, tpr, label=f"{model_name} ({roc_auc_score(test.target, probability):.3f})", color=color)
            ax_pr.plot(recall, precision, label=f"{model_name} ({average_precision_score(test.target, probability):.3f})", color=color)
            fraction, mean = calibration_curve(test.target, probability, n_bins=10, strategy="quantile")
            ax_cal.plot(mean, fraction, marker="o", label=model_name, color=color)
            if model_name != "dummy":
                threshold = float(choices.loc[model_name, "threshold"])
                prediction = probability >= threshold
                matrix = confusion_matrix(test.target, prediction, labels=[0, 1])
                fig, ax = plt.subplots(figsize=(5, 4)); sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set(xlabel="Predicted", ylabel="Actual", title=f"{config}: {model_name}\nvalidation-F2 threshold={threshold:.3f}")
                fig.tight_layout(); fig.savefig(output_dir / f"confusion__{config}__{model_name}.png", dpi=180); plt.close(fig)
        ax_roc.plot([0, 1], [0, 1], "--", color="gray"); ax_roc.set(xlabel="False-positive rate", ylabel="True-positive rate", title=f"ROC curves: {config}"); ax_roc.legend(fontsize=8)
        ax_pr.axhline(test.target.mean(), ls="--", color="gray"); ax_pr.set(xlabel="Recall", ylabel="Precision", title=f"Precision-recall curves: {config}"); ax_pr.legend(fontsize=8)
        ax_cal.plot([0, 1], [0, 1], "--", color="gray"); ax_cal.set(xlabel="Mean predicted probability", ylabel="Observed fraction positive", title=f"Calibration: {config}"); ax_cal.legend(fontsize=8)
        for fig, name in [(fig_roc, "roc"), (fig_pr, "pr"), (fig_cal, "calibration")]:
            fig.tight_layout(); fig.savefig(output_dir / f"{name}__{config}.png", dpi=180); plt.close(fig)


def create_eda_notebook(project_dir: Path) -> None:
    notebook_dir = project_dir / "notebooks"; notebook_dir.mkdir(exist_ok=True)
    notebook = nbf.v4.new_notebook()
    notebook["cells"] = [
        nbf.v4.new_markdown_cell("# EDA: Young-adult profiles associated with previously reported premature CHD/MI\n\nThis cross-sectional analysis does **not** predict the exact occurrence of a future heart attack."),
        nbf.v4.new_code_cell("from pathlib import Path\nimport json, sys\nfrom IPython.display import Image, display\nPROJECT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\nsys.path.insert(0, str(PROJECT))\nfrom scripts.run_project import load_cohort, create_eda\nDATA = PROJECT / 'LLCP2023.csv'"),
        nbf.v4.new_code_cell("cohort = load_cohort(DATA)\nsummary = create_eda(cohort, PROJECT / 'figures' / 'eda')\nsummary"),
        nbf.v4.new_code_cell("for path in sorted((PROJECT / 'figures' / 'eda').glob('*.png')):\n    print(path.name)\n    display(Image(filename=str(path), width=1000))"),
        nbf.v4.new_markdown_cell("`TOLDHI3` is not ordinarily imputed. `cholesterol_status` combines it with `CHOLCHK3` using the official questionnaire skip logic, retaining never-checked, unknown/not-sure, and refused categories."),
    ]
    notebook["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nbf.write(notebook, notebook_dir / "01_eda.ipynb")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=PROJECT / "LLCP2023.csv")
    parser.add_argument("--stage", choices=["eda", "models", "all"], default="all")
    args = parser.parse_args()
    project_dir = PROJECT; report_dir = project_dir / "reports"
    report_dir.mkdir(exist_ok=True); write_column_discrepancy(args.data, report_dir); write_model_dictionary(report_dir)
    frame = load_cohort(args.data); splits = save_splits(frame, project_dir / "data")
    if args.stage in {"eda", "all"}:
        create_eda(frame, project_dir / "figures" / "eda"); create_eda_notebook(project_dir)
    if args.stage in {"models", "all"}:
        run_model_comparison(frame, splits, project_dir)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    main()
