#!/usr/bin/env python3
"""Reproduce post-model audit, explainability, subgroup, and demo artifacts."""

from __future__ import annotations

import ast
import argparse
import json
import os
import sys
import warnings
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
MPL_CACHE = PROJECT / ".cache" / "matplotlib"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from scripts.run_project import (
    FEATURE_CONFIGS,
    HYPERPARAMETERS,
    NUMERIC,
    RANDOM_STATE,
    load_cohort,
    make_pipeline,
    save_splits,
)


REPORTS = PROJECT / "reports"
FIGURES = PROJECT / "figures"
MODEL_PATH = PROJECT / "models" / "extended_low_cost__logistic_regression.joblib"
WINNER = ("extended_low_cost", "logistic_regression", "class_weight")

LABELS = {
    "_SEX": {"1": "Male", "2": "Female"},
    "GENHLTH": {"1": "Excellent", "2": "Very good", "3": "Good", "4": "Fair", "5": "Poor"},
    "EXERANY2": {"1": "Yes", "2": "No"},
    "_SMOKER3": {"1": "Every day", "2": "Some days", "3": "Former", "4": "Never"},
    "DRNKANY6": {"1": "Yes", "2": "No"},
    "_TOTINDA": {"1": "Active", "2": "Inactive"},
    "_EDUCAG": {"1": "Did not graduate high school", "2": "High-school graduate", "3": "Some college/technical school", "4": "College/technical graduate"},
    "_INCOMG1": {"1": "< $15k", "2": "$15k-<$25k", "3": "$25k-<$35k", "4": "$35k-<$50k", "5": "$50k-<$100k", "6": "$100k-<$200k", "7": "$200k+"},
    "BPHIGH6": {"1": "High blood pressure", "2": "Pregnancy only", "3": "No", "4": "Borderline/elevated"},
    "cholesterol_status": {"high": "High", "not_high": "Not high", "never_checked": "Never checked", "unknown_not_sure": "Unknown/not sure", "refused": "Refused"},
    "DIABETE4": {"1": "Diabetes", "2": "Pregnancy only", "3": "No", "4": "Pre-diabetes/borderline"},
    "CHCKDNY2": {"1": "Kidney disease", "2": "No"},
    "DIFFWALK": {"1": "Difficulty walking", "2": "No"},
}


def metric_row(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    prediction = probability >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    recall = recall_score(y_true, prediction, zero_division=0)
    specificity = tn / (tn + fp)
    return {
        "accuracy": accuracy_score(y_true, prediction),
        "precision": precision_score(y_true, prediction, zero_division=0),
        "recall": recall,
        "specificity": specificity,
        "false_positive_rate": fp / (tn + fp),
        "positive_class_f1": f1_score(y_true, prediction, zero_division=0),
        "macro_f1": f1_score(y_true, prediction, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, prediction, average="weighted", zero_division=0),
        "balanced_accuracy": (recall + specificity) / 2,
        "pr_auc": average_precision_score(y_true, probability) if len(np.unique(y_true)) == 2 else np.nan,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def relabel_final_metrics() -> pd.DataFrame:
    table = pd.read_csv(REPORTS / "model_comparison.csv")
    table = table.rename(columns={
        "f1": "positive_class_f1", "f1_macro": "macro_f1",
        "f1_ci_low": "positive_class_f1_ci_low", "f1_ci_high": "positive_class_f1_ci_high",
    })
    negative_f1 = 2 * table.tn / (2 * table.tn + table.fp + table.fn)
    table["weighted_f1"] = (
        (table.tn + table.fp) * negative_f1
        + (table.tp + table.fn) * table.positive_class_f1
    ) / (table.tn + table.fp + table.fn + table.tp)
    assert {"positive_class_f1", "macro_f1", "weighted_f1", "accuracy", "balanced_accuracy"} <= set(table)
    columns = [
        "feature_config", "model", "imbalance_treatment", "threshold_name", "threshold",
        "accuracy", "balanced_accuracy", "precision", "recall", "specificity",
        "positive_class_f1", "macro_f1", "weighted_f1", "roc_auc", "pr_auc", "brier_score",
        "tn", "fp", "fn", "tp", "recall_ci_low", "recall_ci_high",
        "positive_class_f1_ci_low", "positive_class_f1_ci_high", "pr_auc_ci_low", "pr_auc_ci_high",
    ]
    table = table[columns]
    table.to_csv(REPORTS / "model_comparison.csv", index=False)
    table.to_csv(REPORTS / "final_model_comparison.csv", index=False)
    return table


def relabel_cv_metrics() -> None:
    for filename in ["cross_validation_results.csv", "smotenc_comparison.csv"]:
        table = pd.read_csv(REPORTS / filename)
        renamed = {}
        for column in table.columns:
            new = column.replace("f1_macro", "macro_f1")
            if "f1" in new and "positive_class_f1" not in new and "macro_f1" not in new:
                new = new.replace("f1", "positive_class_f1")
            renamed[column] = new
        table.rename(columns=renamed).to_csv(REPORTS / filename, index=False)


def selected_hyperparameters() -> dict:
    hyper = pd.read_csv(REPORTS / "hyperparameter_selection.csv")
    best = hyper.sort_values("validation_pr_auc", ascending=False).groupby(["feature_config", "model"], as_index=False).first()
    return {(row.feature_config, row.model): json.loads(row.params) for row in best.itertuples()}


def create_selection_audit(frame: pd.DataFrame, splits: dict[str, np.ndarray]) -> pd.DataFrame:
    train = frame.iloc[splits["train"]]
    validation = frame.iloc[splits["validation"]]
    params = selected_hyperparameters()
    original = pd.read_csv(REPORTS / "validation_selection.csv")
    rows = []
    for candidate in original.itertuples():
        config, model, treatment = candidate.feature_config, candidate.model, candidate.imbalance_treatment
        model_params = {name: params[(config, name)] for name in HYPERPARAMETERS} if model == "soft_voting" else params[(config, model)]
        weights = ast.literal_eval(candidate.voting_weights) if model == "soft_voting" and pd.notna(candidate.voting_weights) else None
        pipeline = make_pipeline(config, model, model_params, treatment, weights)
        pipeline.fit(train[FEATURE_CONFIGS[config]], train.target)
        probability = pipeline.predict_proba(validation[FEATURE_CONFIGS[config]])[:, 1]
        metrics = metric_row(validation.target.to_numpy(), probability, float(candidate.threshold))
        f2 = 5 * metrics["precision"] * metrics["recall"] / (4 * metrics["precision"] + metrics["recall"]) if metrics["precision"] + metrics["recall"] else 0.0
        assert np.isclose(metrics["precision"], candidate.precision) and np.isclose(metrics["recall"], candidate.recall)
        rows.append({
            "feature_config": config, "model": model, "imbalance_treatment": treatment,
            "validation_threshold": candidate.threshold, "validation_precision": metrics["precision"],
            "validation_recall": metrics["recall"], "validation_specificity": metrics["specificity"],
            "validation_f2": f2, "validation_pr_auc": metrics["pr_auc"],
            "voting_weights": candidate.voting_weights if pd.notna(candidate.voting_weights) else "",
            "selected_winner": (config, model, treatment) == WINNER,
        })
    audit = pd.DataFrame(rows).sort_values(["validation_f2", "validation_pr_auc"], ascending=False).reset_index(drop=True)
    audit.insert(0, "validation_rank", np.arange(1, len(audit) + 1))
    winner = audit[audit.selected_winner]
    assert len(winner) == 1 and winner.iloc[0].validation_rank == 1
    audit.to_csv(REPORTS / "model_selection_audit.csv", index=False)
    return audit


def create_operating_points(test: pd.DataFrame, pipeline, threshold: float) -> pd.DataFrame:
    output = FIGURES / "models" / "selected_model_operating_points"
    output.mkdir(parents=True, exist_ok=True)
    probability = pipeline.predict_proba(test[FEATURE_CONFIGS["extended_low_cost"]])[:, 1]
    definitions = [
        ("Default threshold", 0.5, "Flags more profiles and finds more reported cases, but produces many more false positives and very low precision."),
        ("F2 operating point", threshold, "Reduces false positives and improves precision/F1, while missing more reported cases than the default threshold."),
    ]
    rows = []
    for name, value, tradeoff in definitions:
        metrics = metric_row(test.target.to_numpy(), probability, value)
        rows.append({"operating_point": name, "threshold": value, **metrics, "operational_tradeoff": tradeoff})
        matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
        fig, ax = plt.subplots(figsize=(5.4, 4.3))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set(xlabel="Predicted class", ylabel="Reported outcome", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
        ax.set_title(f"Selected Logistic Regression\n{name} ({value:.6f})")
        fig.tight_layout(); fig.savefig(output / f"confusion__{name.lower().replace(' ', '_')}__{value:.6f}.png", dpi=180); plt.close(fig)
    table = pd.DataFrame(rows)
    table.to_csv(REPORTS / "selected_model_operating_points.csv", index=False)
    return table


def transformed_matrix_and_names(pipeline, frame: pd.DataFrame) -> tuple[np.ndarray, list[str], list[str]]:
    before = pipeline.named_steps["before_sampling"]
    after = pipeline.named_steps["after_sampling"]
    intermediate = before.transform(frame)
    matrix = after.transform(intermediate)
    matrix = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
    features = FEATURE_CONFIGS["extended_low_cost"]
    numeric = [name for name in NUMERIC if name in features]
    categorical = [name for name in features if name not in numeric]
    ordinal = before.named_transformers_["categorical"].named_steps["ordinal"]
    onehot = after.named_transformers_["categorical"]
    names, originals = list(numeric), list(numeric)
    for feature, raw_categories, encoded_categories in zip(categorical, ordinal.categories_, onehot.categories_):
        for code in encoded_categories:
            index = int(round(float(code)))
            raw = str(raw_categories[index]) if 0 <= index < len(raw_categories) else "unknown"
            names.append(f"{feature}={LABELS.get(feature, {}).get(raw, raw)}")
            originals.append(feature)
    assert matrix.shape[1] == len(names) == len(originals)
    return matrix, names, originals


def create_shap_outputs(combined: pd.DataFrame, test: pd.DataFrame, pipeline, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = FIGURES / "shap"; output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_STATE)
    background_rows = rng.choice(len(combined), size=min(3000, len(combined)), replace=False)
    display_rows = rng.choice(len(test), size=min(2500, len(test)), replace=False)
    background, names, originals = transformed_matrix_and_names(pipeline, combined.iloc[background_rows][FEATURE_CONFIGS["extended_low_cost"]])
    display, _, _ = transformed_matrix_and_names(pipeline, test.iloc[display_rows][FEATURE_CONFIGS["extended_low_cost"]])
    explainer = shap.LinearExplainer(pipeline.named_steps["model"], background)
    explanation = explainer(display)
    explanation.feature_names = names

    shap.plots.beeswarm(explanation, max_display=20, show=False)
    plt.gcf().set_size_inches(10, 7); plt.title("Global SHAP distribution (log-odds contribution)"); plt.tight_layout()
    plt.savefig(output / "01_global_beeswarm.png", dpi=180, bbox_inches="tight"); plt.close()

    shap.plots.bar(explanation.abs.mean(0), max_display=20, show=False)
    plt.gcf().set_size_inches(9, 7); plt.title("Mean absolute SHAP by encoded feature"); plt.tight_layout()
    plt.savefig(output / "02_mean_absolute_shap_encoded.png", dpi=180, bbox_inches="tight"); plt.close()

    encoded = pd.DataFrame({
        "preprocessed_feature": names, "original_feature": originals,
        "mean_absolute_shap": np.abs(explanation.values).mean(axis=0),
    }).sort_values("mean_absolute_shap", ascending=False)
    grouped = pd.DataFrame([
        {
            "original_feature": feature,
            "mean_absolute_shap": np.abs(explanation.values[:, np.array(originals) == feature].sum(axis=1)).mean(),
        }
        for feature in dict.fromkeys(originals)
    ]).sort_values("mean_absolute_shap", ascending=False)
    encoded.to_csv(REPORTS / "shap_encoded_importance.csv", index=False)
    grouped.to_csv(REPORTS / "shap_grouped_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.barplot(data=grouped, y="original_feature", x="mean_absolute_shap", color="#0F766E", ax=ax)
    ax.set(xlabel="Mean |grouped SHAP| (log-odds)", ylabel="Original feature", title="Grouped SHAP importance for the selected model")
    fig.tight_layout(); fig.savefig(output / "03_grouped_shap_importance.png", dpi=180); plt.close(fig)

    probability = pipeline.predict_proba(test[FEATURE_CONFIGS["extended_low_cost"]])[:, 1]
    prediction = probability >= threshold
    outcomes = {
        "TP": (test.target.to_numpy() == 1) & prediction,
        "TN": (test.target.to_numpy() == 0) & ~prediction,
        "FP": (test.target.to_numpy() == 0) & prediction,
        "FN": (test.target.to_numpy() == 1) & ~prediction,
    }
    local_rows = []
    for label, mask in outcomes.items():
        candidates = np.flatnonzero(mask)
        median = np.median(probability[candidates])
        index = candidates[np.argmin(np.abs(probability[candidates] - median))]
        local_matrix, _, _ = transformed_matrix_and_names(pipeline, test.iloc[[index]][FEATURE_CONFIGS["extended_low_cost"]])
        local = explainer(local_matrix)[0]
        local.feature_names = names
        shap.plots.waterfall(local, max_display=12, show=False)
        plt.gcf().set_size_inches(9, 6); plt.title(f"Representative {label} (closest to within-group median score)"); plt.tight_layout()
        plt.savefig(output / f"04_local_waterfall_{label.lower()}.png", dpi=180, bbox_inches="tight"); plt.close()
        local_rows.append({"case_type": label, "reported_outcome": int(test.iloc[index].target), "predicted_class": int(prediction[index]), "screening_score": probability[index], "selection_rule": "closest to median score within outcome type"})
    pd.DataFrame(local_rows).to_csv(REPORTS / "shap_local_case_summary.csv", index=False)
    assert "participant_id" not in encoded.columns and "participant_id" not in grouped.columns

    background_record = {
        "expected_log_odds": float(explainer.expected_value),
        "background_mean": background.mean(axis=0).tolist(),
        "preprocessed_feature_names": names,
        "original_features": originals,
        "f2_threshold": threshold,
    }
    (REPORTS / "shap_background.json").write_text(json.dumps(background_record, indent=2) + "\n")
    return encoded, grouped


def subgroup_bootstrap(y: np.ndarray, probability: np.ndarray, threshold: float, iterations: int = 500) -> dict:
    rng = np.random.default_rng(RANDOM_STATE); collected = {name: [] for name in ["recall", "specificity", "positive_class_f1", "pr_auc"]}
    for _ in range(iterations):
        index = rng.integers(0, len(y), len(y)); sy, sp = y[index], probability[index]
        if len(np.unique(sy)) < 2: continue
        metrics = metric_row(sy, sp, threshold)
        for name in collected: collected[name].append(metrics[name])
    result = {}
    for name, values in collected.items():
        result[f"{name}_ci_low"], result[f"{name}_ci_high"] = np.quantile(values, [0.025, 0.975]) if values else (np.nan, np.nan)
    return result


def create_subgroup_outputs(test: pd.DataFrame, pipeline, threshold: float) -> pd.DataFrame:
    output = FIGURES / "subgroups"; output.mkdir(parents=True, exist_ok=True)
    data = test.copy()
    data["age_group"] = pd.cut(data["_AGE80"], [17, 24, 29, 34, 39], labels=["18-24", "25-29", "30-34", "35-39"])
    orders = {
        "sex_group": ["Male", "Female", "Missing/unknown"],
        "income_group": ["< $15k", "$15k-<$25k", "$25k-<$35k", "$35k-<$50k", "$50k-<$100k", "$100k-<$200k", "$200k+", "Missing/unknown"],
        "education_group": ["Did not graduate high school", "High-school graduate", "Some college/technical school", "College/technical graduate", "Missing/unknown"],
    }
    data["sex_group"] = pd.Categorical(data["_SEX"].map(LABELS["_SEX"]).fillna("Missing/unknown"), orders["sex_group"], ordered=True)
    data["income_group"] = pd.Categorical(data["_INCOMG1"].map(LABELS["_INCOMG1"]).fillna("Missing/unknown"), orders["income_group"], ordered=True)
    data["education_group"] = pd.Categorical(data["_EDUCAG"].map(LABELS["_EDUCAG"]).fillna("Missing/unknown"), orders["education_group"], ordered=True)
    probability = pipeline.predict_proba(data[FEATURE_CONFIGS["extended_low_cost"]])[:, 1]
    data["screening_score"] = probability
    rows = []
    dimensions = {"Age": "age_group", "Sex": "sex_group", "Income": "income_group", "Education": "education_group"}
    for dimension, column in dimensions.items():
        for group, subset in data.groupby(column, observed=True, sort=True):
            y = subset.target.to_numpy(); p = subset.screening_score.to_numpy(); metrics = metric_row(y, p, threshold)
            positives = int(y.sum())
            rows.append({
                "dimension": dimension, "group": str(group), "sample_size": len(subset), "positive_cases": positives,
                "prevalence": y.mean(), "stability_flag": "Unstable: <30 positives" if positives < 30 else "Stable: >=30 positives",
                "global_threshold": threshold, **metrics, **subgroup_bootstrap(y, p, threshold),
            })
    table = pd.DataFrame(rows)
    table.to_csv(REPORTS / "subgroup_metrics.csv", index=False)

    for metric, title in [("recall", "Recall"), ("pr_auc", "PR-AUC")]:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        for ax, dimension in zip(axes.flat, dimensions):
            subset = table[table.dimension.eq(dimension)]
            colors = ["#D97706" if value.startswith("Unstable") else "#0F766E" for value in subset.stability_flag]
            ax.bar(subset.group, subset[metric], color=colors)
            ax.set_title(dimension); ax.set_ylabel(title); ax.tick_params(axis="x", rotation=30)
            ax.set_ylim(0, max(1.0, float(subset[metric].max()) * 1.15))
        fig.suptitle(f"Selected Logistic Regression: subgroup {title} at the same F2 threshold ({threshold:.6f})\nOrange indicates fewer than 30 positive test cases")
        fig.tight_layout(); fig.savefig(output / f"{metric}_by_subgroup.png", dpi=180); plt.close(fig)
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=PROJECT / "LLCP2023.csv")
    parser.add_argument("--refresh-audit", action="store_true", help="Explicitly refit validation candidates; never used by submission QA")
    args = parser.parse_args()
    frame = load_cohort(args.data)
    splits = save_splits(frame, PROJECT / "data")
    test = frame.iloc[splits["test"]].copy()
    combined = frame.iloc[np.concatenate([splits["train"], splits["validation"]])].copy()
    pipeline = joblib.load(MODEL_PATH)
    selection = json.loads((REPORTS / "model_selection.json").read_text())
    winner = selection["best_model_selected_on_validation"]
    assert (winner["feature_config"], winner["model"], winner["imbalance_treatment"]) == WINNER
    threshold = float(winner["threshold"])
    relabel_final_metrics()
    relabel_cv_metrics()
    audit_path = REPORTS / "model_selection_audit.csv"
    if args.refresh_audit:
        create_selection_audit(frame, splits)
    elif not audit_path.exists():
        raise FileNotFoundError("Frozen model_selection_audit.csv is required; QA will not refit candidates")
    create_operating_points(test, pipeline, threshold)
    create_shap_outputs(combined, test, pipeline, threshold)
    create_subgroup_outputs(test, pipeline, threshold)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
