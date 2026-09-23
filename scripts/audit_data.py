#!/usr/bin/env python3
"""Audit the BRFSS 2023 young-adult cohort; intentionally does no modelling."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd


TARGET = "_MICHD"
AGE = "_AGE80"
SURVEY_WEIGHT = "_LLCPWT"

CONTEXT_FIELDS = [
    "_AGE80",
    "SEXVAR",
]

LIFESTYLE_FEATURES = [
    "EXERANY2",
    "_SMOKER3",
    "DRNKANY6",
    "_RFBING6",
    "_RFDRHV8",
]

HEALTH_HISTORY_FEATURES = [
    "_BMI5",
    "GENHLTH",
    "PHYSHLTH",
    "MENTHLTH",
    "BPHIGH6",
    "TOLDHI3",
    "CVDSTRK3",
    "CHCKDNY2",
    "DIABETE4",
]

LEAKAGE_FIELDS = ["CVDINFR4", "CVDCRHD4"]
KNOWLEDGE_NOT_SYMPTOMS = [f"HASYMP{i}" for i in range(1, 7)]
PROJECT = Path(__file__).resolve().parents[1]
MODEL_FEATURES = LIFESTYLE_FEATURES + HEALTH_HISTORY_FEATURES
SELECTED_FEATURES = CONTEXT_FIELDS + MODEL_FEATURES

FEATURE_INFO = {
    "_AGE80": ("demographic", "Imputed age, collapsed only above 80", "18-39"),
    "SEXVAR": ("demographic", "Sex of respondent", "1,2"),
    "_BMI5": ("lifestyle/non-invasive", "Computed BMI; two implied decimal places", "1-9999"),
    "EXERANY2": ("lifestyle", "Any exercise in past 30 days", "1,2"),
    "_SMOKER3": ("lifestyle", "Computed four-level smoking status", "1,2,3,4"),
    "DRNKANY6": ("lifestyle", "Any alcoholic drink in past 30 days", "1,2"),
    "_RFBING6": ("lifestyle", "Computed binge-drinking indicator", "1,2"),
    "_RFDRHV8": ("lifestyle", "Computed heavy-drinking indicator", "1,2"),
    "GENHLTH": ("low-cost health status/history", "Self-rated general health", "1,2,3,4,5"),
    "PHYSHLTH": ("low-cost health status/history", "Physically unhealthy days in past 30 days", "1-30,88"),
    "MENTHLTH": ("low-cost health status/history", "Mentally unhealthy days in past 30 days", "1-30,88"),
    "BPHIGH6": ("low-cost health history", "Ever told blood pressure was high", "1,2,3,4"),
    "TOLDHI3": ("low-cost health history", "Ever told cholesterol was high", "1,2"),
    "CVDSTRK3": ("low-cost health history", "Ever diagnosed with stroke", "1,2"),
    "CHCKDNY2": ("low-cost health history", "Ever told of kidney disease", "1,2"),
    "DIABETE4": ("low-cost health history", "Ever told of diabetes", "1,2,3,4"),
}

VALID_LABELS = {
    "_AGE80": "18-39=exact imputed age within study cohort",
    "SEXVAR": "1=Male; 2=Female",
    "_BMI5": "1-9999=BMI with two implied decimal places",
    "EXERANY2": "1=Yes; 2=No",
    "_SMOKER3": "1=Every day; 2=Some days; 3=Former; 4=Never",
    "DRNKANY6": "1=Yes; 2=No",
    "_RFBING6": "1=No; 2=Yes",
    "_RFDRHV8": "1=No; 2=Yes",
    "GENHLTH": "1=Excellent; 2=Very good; 3=Good; 4=Fair; 5=Poor",
    "PHYSHLTH": "1-30=Number of days; 88=None",
    "MENTHLTH": "1-30=Number of days; 88=None",
    "BPHIGH6": "1=Yes; 2=Pregnancy only; 3=No; 4=Borderline/pre-hypertensive/elevated",
    "TOLDHI3": "1=Yes; 2=No",
    "CVDSTRK3": "1=Yes; 2=No",
    "CHCKDNY2": "1=Yes; 2=No",
    "DIABETE4": "1=Yes; 2=Pregnancy only; 3=No; 4=Pre-diabetes/borderline",
}

TRANSFORMS = {
    "_AGE80": "Retain as numeric for filtering/subgroups only",
    "SEXVAR": "Retain as categorical for subgroup evaluation only",
    "_BMI5": "Divide by 100 after splitting",
    "PHYSHLTH": "Recode 88 to 0 after splitting",
    "MENTHLTH": "Recode 88 to 0 after splitting",
}

# Official 2023 codebook labels. These responses become missing later inside
# preprocessing; they are not silently treated as clinical/lifestyle categories.
INVALID_LABELS = {
    "EXERANY2": {7: "Don't know/Not sure", 9: "Refused"},
    "_SMOKER3": {9: "Don't know/Refused/Missing"},
    "DRNKANY6": {7: "Don't know/Not sure", 9: "Refused/Missing"},
    "_RFBING6": {9: "Don't know/Refused/Missing"},
    "_RFDRHV8": {9: "Don't know/Refused/Missing"},
    "GENHLTH": {7: "Don't know/Not sure", 9: "Refused"},
    "PHYSHLTH": {77: "Don't know/Not sure", 99: "Refused"},
    "MENTHLTH": {77: "Don't know/Not sure", 99: "Refused"},
    "BPHIGH6": {7: "Don't know/Not sure", 9: "Refused"},
    "TOLDHI3": {7: "Don't know/Not sure", 9: "Refused"},
    "CVDSTRK3": {7: "Don't know/Not sure", 9: "Refused"},
    "CHCKDNY2": {7: "Don't know/Not sure", 9: "Refused"},
    "DIABETE4": {7: "Don't know/Not sure", 9: "Refused"},
}


def valid_mask(series: pd.Series, feature: str) -> pd.Series:
    if feature == "_AGE80":
        return series.between(18, 39)
    if feature == "_BMI5":
        return series.between(1, 9999)
    if feature in {"PHYSHLTH", "MENTHLTH"}:
        return series.between(1, 30) | series.eq(88)
    allowed = {
        "SEXVAR": [1, 2],
        "EXERANY2": [1, 2],
        "_SMOKER3": [1, 2, 3, 4],
        "DRNKANY6": [1, 2],
        "_RFBING6": [1, 2],
        "_RFDRHV8": [1, 2],
        "GENHLTH": [1, 2, 3, 4, 5],
        "BPHIGH6": [1, 2, 3, 4],
        "TOLDHI3": [1, 2],
        "CVDSTRK3": [1, 2],
        "CHCKDNY2": [1, 2],
        "DIABETE4": [1, 2, 3, 4],
    }
    return series.isin(allowed[feature])


def load_data(path: Path, columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    if path.suffix.lower() == ".xpt":
        frame = pd.read_sas(path, format="xport", encoding="latin1")
        available = list(frame.columns)
        return frame[[c for c in columns if c in frame]], available
    available = list(pd.read_csv(path, nrows=0).columns)
    return pd.read_csv(path, usecols=[c for c in columns if c in available], low_memory=False), available


def verify_codebook(path: Path) -> None:
    text = html.unescape(path.read_text(encoding="cp1252")).replace("\xa0", " ")
    missing = [name for name in [TARGET, SURVEY_WEIGHT, *SELECTED_FEATURES, *LEAKAGE_FIELDS, *KNOWLEDGE_NOT_SYMPTOMS] if f"SAS Variable Name: {name}" not in text]
    if missing:
        raise ValueError(f"Codebook definitions not found: {missing}")


def audit(data_path: Path, codebook_path: Path, output_dir: Path) -> dict:
    verify_codebook(codebook_path)
    assert not set(MODEL_FEATURES) & set(LEAKAGE_FIELDS + KNOWLEDGE_NOT_SYMPTOMS)
    required = ["_STATE", "SEQNO", AGE, TARGET, SURVEY_WEIGHT, *SELECTED_FEATURES, *LEAKAGE_FIELDS, *KNOWLEDGE_NOT_SYMPTOMS]
    data, available_columns = load_data(data_path, list(dict.fromkeys(required)))
    absent = [c for c in [AGE, TARGET] if c not in available_columns]
    if absent:
        raise ValueError(f"Required columns missing from dataset: {absent}")

    total_rows = len(data)
    full_target = data[TARGET].value_counts(dropna=False)
    young = data[data[AGE].between(18, 39)].copy()
    cohort = young[young[TARGET].isin([1, 2])].copy()
    positive = int(cohort[TARGET].eq(1).sum())
    negative = int(cohort[TARGET].eq(2).sum())
    assert positive + negative == len(cohort)
    weighted_positive_pct = None
    if SURVEY_WEIGHT in cohort and cohort[SURVEY_WEIGHT].notna().all():
        weighted_positive_pct = round(100 * cohort.loc[cohort[TARGET].eq(1), SURVEY_WEIGHT].sum() / cohort[SURVEY_WEIGHT].sum(), 6)
    weighted_text = f"{weighted_positive_pct:.4f}%" if weighted_positive_pct is not None else "unavailable"
    derived_target = pd.Series(float("nan"), index=data.index)
    derived_target[data["CVDINFR4"].eq(1) | data["CVDCRHD4"].eq(1)] = 1
    derived_target[data["CVDINFR4"].eq(2) & data["CVDCRHD4"].eq(2)] = 2
    target_derivation_mismatches = int((derived_target.fillna(-1) != data[TARGET].fillna(-1)).sum())
    missing_target_sources = {}
    for (heart_attack, chd), count in young[young[TARGET].isna()].groupby(LEAKAGE_FIELDS, dropna=False).size().items():
        fmt = lambda value: "blank" if pd.isna(value) else str(int(value))
        missing_target_sources[f"CVDINFR4={fmt(heart_attack)}, CVDCRHD4={fmt(chd)}"] = int(count)

    id_columns = [c for c in ["_STATE", "SEQNO"] if c in data]
    duplicate_ids = int(data.duplicated(id_columns).sum()) if id_columns else None

    audit_rows = []
    for feature in SELECTED_FEATURES:
        group, description, valid_codes = FEATURE_INFO[feature]
        configurations = (
            "filter/subgroup only"
            if feature in CONTEXT_FIELDS
            else "both"
            if feature in LIFESTYLE_FEATURES
            else "lifestyle_plus_health_history"
        )
        if feature not in cohort:
            audit_rows.append({
                "feature": feature,
                "feature_group": group,
                "configurations": configurations,
                "description": description,
                "official_valid_codes": valid_codes,
                "available_in_dataset": False,
            })
            continue
        series = cohort[feature]
        usable = valid_mask(series, feature)
        raw_missing = series.isna()
        invalid = ~raw_missing & ~usable
        breakdown = []
        for value, count in series[invalid].value_counts().sort_index().items():
            key = int(value) if float(value).is_integer() else float(value)
            label = INVALID_LABELS.get(feature, {}).get(key, "Out of official valid range")
            breakdown.append(f"{key}={label} (n={count})")
        n = len(cohort)
        audit_rows.append({
            "feature": feature,
            "feature_group": group,
            "configurations": configurations,
            "description": description,
            "official_valid_codes": valid_codes,
            "available_in_dataset": True,
            "usable_n": int(usable.sum()),
            "usable_pct": round(100 * usable.mean(), 4),
            "raw_missing_n": int(raw_missing.sum()),
            "invalid_response_n": int(invalid.sum()),
            "effective_missing_n": int((~usable).sum()),
            "effective_missing_pct": round(100 * (~usable).mean(), 4),
            "invalid_codes_observed": "; ".join(breakdown),
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_audit = pd.DataFrame(audit_rows)
    feature_audit.to_csv(output_dir / "feature_audit.csv", index=False)

    dictionary = feature_audit[["feature", "feature_group", "configurations", "description", "official_valid_codes"]].copy()
    dictionary["official_value_labels"] = dictionary["feature"].map(VALID_LABELS)
    dictionary["planned_transform"] = dictionary["feature"].map(lambda f: TRANSFORMS.get(f, "Treat as categorical after splitting"))
    dictionary["special_response_handling"] = dictionary["feature"].map(
        lambda f: "; ".join(f"{k}={v}" for k, v in INVALID_LABELS.get(f, {}).items()) or "Blank only"
    )
    dictionary = pd.concat([
        pd.DataFrame([{
            "feature": TARGET,
            "feature_group": "target",
            "configurations": "outcome",
            "description": "Ever reported myocardial infarction or coronary heart disease",
            "official_valid_codes": "1,2",
            "official_value_labels": "1=Positive/reported MI or CHD; 2=Negative/did not report MI or CHD",
            "planned_transform": "Map 1 to 1 and 2 to 0; exclude blank target before splitting",
            "special_response_handling": "Blank when either source item is unknown, refused, or missing",
        }]),
        dictionary,
    ], ignore_index=True)
    dictionary.to_csv(output_dir / "data_dictionary.csv", index=False)

    expected_counts = {1: 36311, 2: 392427, "missing": 4585}
    observed_counts = {
        1: int(full_target.get(1, 0)),
        2: int(full_target.get(2, 0)),
        "missing": int(data[TARGET].isna().sum()),
    }
    summary = {
        "project_phase": "initial_data_audit_only_no_modelling",
        "dataset_file": data_path.name,
        "dataset_format": data_path.suffix.lower().lstrip("."),
        "codebook_file": codebook_path.name,
        "total_rows": total_rows,
        "total_columns_in_dataset": len(available_columns),
        "young_age_18_39_rows": len(young),
        "young_rows_with_valid_target": len(cohort),
        "young_rows_excluded_for_missing_target": int(young[TARGET].isna().sum()),
        "young_missing_target_source_code_combinations": missing_target_sources,
        "positive_n": positive,
        "negative_n": negative,
        "positive_pct": round(100 * positive / len(cohort), 6),
        "negative_pct": round(100 * negative / len(cohort), 6),
        "survey_weighted_positive_pct": weighted_positive_pct,
        "positive_to_negative_ratio": round(positive / negative, 6),
        "duplicate_composite_state_seqno": duplicate_ids,
        "selected_columns_available": {c: c in available_columns for c in SELECTED_FEATURES},
        "explicitly_excluded_predictors": LEAKAGE_FIELDS + KNOWLEDGE_NOT_SYMPTOMS,
        "target_derivation_mismatches": target_derivation_mismatches,
        "full_target_counts_match_official_codebook": observed_counts == expected_counts,
        "full_target_counts_observed": observed_counts,
        "full_target_counts_official": expected_counts,
    }
    (output_dir / "data_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    missing_table = feature_audit[["feature", "usable_n", "usable_pct", "raw_missing_n", "invalid_response_n", "effective_missing_n", "effective_missing_pct"]]
    report = f"""# BRFSS 2023 initial data audit

Status: audit complete; modelling intentionally not started.

## Cohort and target

- Supplied dataset: `{data_path.name}` ({total_rows:,} rows, {len(available_columns)} columns).
- `_AGE80` filter 18-39: {len(young):,} rows.
- Valid `_MICHD` cohort: {len(cohort):,} rows; {int(young[TARGET].isna().sum()):,} young-adult rows have a missing target and are excluded.
- Positive (`_MICHD=1`): {positive:,} ({100 * positive / len(cohort):.4f}%).
- Negative (`_MICHD=2`): {negative:,} ({100 * negative / len(cohort):.4f}%).
- Survey-weighted positive percentage using `_LLCPWT`: {weighted_text} (descriptive only).
- Positive:negative ratio: 1:{negative / positive:.2f}.
- Duplicate respondent IDs using `_STATE` + `SEQNO`: {duplicate_ids:,}.
- Full-file `_MICHD` frequencies {'match' if observed_counts == expected_counts else 'DO NOT MATCH'} the official codebook.
- `_MICHD` values inconsistent with the documented `CVDINFR4`/`CVDCRHD4` derivation: {target_derivation_mismatches:,}.
- Missing young-adult targets arise from source responses: {'; '.join(f'{k} (n={v})' for k, v in missing_target_sources.items())}.

## Selected-feature missingness after age and target filtering

```csv
{missing_table.to_csv(index=False).strip()}
```

`effective_missing_n` combines blank values with official refused, unknown/not-sure, and derived missing codes. Exact observed codes and labels are in `feature_audit.csv`.

## Planned feature configurations

- Lifestyle-only: `EXERANY2`, `_SMOKER3`, `DRNKANY6`, `_RFBING6`, `_RFDRHV8`.
- Lifestyle plus low-cost non-invasive health/history: all lifestyle fields plus `_BMI5`, `GENHLTH`, `PHYSHLTH`, `MENTHLTH`, `BPHIGH6`, `TOLDHI3`, `CVDSTRK3`, `CHCKDNY2`, `DIABETE4`.
- `_AGE80` and `SEXVAR` are retained for cohort definition and subgroup evaluation, not included in the literal lifestyle-only predictor list.

## Leakage and interpretation controls

- `CVDINFR4` and `CVDCRHD4` are excluded because the official codebook states that `_MICHD` is derived from them.
- `HASYMP1`-`HASYMP6` are excluded because they measure knowledge of possible symptoms, not experienced symptoms.
- `_MICHD` is prevalent self-reported history of CHD/MI in a cross-sectional survey. It is not a future-event label and cannot establish future heart-attack risk or causality.
- No train/validation/test split, preprocessing fit, resampling, or model training has been run in this phase.
"""
    (output_dir / "data_audit.md").write_text(report)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=PROJECT / ("LLCP2023.XPT" if (PROJECT / "LLCP2023.XPT").exists() else "LLCP2023.csv"))
    parser.add_argument("--codebook", type=Path, default=PROJECT / "USCODE23_LLCP_021924.HTML")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "reports" / "audit")
    args = parser.parse_args()
    if not args.data.exists() or not args.codebook.exists():
        raise FileNotFoundError("Dataset and official codebook must both exist")
    summary = audit(args.data, args.codebook, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
