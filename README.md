# Explainable Prediction of Premature Heart Disease Among Young Adults Using Non-Invasive Health and Lifestyle Indicators

This project uses CDC BRFSS 2023 data to identify young adults whose non-invasive profiles are associated with **previously reported** premature coronary heart disease (CHD) or myocardial infarction (MI). It does not predict the exact occurrence of a future heart attack and is not a diagnostic tool.

## Reproduce

Python 3.11 is recommended.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/reproduce_submission.py --data LLCP2023.csv
```

The submission reproducer intentionally does not retrain models, alter the test set, refresh candidate selection, or select a threshold. It reruns the audit and EDA, executes the notebook, verifies the frozen model-selection artifacts, regenerates post-model figures from the saved winner, exports the report PDF, tests the saved pipeline and Streamlit app, and records QA status. Full model training remains available through `scripts/run_project.py --stage models`, but it is outside final QA. All randomness uses `random_state=42`.

Run the optional demonstration with:

```bash
.venv/bin/streamlit run app.py
```

## Cohort and features

The cohort is filtered to `_AGE80` 18-39. `_MICHD=1` is positive and `_MICHD=2` is negative; missing targets are excluded.

- Lifestyle-only: `_AGE80`, `_SEX`, `GENHLTH`, `PHYSHLTH`, `MENTHLTH`, `EXERANY2`, `_BMI5`, `_SMOKER3`, `DRNKANY6`, `_TOTINDA`, `_EDUCAG`, `_INCOMG1`.
- Extended low-cost: all lifestyle fields plus `BPHIGH6`, derived `cholesterol_status`, `DIABETE4`, `CHCKDNY2`, and `DIFFWALK`.

`cholesterol_status` applies the official questionnaire skip logic: `TOLDHI3` is asked only after a reported cholesterol check. The feature therefore retains `high`, `not_high`, `never_checked`, `unknown_not_sure`, and `refused` rather than ordinarily imputing structurally absent `TOLDHI3` values.

## Leakage controls and evaluation

Predictors are selected by a strict whitelist. `CVDINFR4`, `CVDCRHD4`, `_MICHD`, `HASYMP1`-`HASYMP6`, `_LLCPWT`, undocumented fields, and target-derived fields are excluded. `_LLCPWT` is used only for weighted EDA.

The single stratified 70/15/15 split is saved under `data/splits/`. The test IDs are never used for preprocessing, cross-validation, hyperparameter selection, imbalance selection, voting weights, or threshold selection. Pipelines fit numeric median imputation, categorical missing categories, ordinal encoding for SMOTENC, scaling, SMOTENC, and final one-hot encoding inside training folds.

Training compares no imbalance treatment, class weights, and SMOTENC ratios 0.10 and 0.25 using five-fold `StratifiedKFold`. Validation data select model parameters, treatment, soft-voting weights, and the F2 operating point. Final metrics explicitly separate positive-class F1, macro F1, and weighted F1, and include accuracy, balanced accuracy, precision, recall, specificity, ROC-AUC, average precision (reported as PR-AUC), Brier score, confusion matrices, calibration, and bootstrap 95% intervals.

Accuracy is retained only as a secondary completeness metric because the outcome is rare. Model selection and interpretation prioritize validation F2, precision, recall, positive-class F1, PR-AUC, balanced accuracy, and confusion-matrix trade-offs.

## Outputs

- `notebooks/01_eda.ipynb`: executed EDA notebook.
- `figures/eda/`, `figures/models/`, `figures/shap/`, and `figures/subgroups/`: saved analysis, evaluation, explainability, and subgroup figures.
- `reports/cross_validation_results.csv`: training-only five-fold results.
- `reports/validation_selection.csv`: validation treatment and threshold results.
- `reports/final_model_comparison.csv`: untouched-test results with separately labelled F1 variants, accuracy, and balanced accuracy.
- `reports/model_selection_audit.csv`: all candidate/treatment validation thresholds and metrics, ranked without test results.
- `reports/selected_model_operating_points.csv`: threshold 0.50 and the validation-selected F2 operating point.
- `reports/subgroup_metrics.csv`: fixed-threshold subgroup results with bootstrap intervals and stability flags.
- `reports/shap_grouped_importance.csv`: original-variable grouped SHAP importance.
- `reports/model_results.xlsx`: formatted comparison workbook.
- `reports/final_assignment_report.md`: assignment-ready report content.
- `reports/ethics_and_limitations.md`: responsible-use analysis.
- `reports/three_minute_demo_script.md`: presentation script.
- `models/`: fitted train-plus-validation pipelines.
- `reports/column_discrepancy.md`: explanation of the supplied 350-column CSV versus the current 345-variable CDC release.

The model selected solely on validation F2 was the class-weighted extended Logistic Regression, with an F2 operating point of 0.787903. On the untouched test set at that threshold it achieved recall 0.297, precision 0.071, positive-class F1 0.115, ROC-AUC 0.794, and PR-AUC 0.069. CatBoost had the highest observed test PR-AUC (0.094), but it was not relabelled as the selected best model after test inspection.

Class weighting only marginally exceeded unweighted extended Logistic Regression on validation F2 (0.223396 versus 0.222492; difference 0.000904).

Extended low-cost indicators improved ranking performance over lifestyle-only indicators. Class weighting was more effective than SMOTENC for the selected model. The heterogeneous soft-voting ensemble did not outperform the simpler validation-selected Logistic Regression model.

## Limitations

BRFSS is cross-sectional, self-reported, subject to recall and selection biases, and contains state/module availability differences. Associations cannot establish causality or future-event risk. Calibration is weak; demonstration output is therefore labelled a screening score rather than a medical probability. A below-threshold score never rules out disease. The model has not been validated for India and must not be used for clinical decisions there.
