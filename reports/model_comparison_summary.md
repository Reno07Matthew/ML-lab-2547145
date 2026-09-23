# Model-comparison checkpoint

## Cohort and split

Feature cleaning did not remove predictor-incomplete participants: all 99,114 young adults with a valid `_MICHD` target remained. Preprocessing handles missing predictors inside each fitted pipeline.

| Split | Rows | Positive | Negative |
|---|---:|---:|---:|
| Train | 69,379 | 735 | 68,644 |
| Validation | 14,867 | 157 | 14,710 |
| Test | 14,868 | 158 | 14,710 |

The same test IDs were used for every final evaluation. The test set was not accessed until the validation-based model, imbalance treatment, voting weights, and threshold selections were frozen.

## Validation-selected model

The best model selected on validation F2 was extended low-cost logistic regression with class weighting. Its validation-selected F2 operating point was 0.787903, with validation F2 0.2234, precision 0.0912, recall 0.3503, PR-AUC 0.0864, and ROC-AUC 0.7915.

This remains the declared best model. CatBoost subsequently produced a higher test PR-AUC, but the test set was not used to revise the selection.

## Selected-candidate five-fold cross-validation

Metrics below are mean training-fold results at threshold 0.50. Full means and standard deviations are in `cross_validation_results.csv`.

| Configuration | Model | Treatment | Recall | Positive-class F1 | ROC-AUC | PR-AUC |
|---|---|---|---:|---:|---:|---:|
| Extended | CatBoost | None | 0.0027 | 0.0053 | 0.7760 | 0.0763 |
| Extended | Extra Trees | None | 0.0000 | 0.0000 | 0.7646 | 0.0696 |
| Extended | Logistic Regression | Class weight | 0.6476 | 0.0604 | 0.7792 | 0.0836 |
| Extended | Random Forest | None | 0.0000 | 0.0000 | 0.7676 | 0.0781 |
| Extended | Soft Voting | None | 0.0014 | 0.0027 | 0.7817 | 0.0822 |
| Lifestyle | CatBoost | SMOTENC 0.10 | 0.0054 | 0.0101 | 0.7182 | 0.0419 |
| Lifestyle | Extra Trees | None | 0.0000 | 0.0000 | 0.7162 | 0.0396 |
| Lifestyle | Logistic Regression | Class weight | 0.6259 | 0.0513 | 0.7549 | 0.0539 |
| Lifestyle | Random Forest | None | 0.0000 | 0.0000 | 0.7261 | 0.0443 |
| Lifestyle | Soft Voting | None | 0.0000 | 0.0000 | 0.7511 | 0.0508 |

Untreated rare-event models mostly predicted no positives at 0.50. Their validation-selected thresholds were therefore used for F2 operating-point evaluation.

## Untouched-test results at validation-selected operating points

| Configuration | Model | Treatment | Threshold | Precision | Recall | Positive-class F1 | Specificity | ROC-AUC | PR-AUC |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Extended | CatBoost | None | 0.0363 | 0.0764 | 0.3038 | 0.1221 | 0.9606 | 0.7869 | 0.0935 |
| Extended | Extra Trees | None | 0.0405 | 0.0781 | 0.3101 | 0.1248 | 0.9607 | 0.7846 | 0.0704 |
| Extended | Logistic Regression | Class weight | 0.7879 | 0.0711 | 0.2975 | 0.1148 | 0.9583 | 0.7938 | 0.0692 |
| Extended | Random Forest | None | 0.0456 | 0.0902 | 0.2911 | 0.1377 | 0.9685 | 0.7914 | 0.0797 |
| Extended | Soft Voting | None | 0.0359 | 0.0786 | 0.3228 | 0.1264 | 0.9593 | 0.7925 | 0.0813 |
| Lifestyle | CatBoost | SMOTENC 0.10 | 0.0875 | 0.0445 | 0.2975 | 0.0774 | 0.9314 | 0.7332 | 0.0406 |
| Lifestyle | Extra Trees | None | 0.0281 | 0.0497 | 0.3418 | 0.0868 | 0.9298 | 0.7381 | 0.0398 |
| Lifestyle | Logistic Regression | Class weight | 0.8133 | 0.0759 | 0.2595 | 0.1175 | 0.9661 | 0.7719 | 0.0449 |
| Lifestyle | Random Forest | None | 0.0272 | 0.0486 | 0.3608 | 0.0856 | 0.9241 | 0.7440 | 0.0468 |
| Lifestyle | Soft Voting | None | 0.0214 | 0.0473 | 0.4430 | 0.0855 | 0.9041 | 0.7670 | 0.0477 |

Accuracy, positive-class F1, macro F1, weighted F1, balanced accuracy, Brier scores, confusion matrices, default-threshold results, calibration data, and bootstrap 95% confidence intervals are in `final_model_comparison.csv`.

## SMOTENC conclusion

SMOTENC did not improve overall minority-class discrimination. In cross-validation, a 0.25 ratio raised recall and positive-class F1 at the arbitrary 0.50 threshold for every non-dummy model, but reduced PR-AUC for every model/configuration comparison. Only lifestyle CatBoost selected SMOTENC 0.10 on validation; its test PR-AUC was 0.0406, below lifestyle soft voting (0.0477) and the no-SMOTENC extended models. The best validation-selected extended model used class weighting, not SMOTENC.

These models identify profiles associated with previously reported premature CHD or MI. They do not predict the exact occurrence of a future heart attack and are not diagnostic tools.
