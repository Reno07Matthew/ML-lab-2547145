# Three-minute demonstration script

## 0:00–0:30 — Research question

“This project asks whether low-cost, non-invasive BRFSS indicators can identify young adults whose profiles are associated with previously reported premature coronary heart disease or myocardial infarction. It does not predict an exact future heart attack and it is not diagnostic.”

## 0:30–1:00 — Data and leakage controls

“I filtered BRFSS 2023 to ages 18–39 using `_AGE80`, leaving 99,114 valid records and 1,050 positives. The split was stratified 70/15/15. The test set remained untouched until all validation choices were frozen. Target components, symptoms, survey weight, and undocumented columns were excluded. Cholesterol status follows the official `CHOLCHK3`/`TOLDHI3` skip logic rather than ordinary imputation.”

## 1:00–1:40 — Selection and operating points

“Five-fold cross-validation compared Logistic Regression, Random Forest, CatBoost, Extra Trees, and a soft-voting ensemble with no treatment, class weights, and SMOTENC. The declared winner is extended class-weighted Logistic Regression because it ranked first on validation F2—not because of test performance.

At threshold 0.50, recall is 65.8%, but there are 3,192 false positives. At the validation-selected F2 operating point, 0.787903, false positives fall to 614 and precision/F1 improve, but recall falls to 29.7%. That is the operational trade-off.”

## 1:40–2:15 — Explainability and subgroups

“Grouped SHAP importance highlights blood-pressure history, general health, smoking, education, sex, income, age, and BMI. SHAP explains the fitted association; it does not show causality. Local waterfalls use median-score representatives for TP, TN, FP, and FN, with participant IDs removed.

Every subgroup uses the same threshold. Groups with fewer than 30 positive test cases are marked unstable, which includes both youngest age groups and every income category.”

## 2:15–2:45 — Streamlit demonstration

“The demonstration loads the saved pipeline and returns a screening score, never a medical probability. It shows the top model-attribution factors and always warns that a low score does not rule out disease and a high score does not establish disease.”

## 2:45–3:00 — Conclusion

“Extended low-cost indicators improved ranking over lifestyle-only indicators. Class weighting was more effective than SMOTENC. The heterogeneous ensemble did not outperform the simpler validation-selected Logistic Regression model. External validation, calibration, and prospective clinical evaluation are required, especially before any use in India.”
