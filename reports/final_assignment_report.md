# Explainable Prediction of Premature Heart Disease Among Young Adults Using Non-Invasive Health and Lifestyle Indicators

Reno Reji Matthew  
Roll No. 2547145  
Class: 4 MCA A

## Aim and claim

This project uses CDC BRFSS 2023 data to identify young adults whose profiles are associated with previously reported premature coronary heart disease or myocardial infarction. It does not predict the exact occurrence of a future heart attack and is not a diagnostic system.

## Data and audit

Participants were restricted to ages 18–39 using `_AGE80`. `_MICHD=1` was the positive class and `_MICHD=2` the negative class. The analysed cohort contained 99,114 participants, including 1,050 positives. Unweighted prevalence was 1.059%; survey-weighted prevalence was 1.169%.

The supplied CSV contained 350 columns whereas the current CDC release documents 345 variables. Six CSV-only variables—`BIRTHSEX`, `CELSXBRT`, `LNDSXBRT`, `RCSGEND1`, `RCSXBRTH`, and `TRNSGNDR`—were excluded. The current layout contains `RCSBORG1`, which is absent from the supplied CSV. Modelling used a strict documented-feature whitelist.

## Features and preprocessing

Lifestyle-only predictors were age, sex, general health, physical- and mental-health days, exercise, BMI, smoking, alcohol, calculated activity, education, and income. The extended configuration added blood-pressure history, derived cholesterol status, diabetes, kidney disease, and difficulty walking.

`TOLDHI3` was treated as structural missingness using `CHOLCHK3` questionnaire skip logic. The derived cholesterol feature retained high, not high, never checked, unknown/not sure, and refused categories. All imputers, scalers, encoders, and SMOTENC steps were fitted only within training data or training folds.

Forbidden target-derived fields, `CVDINFR4`, `CVDCRHD4`, `_MICHD`, `HASYMP1`–`HASYMP6`, undocumented columns, and `_LLCPWT` were excluded from predictors. `_LLCPWT` was used only for weighted EDA.

## Split and selection

One stratified 70/15/15 split used `random_state=42`: 69,379 training records with 735 positives, 14,867 validation records with 157 positives, and 14,868 untouched test records with 158 positives. Participant IDs for all splits were saved separately.

Five-fold stratified cross-validation compared Logistic Regression, Random Forest, CatBoost, Extra Trees, and soft voting under no treatment, class weights, and SMOTENC ratios 0.10 and 0.25. Hyperparameters, imbalance treatment, voting weights, and thresholds were selected using training and validation data only.

The class-weighted extended Logistic Regression ranked first on validation F2: threshold 0.787903, precision 0.0912, recall 0.3503, specificity 0.9627, F2 0.2234, and PR-AUC 0.0864. It remains the declared winner. CatBoost’s later test PR-AUC was not used to revise the selection.

Class weighting only marginally exceeded the unweighted extended Logistic Regression on validation F2 (0.223396 versus 0.222492; absolute difference 0.000904). This small validation margin should not be overstated.

## Test operating points

At threshold 0.50, the selected model produced TN=11,518, FP=3,192, FN=54, TP=104; accuracy 0.7817, balanced accuracy 0.7206, precision 0.0316, recall 0.6582, specificity 0.7830, and positive-class F1 0.0602. This setting found more positive cases but flagged many negative cases.

At the validation-selected **F2 operating point** of 0.787903, it produced TN=14,096, FP=614, FN=111, TP=47; accuracy 0.9512, balanced accuracy 0.6279, precision 0.0711, recall 0.2975, specificity 0.9583, and positive-class F1 0.1148. It reduced false positives and improved precision/F1 but missed more reported cases than threshold 0.50. This is not described as a high-sensitivity threshold.

Because the positive class represents only about 1.1% of the cohort, accuracy is reported as a secondary completeness metric, not the principal measure of success. Model selection used validation F2; interpretation emphasizes precision, recall, positive-class F1, PR-AUC, balanced accuracy, and the confusion matrices.

## Explainability

Linear SHAP values were computed using the fitted preprocessing output and Logistic Regression model. One-hot SHAP contributions were recombined at the original-variable level. The leading grouped features were blood-pressure history, general health, smoking, education, sex, income, age, BMI, alcohol, difficulty walking, and cholesterol status. These are predictive attributions, not causal effects.

Local waterfall plots use the case nearest the median screening score within each TP, TN, FP, and FN outcome type. This avoids participant identifiers and avoids selecting unusually extreme examples.

## Subgroups

All subgroup metrics use the same global 0.787903 threshold. Age-group recall ranged from 0.196 for ages 30–34 to 0.352 for ages 35–39. Recall was 0.341 for males and 0.239 for females. These are descriptive differences, not proof of equal or unequal treatment effects.

The 18–24 and 25–29 groups had fewer than 30 positive cases. Every income subgroup also had fewer than 30 positives. Their metrics and bootstrap intervals are marked unstable. Education groups except “did not graduate high school” and missing/unknown had at least 30 positives.

## Ethics and limitations

The outcome is self-reported and depends on prior diagnosis. Undiagnosed disease can therefore be labelled negative, and healthcare access can affect both diagnosis and learned associations. The event is rare, calibration is weak, and both false negatives and false positives have material consequences. BRFSS is cross-sectional, so temporal order, causality, and future-event risk cannot be established. The model has not been validated for India and must not be deployed there for clinical decisions.

## Conclusion

Extended low-cost health-history indicators improved ranking performance over lifestyle-only indicators. Class weighting was more effective than SMOTENC for the validation-selected model. The heterogeneous soft-voting ensemble did not outperform the simpler validation-selected Logistic Regression model. The resulting model is an explainable research screening model for association with previously reported premature CHD/MI—not a diagnostic tool or a predictor of the exact occurrence of a future heart attack.

## References

1. CDC. 2023 BRFSS Survey Data and Documentation. https://www.cdc.gov/brfss/annual_data/annual_2023.html
2. CDC. 2023 BRFSS Variable Layout. https://www.cdc.gov/brfss/annual_data/2023/llcp_varlayout_23_onecolumn.html
3. CDC. BRFSS Data Documentation. https://www.cdc.gov/brfss/data_documentation/index.htm
4. CDC. BRFSS Methods and Limitations. https://www.cdc.gov/dhds/methods/index.html
5. CDC. 2023 BRFSS Codebook, local archived copy: `USCODE23_LLCP_021924.HTML`.
