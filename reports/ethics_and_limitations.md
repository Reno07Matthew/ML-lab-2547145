# Ethics and limitations

## Appropriate interpretation

The selected model identifies young adults whose non-invasive profiles are associated with **previously reported** premature coronary heart disease or myocardial infarction in BRFSS 2023. It does not predict the exact occurrence of a future heart attack, establish causality, provide a diagnosis, or estimate a clinically calibrated personal probability.

## Measurement and label limitations

- **Self-reported diagnosis:** `_MICHD` is derived from participants reporting that a health professional had told them they had coronary heart disease or myocardial infarction. Recall error, misunderstanding, and social-desirability effects can therefore affect both predictors and the outcome. CDC notes that BRFSS is self-reported and may be less accurate than physical measurement: https://www.cdc.gov/dhds/methods/index.html
- **Undiagnosed cases labelled negative:** A participant with unrecognised disease or without access to diagnostic care can answer “No” and enter the negative class. The model consequently learns “previously reported diagnosis,” not biological absence of disease.
- **Healthcare-access bias:** Diagnosis requires contact with healthcare services. Income, education, insurance, geography, and other access-related factors can influence whether disease is detected and reported. These variables may therefore encode access as well as health.
- **Survey coverage:** BRFSS is a telephone survey of non-institutionalised US adults. CDC documents possible noncoverage, nonresponse, and self-report limitations: https://www.cdc.gov/brfss/about/brfss_faq.htm

## Model and operating-point limitations

- **Rare-event uncertainty:** Only 158 positive cases were present in the test split. Precision, recall, PR-AUC, and subgroup estimates consequently have wide confidence intervals.
- **False negatives:** At the F2 operating point, 111 of 158 positive test cases were not flagged. A below-threshold result must never be presented as reassurance or as evidence that disease is absent.
- **False positives:** The F2 operating point generated 614 false positives. At threshold 0.50 this increased to 3,192, showing the resource and anxiety costs of broad screening.
- **Weak calibration:** The class-weighted Logistic Regression Brier score was 0.172. Its raw output must be treated as a model screening score, not a medical probability.
- **Threshold dependence:** The F2 threshold was selected on one validation split. Other populations, service capacities, or harm trade-offs may require prospective threshold evaluation rather than automatic reuse.
- **Subgroup instability:** Any subgroup with fewer than 30 positive test cases is explicitly marked unstable. All income categories and the two youngest age groups meet that condition, so apparent differences should not be over-interpreted.

## Study-design and transportability limitations

- **Cross-sectional design:** CDC describes BRFSS as a cross-sectional survey: https://www.cdc.gov/brfss/data_documentation/index.htm. Temporal order cannot be established; predictors such as general health or exercise may have changed after diagnosis. Associations cannot establish future risk or causality.
- **US-specific development:** The sample, questionnaire, healthcare access, diagnosis patterns, and social context are from the United States and territories represented in BRFSS 2023.
- **No validation for India:** The model has not been externally validated, recalibrated, or assessed for clinical utility in India. It should not be deployed for Indian screening, triage, diagnosis, or treatment decisions.

## Responsible-use requirements

Any future evaluation should use prospective or temporally separated data, clinical outcome verification, explicit calibration assessment, decision-curve or resource-impact analysis, and independent external validation. Human review, privacy protection, clear communication of uncertainty, and monitoring for subgroup harms are essential. Model outputs must never be used to deny care, insurance, employment, or other services.
