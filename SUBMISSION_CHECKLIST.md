# Final submission quality-assurance checklist

QA date: 2026-08-15

| Check | Status | Evidence |
|---|---|---|
| Clean reproducibility run | **PASS** | Fresh Python 3.11 virtual environment; pinned requirements installed; audit, EDA, notebook, frozen post-model artifacts, PDF, and QA manifest regenerated without model fitting. 78 artifacts verified. |
| Notebook execution | **PASS** | `notebooks/01_eda.ipynb` executed with no error outputs in the clean environment. |
| Model loading | **PASS** | `models/extended_low_cost__logistic_regression.joblib` loaded under the pinned scikit-learn version; the Streamlit end-to-end test called `predict_proba`. |
| Streamlit test | **PASS** | AppTest submitted the default form, produced a `Research screening score`, displayed top factors, used fixed threshold `0.787903`, and showed non-diagnostic and India limitations without exceptions. |
| Metrics workbook integrity | **PASS** | `reports/model_results.xlsx` opened as a valid XLSX archive. Positive-class F1, macro F1, weighted F1, accuracy, and balanced accuracy are separately labelled in the final comparison source. |
| PDF validation | **PASS** | 11 A4 pages; all rendered pages visually inspected; embedded body fonts; no blank/clipped pages; Figures 1-16 present; four CDC hyperlinks present; no machine-specific home path in extracted text. |
| Frozen selection controls | **PASS** | Winner remains extended low-cost, class-weighted Logistic Regression; fixed validation-selected F2 threshold remains `0.7879034938787791`; test results were not used for reselection. |
| Missing files | **NONE** | `reports/reproducibility_status.json` records an empty `missing_files` list. |

Class weighting only marginally exceeded unweighted extended Logistic Regression on validation F2: `0.2233956133` versus `0.2224919094`, an absolute margin of `0.0009037039`.

The 461 MB raw CSV is excluded from `submission_ready/`. Follow `DATA_DOWNLOAD.md`; exact reproduction requires a file matching the recorded SHA-256 checksum.

## Exact reproduction command

Run from the project root after placing `LLCP2023.csv` there:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/reproduce_submission.py --data LLCP2023.csv
```

This QA command intentionally does not retrain models, modify the test set, refresh model selection, or choose a threshold.
