#!/usr/bin/env python3
"""Export the assignment report PDF from frozen tables and figures."""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
import reportlab
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)


PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "output" / "pdf" / "final_assignment_report.pdf"
INK, RULE, PALE = colors.black, colors.HexColor("#999999"), colors.HexColor("#F2F2F2")
REPORTLAB_FONTS = Path(reportlab.__file__).resolve().parent / "fonts"
pdfmetrics.registerFont(TTFont("Vera", REPORTLAB_FONTS / "Vera.ttf"))
pdfmetrics.registerFont(TTFont("Vera-Bold", REPORTLAB_FONTS / "VeraBd.ttf"))
pdfmetrics.registerFontFamily("Vera", normal="Vera", bold="Vera-Bold", italic="Vera", boldItalic="Vera-Bold")


def styles():
    sheet = getSampleStyleSheet()
    sheet.add(ParagraphStyle(name="Title2", parent=sheet["Title"], fontName="Vera-Bold", fontSize=22, leading=27, textColor=INK, alignment=TA_CENTER, spaceAfter=14))
    sheet.add(ParagraphStyle(name="Subtitle", parent=sheet["Normal"], fontName="Vera", fontSize=11, leading=16, textColor=INK, alignment=TA_CENTER, spaceAfter=18))
    sheet.add(ParagraphStyle(name="H1x", parent=sheet["Heading1"], fontName="Vera-Bold", fontSize=15, leading=19, textColor=INK, spaceBefore=8, spaceAfter=8))
    sheet.add(ParagraphStyle(name="H2x", parent=sheet["Heading2"], fontName="Vera-Bold", fontSize=11.5, leading=15, textColor=INK, spaceBefore=7, spaceAfter=5))
    sheet.add(ParagraphStyle(name="Bodyx", parent=sheet["BodyText"], fontName="Vera", fontSize=9.2, leading=13, textColor=INK, spaceAfter=7))
    sheet.add(ParagraphStyle(name="Smallx", parent=sheet["BodyText"], fontName="Vera", fontSize=7.5, leading=10, textColor=INK, spaceAfter=4))
    sheet.add(ParagraphStyle(name="Captionx", parent=sheet["BodyText"], fontName="Vera", fontSize=7.5, leading=10, textColor=INK, alignment=TA_CENTER, spaceAfter=7))
    sheet.add(ParagraphStyle(name="Callout", parent=sheet["BodyText"], fontName="Vera", fontSize=9.3, leading=13, textColor=INK, backColor=PALE, borderColor=INK, borderWidth=0.6, borderPadding=8, spaceAfter=10))
    return sheet


S = styles()


def p(text: str, style: str = "Bodyx") -> Paragraph:
    return Paragraph(text, S[style])


def table(rows, widths=None, header=True, font_size=7.2):
    formatted = [[Paragraph(escape(str(value)), S["Smallx"]) for value in row] for row in rows]
    item = Table(formatted, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.3, RULE),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
    ]
    if header:
        commands += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DDDDDD")), ("TEXTCOLOR", (0, 0), (-1, 0), INK)]
    for row in range(1 if header else 0, len(rows)):
        if row % 2 == 0:
            commands.append(("BACKGROUND", (0, row), (-1, row), colors.HexColor("#F5F5F5")))
    item.setStyle(TableStyle(commands))
    return item


def image(path: str, width: float, caption: str):
    source = PROJECT / path
    if not source.exists():
        raise FileNotFoundError(source)
    item = Image(str(source))
    item.drawHeight = width * item.imageHeight / item.imageWidth
    item.drawWidth = width
    return [item, p(escape(caption), "Captionx")]


def image_grid(paths_and_captions, width=3.25 * inch):
    cells = []
    for path, caption in paths_and_captions:
        source = PROJECT / path
        item = Image(str(source))
        item.drawHeight = width * item.imageHeight / item.imageWidth
        item.drawWidth = width
        cells.append([item, p(escape(caption), "Captionx")])
    rows = []
    for index in range(0, len(cells), 2):
        pair = cells[index:index + 2]
        rows.append([cell[0] for cell in pair] + ([""] if len(pair) == 1 else []))
        rows.append([cell[1] for cell in pair] + ([""] if len(pair) == 1 else []))
    item = Table(rows, colWidths=[width, width], hAlign="CENTER")
    item.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]))
    return item


def page(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(RULE)
    canvas.line(18 * mm, 16 * mm, 192 * mm, 16 * mm)
    canvas.setFont("Vera", 7.5)
    canvas.setFillColor(INK)
    canvas.drawString(18 * mm, 10.5 * mm, "BRFSS 2023 explainable premature CHD/MI profile screening")
    canvas.drawRightString(192 * mm, 10.5 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    split = pd.read_csv(PROJECT / "reports" / "split_summary.csv")
    audit = pd.read_csv(PROJECT / "reports" / "model_selection_audit.csv").head(8)
    operating = pd.read_csv(PROJECT / "reports" / "selected_model_operating_points.csv")
    shap = pd.read_csv(PROJECT / "reports" / "shap_grouped_importance.csv").head(10)
    subgroup = pd.read_csv(PROJECT / "reports" / "subgroup_metrics.csv")

    story = [Spacer(1, 18 * mm), p("Explainable Prediction of Premature Heart Disease Among Young Adults Using Non-Invasive Health and Lifestyle Indicators", "Title2"),
        p("CDC BRFSS 2023 | Final assignment report | Reproducible machine-learning analysis", "Subtitle"),
        p("<b>Reno Reji Matthew</b><br/>Roll No. 2547145<br/>Class: 4 MCA A", "Subtitle"),
        p("Scope", "H1x"),
        p("This project identifies young adults whose non-invasive profiles are associated with previously reported premature coronary heart disease or myocardial infarction (CHD/MI). It does not predict the exact occurrence of a future heart attack and is not a diagnostic system.", "Callout"),
        p("Declared model", "H2x"),
        p("The selected model remains the extended low-cost, class-weighted Logistic Regression chosen solely on validation F2. The fixed F2 operating-point threshold is 0.787903. Neither the winner nor threshold was revised after examining the test set."),
        p("Key conclusion", "H2x"),
        p("Extended low-cost indicators improved ranking performance over lifestyle-only indicators; class weighting was more effective than SMOTENC; and the heterogeneous soft-voting ensemble did not outperform the simpler validation-selected Logistic Regression."),
        Spacer(1, 8 * mm), p("Research demonstration only - not for diagnosis, individual clinical decisions, or deployment in India.", "Callout"), PageBreak()]

    story += [p("1. Data, audit, and feature configurations", "H1x"),
        p("The cohort was filtered to ages 18-39 with _AGE80. The target was _MICHD, coded 1 as positive and 2 as negative. After excluding missing targets, 99,114 respondents remained: 1,050 positive and 98,064 negative. Unweighted prevalence was 1.059%; survey-weighted prevalence was 1.169%."),
        p("The supplied CSV contained 350 columns, whereas the current February 2025 CDC transport release documents 345 variables. Six CSV-only fields (BIRTHSEX, CELSXBRT, LNDSXBRT, RCSGEND1, RCSXBRTH, TRNSGNDR) were excluded; the current-layout RCSBORG1 field was absent from the supplied CSV. Modelling used an explicit documented-field whitelist."),
        p("Features", "H2x"),
        p("Lifestyle-only: age, sex, general health, physical- and mental-health days, exercise, BMI, smoking, alcohol, calculated activity, education, and income. Extended low-cost: all lifestyle features plus blood-pressure history, derived cholesterol status, diabetes, kidney disease, and difficulty walking."),
        p("TOLDHI3 was handled as structural missingness. CHOLCHK3 skip logic retained high, not high, never checked, unknown/not sure, and refused cholesterol-status categories rather than applying ordinary median or mode imputation."),
        p("Leakage controls", "H2x"),
        p("CVDINFR4, CVDCRHD4, _MICHD, HASYMP1-HASYMP6, target-derived fields, undocumented columns, and _LLCPWT were excluded from predictors. _LLCPWT was used only for weighted descriptive EDA."),
        p("Fixed stratified split", "H2x"),
        table([["Split", "Rows", "Positive", "Negative"]] + [[r.split, f"{r.rows:,}", f"{r.positive_n:,}", f"{r.negative_n:,}"] for r in split.itertuples()], [30 * mm, 32 * mm, 32 * mm, 32 * mm]),
        Spacer(1, 5 * mm), *image("figures/eda/01_class_prevalence.png", 6.5 * inch, "Figure 1. Weighted and unweighted prevalence and class imbalance."), PageBreak()]

    story += [p("2. Exploratory data analysis", "H1x"),
        p("Missingness and official invalid-response codes were decoded before modelling. Preprocessing remained inside training-only pipeline steps. The figures below describe associations in the cross-sectional cohort and do not imply causality."),
        image_grid([
            ("figures/eda/02_missingness.png", "Figure 2. Effective feature missingness."),
            ("figures/eda/03_lifestyle_relationships.png", "Figure 3. Target relationships for demographic and lifestyle indicators."),
            ("figures/eda/04_health_history_relationships.png", "Figure 4. Target relationships for low-cost health-history indicators."),
            ("figures/models/calibration__extended_low_cost.png", "Figure 5. Extended-model calibration curves; weak calibration limits probability interpretation."),
        ], 3.15 * inch), PageBreak()]

    story += [p("3. Leakage-safe modelling and validation-only selection", "H1x"),
        p("A single stratified 70/15/15 split used random_state=42. Five-fold StratifiedKFold on training data compared DummyClassifier, Logistic Regression, Random Forest, CatBoost, Extra Trees, and soft voting. Imbalance treatments were none, class weights, and SMOTENC ratios 0.10 and 0.25. Imputation, ordinal encoding, SMOTENC, scaling, and one-hot encoding were fitted only within training folds. Validation data selected hyperparameters, imbalance treatment, voting weights, and threshold."),
        p("Validation model-selection audit", "H2x"),
        table([["Rank", "Configuration", "Model", "Treatment", "Threshold", "Precision", "Recall", "Specificity", "F2", "PR-AUC"]] + [[
            int(r.validation_rank), r.feature_config.replace("_", " "), r.model.replace("_", " "), r.imbalance_treatment,
            f"{r.validation_threshold:.6f}", f"{r.validation_precision:.4f}", f"{r.validation_recall:.4f}", f"{r.validation_specificity:.4f}", f"{r.validation_f2:.4f}", f"{r.validation_pr_auc:.4f}"
        ] for r in audit.itertuples()], [10*mm, 24*mm, 23*mm, 19*mm, 18*mm, 16*mm, 15*mm, 17*mm, 13*mm, 15*mm], font_size=6.2),
        Spacer(1, 5 * mm), p("Class weighting only marginally exceeded unweighted extended Logistic Regression on validation F2: 0.223396 versus 0.222492, an absolute margin of 0.000904. The small difference should not be overstated. The winner was not replaced by CatBoost based on later test PR-AUC."), PageBreak(),
        p("Test-set ranking comparison", "H1x"),
        p("This test-set curve is reported after the validation-selected winner and threshold were frozen. It was not used to change either decision."),
        *image("figures/models/pr__extended_low_cost.png", 6.2 * inch, "Figure 6. Test-set precision-recall curves for final extended models; shown only after validation selection was frozen."), PageBreak()]

    story += [p("4. Untouched-test evaluation at two operating points", "H1x"),
        p("All final models were evaluated on the identical untouched test set. Accuracy is included for completeness but is not the principal success metric because the positive class is rare. Selection used validation F2; interpretation emphasizes precision, recall, positive-class F1, PR-AUC, balanced accuracy, and confusion-matrix trade-offs."),
        table([["Operating point", "Threshold", "Precision", "Recall", "Specificity", "Positive F1", "Macro F1", "Weighted F1", "Balanced acc.", "TN / FP / FN / TP"]] + [[
            r.operating_point, f"{r.threshold:.6f}", f"{r.precision:.4f}", f"{r.recall:.4f}", f"{r.specificity:.4f}", f"{r.positive_class_f1:.4f}", f"{r.macro_f1:.4f}", f"{r.weighted_f1:.4f}", f"{r.balanced_accuracy:.4f}", f"{r.tn} / {r.fp} / {r.fn} / {r.tp}"
        ] for r in operating.itertuples()], [25*mm, 18*mm, 15*mm, 14*mm, 16*mm, 16*mm, 15*mm, 17*mm, 18*mm, 26*mm], font_size=6.4),
        Spacer(1, 5 * mm), image_grid([
            ("figures/models/selected_model_operating_points/confusion__default_threshold__0.500000.png", "Figure 7. Default threshold 0.50: more positives found, with many false positives."),
            ("figures/models/selected_model_operating_points/confusion__f2_operating_point__0.787903.png", "Figure 8. Validation-selected F2 threshold 0.787903: fewer false positives, but more false negatives."),
        ], 3.05 * inch),
        p("Operational trade-off", "H2x"),
        p("At 0.50 the model found 104 of 158 reported positives but flagged 3,192 negatives. At the F2 operating point it found 47 positives and flagged 614 negatives. The second operating point is not a high-sensitivity threshold."), PageBreak()]

    story += [p("5. Explainability", "H1x"),
        p("Linear SHAP values were computed from the selected fitted Logistic Regression pipeline using feature names produced by preprocessing. One-hot contributions were regrouped to their original variables for the primary global ranking. Values are log-odds contributions relative to the analysis background, not causal effects."),
        image_grid([
            ("figures/shap/01_global_beeswarm.png", "Figure 9. Global SHAP beeswarm."),
            ("figures/shap/03_grouped_shap_importance.png", "Figure 10. Mean absolute SHAP grouped by original variable."),
        ], 3.15 * inch),
        p("Leading grouped variables", "H2x"),
        table([["Original feature", "Mean absolute SHAP"]] + [[r.original_feature, f"{r.mean_absolute_shap:.4f}"] for r in shap.itertuples()], [60 * mm, 45 * mm]), PageBreak()]

    story += [p("6. Representative local explanations", "H1x"),
        p("Local cases were chosen as the observation closest to the median screening score within each TP, TN, FP, and FN outcome type. Participant identifiers are absent, and unusually extreme cases were not cherry-picked."),
        image_grid([
            ("figures/shap/04_local_waterfall_tp.png", "Figure 11. Representative true positive."),
            ("figures/shap/04_local_waterfall_tn.png", "Figure 12. Representative true negative."),
            ("figures/shap/04_local_waterfall_fp.png", "Figure 13. Representative false positive."),
            ("figures/shap/04_local_waterfall_fn.png", "Figure 14. Representative false negative."),
        ], 3.1 * inch), PageBreak()]

    stable = subgroup[subgroup.stability_flag.str.startswith("Stable")]
    unstable_n = int((~subgroup.index.isin(stable.index)).sum())
    story += [p("7. Fixed-threshold subgroup evaluation", "H1x"),
        p("Age, sex, income, and education groups were evaluated at the same global threshold of 0.787903. No subgroup-specific thresholds were optimized. Each row reports sample size, positive cases, prevalence, precision, recall, specificity, false-positive rate, positive-class F1, and PR-AUC with bootstrap intervals where feasible."),
        p(f"{unstable_n} subgroup rows contain fewer than 30 positive test cases and are explicitly marked unstable. These estimates have substantial rare-event uncertainty and should not be used to claim subgroup equivalence or inequity."),
        image_grid([
            ("figures/subgroups/recall_by_subgroup.png", "Figure 15. Recall by subgroup; orange indicates fewer than 30 positives."),
            ("figures/subgroups/pr_auc_by_subgroup.png", "Figure 16. PR-AUC by subgroup; orange indicates fewer than 30 positives."),
        ], 3.15 * inch), PageBreak()]

    story += [p("8. Ethics, limitations, and responsible use", "H1x"),
        p("Self-reported diagnosis and undiagnosed disease", "H2x"),
        p("The target depends on a respondent having received and recalled a diagnosis. Undiagnosed cases may be labelled negative. Healthcare access can influence diagnosis, exposure histories, and the associations learned by the model."),
        p("Rare-event uncertainty and errors", "H2x"),
        p("Only 158 positives occurred in the test set. False negatives may delay appropriate assessment; false positives may cause anxiety and unnecessary follow-up. Subgroup uncertainty is especially large where positive counts are below 30."),
        p("Weak calibration", "H2x"),
        p("The selected model is weakly calibrated. Its raw output must be displayed only as a research screening score, not as a medical probability. A below-threshold score never reassures a user that disease is absent; an above-threshold score never establishes disease."),
        p("Cross-sectional design", "H2x"),
        p("BRFSS measures self-reported histories and current profiles at one survey occasion. It cannot establish temporal order, causality, or the exact occurrence of a future heart attack."),
        p("Geographic validity", "H2x"),
        p("The data are from the United States. The model has not been externally validated for India and must not be used for clinical decisions there."),
        p("Demonstration safeguard", "H2x"),
        p("The optional Streamlit app loads the frozen pipeline, uses the fixed 0.787903 F2 threshold, labels output as a screening score, lists the top attribution factors, and displays visible non-diagnostic and geographic warnings."), PageBreak()]

    story += [p("9. Reproducibility and conclusion", "H1x"),
        p("All dependencies are version-pinned in requirements.txt. Code resolves resources with project-relative pathlib paths. The final QA command reruns the audit and EDA, executes the notebook, regenerates post-model explanations and subgroup outputs from the frozen saved pipeline, exports this PDF, loads the model, and tests the Streamlit interface. It intentionally does not retrain, alter the test set, refresh candidate selection, or choose a threshold."),
        p("Exact verification command", "H2x"),
        p("python3.11 -m venv .venv<br/>.venv/bin/python -m pip install -r requirements.txt<br/>.venv/bin/python scripts/reproduce_submission.py --data LLCP2023.csv", "Callout"),
        p("Final conclusion", "H2x"),
        p("Extended low-cost health-history indicators improved ranking performance over lifestyle-only indicators. Class weighting was more effective than SMOTENC, although it exceeded unweighted extended Logistic Regression by only 0.000904 validation-F2 points. The heterogeneous ensemble did not outperform the simpler validation-selected Logistic Regression. The result is an explainable research screening model for association with previously reported premature CHD/MI, not a diagnostic tool or a future-event risk predictor."),
        p("References", "H1x"),
        p('1. CDC. <link href="https://www.cdc.gov/brfss/annual_data/annual_2023.html" color="#000000">2023 BRFSS Survey Data and Documentation</link>.'),
        p('2. CDC. <link href="https://www.cdc.gov/brfss/annual_data/2023/llcp_varlayout_23_onecolumn.html" color="#000000">2023 BRFSS Variable Layout</link>.'),
        p('3. CDC. <link href="https://www.cdc.gov/brfss/data_documentation/index.htm" color="#000000">BRFSS Data Documentation</link>.'),
        p('4. CDC. <link href="https://www.cdc.gov/dhds/methods/index.html" color="#000000">BRFSS Methods and Limitations</link>.'),
        p("5. CDC. 2023 BRFSS Codebook, archived locally as USCODE23_LLCP_021924.HTML."),
        p("Artifacts", "H2x"),
        p("The submission includes the executed EDA notebook, audit and data dictionary, split IDs, final metrics workbook and CSV tables, figures, fitted pipelines, reproducibility scripts, ethics report, demonstration script, Streamlit app, raw-data instructions, and this PDF. The 461 MB raw CSV is excluded and identified by SHA-256 in DATA_DOWNLOAD.md.")]

    document = SimpleDocTemplate(str(OUTPUT), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm, topMargin=17 * mm, bottomMargin=21 * mm, title="Explainable Prediction of Premature Heart Disease Among Young Adults", author="Reno Reji Matthew")
    document.build(story, onFirstPage=page, onLaterPages=page)


if __name__ == "__main__":
    build()
