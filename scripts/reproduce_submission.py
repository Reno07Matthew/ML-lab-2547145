#!/usr/bin/env python3
"""Reproduce and verify submission artifacts without model fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import joblib
import nbformat
import pandas as pd
from streamlit.testing.v1 import AppTest


PROJECT = Path(__file__).resolve().parents[1]
THRESHOLD = 0.7879034938787791
WINNER = ("extended_low_cost", "logistic_regression", "class_weight")


def run(*parts: str) -> None:
    environment = os.environ.copy()
    environment.update({
        "PATH": f"{Path(sys.executable).parent}{os.pathsep}{environment.get('PATH', '')}",
        "PYTHONNOUSERSITE": "1",
        "IPYTHONDIR": str(PROJECT / ".cache" / "ipython"),
        "JUPYTER_CONFIG_DIR": str(PROJECT / ".cache" / "jupyter"),
    })
    subprocess.run([sys.executable, *parts], cwd=PROJECT, env=environment, check=True)


def verify() -> dict:
    metrics = pd.read_csv(PROJECT / "reports" / "final_model_comparison.csv")
    required_metrics = {"positive_class_f1", "macro_f1", "weighted_f1", "accuracy", "balanced_accuracy"}
    assert required_metrics <= set(metrics) and "f1" not in metrics.columns

    selection = json.loads((PROJECT / "reports" / "model_selection.json").read_text())
    chosen = selection["best_model_selected_on_validation"]
    assert (chosen["feature_config"], chosen["model"], chosen["imbalance_treatment"]) == WINNER
    assert abs(float(chosen["threshold"]) - THRESHOLD) < 1e-12

    audit = pd.read_csv(PROJECT / "reports" / "model_selection_audit.csv")
    winner = audit[audit["selected_winner"].astype(str).str.lower().eq("true")]
    assert len(winner) == 1 and int(winner.iloc[0]["validation_rank"]) == 1
    lr = audit[(audit.feature_config == WINNER[0]) & (audit.model == WINNER[1])]
    margin = float(lr.loc[lr.imbalance_treatment.eq("class_weight"), "validation_f2"].iloc[0] - lr.loc[lr.imbalance_treatment.eq("none"), "validation_f2"].iloc[0])
    assert 0 < margin < 0.001

    pipeline = joblib.load(PROJECT / "models" / "extended_low_cost__logistic_regression.joblib")
    assert hasattr(pipeline, "predict_proba")

    workbook = PROJECT / "reports" / "model_results.xlsx"
    with zipfile.ZipFile(workbook) as archive:
        assert archive.testzip() is None and "xl/workbook.xml" in archive.namelist()
        comparison_sheet = archive.read("xl/worksheets/sheet1.xml").decode()
        assert {"positive_class_f1", "macro_f1", "weighted_f1", "accuracy", "balanced_accuracy"} <= {name for name in required_metrics if name in comparison_sheet}

    notebook = nbformat.read(PROJECT / "notebooks" / "01_eda.ipynb", as_version=4)
    errors = [output for cell in notebook.cells if cell.cell_type == "code" for output in cell.get("outputs", []) if output.get("output_type") == "error"]
    assert not errors and any(cell.cell_type == "code" and cell.get("execution_count") for cell in notebook.cells)

    app = AppTest.from_file(str(PROJECT / "app.py"), default_timeout=30).run()
    assert not app.exception
    visible = " ".join(item.value for item in [*app.title, *app.error, *app.caption])
    assert "screening-score" in visible.lower() or "screening score" in visible.lower()
    assert "not a diagnosis" in visible.lower() and "not been validated for india" in visible.lower()
    assert "0.787903" in visible
    app.button[0].click().run()
    assert not app.exception and app.metric[0].label == "Research screening score"
    assert len(app.dataframe) == 1 and (len(app.info) + len(app.warning)) == 1

    expected = [
        PROJECT / "output" / "pdf" / "final_assignment_report.pdf",
        PROJECT / "reports" / "model_results.xlsx",
        PROJECT / "reports" / "subgroup_metrics.csv",
        PROJECT / "reports" / "shap_grouped_importance.csv",
        PROJECT / "figures" / "shap" / "03_grouped_shap_importance.png",
    ]
    missing = [str(path.relative_to(PROJECT)) for path in expected if not path.exists()]
    artifact_roots = [PROJECT / name for name in ("reports", "figures", "models", "data", "notebooks", "output")]
    artifacts = sorted(path for root in artifact_roots for path in root.rglob("*") if path.is_file() and not path.name.endswith(".inspect.ndjson") and path.name not in {"artifact_manifest.csv", "reproducibility_status.json"})
    manifest = pd.DataFrame([{
        "project_relative_path": str(path.relative_to(PROJECT)),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    } for path in artifacts])
    manifest.to_csv(PROJECT / "reports" / "artifact_manifest.csv", index=False)
    status = {
        "downstream_artifact_reproduction": "PASS",
        "model_loading": "PASS",
        "metrics_workbook_integrity": "PASS",
        "notebook_execution": "PASS",
        "streamlit_app_test": "PASS",
        "fixed_winner": "PASS",
        "fixed_threshold": THRESHOLD,
        "validation_f2_class_weight_margin": margin,
        "verified_artifact_count": len(artifacts),
        "missing_files": missing,
    }
    assert not missing
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=PROJECT / "LLCP2023.csv")
    args = parser.parse_args()
    data = args.data.resolve()
    run("scripts/audit_data.py", "--data", str(data))
    run("scripts/run_project.py", "--data", str(data), "--stage", "eda")
    run("-m", "nbconvert", "--to", "notebook", "--execute", "--inplace", "notebooks/01_eda.ipynb", "--ExecutePreprocessor.timeout=600")
    run("scripts/post_model.py", "--data", str(data))
    run("scripts/export_report.py")
    status = verify()
    (PROJECT / "reports" / "reproducibility_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
