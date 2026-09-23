#!/usr/bin/env python3
"""Create the raw-data-free submission folder."""

from __future__ import annotations

import shutil
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
DESTINATION = PROJECT / "submission_ready"
ITEMS = [
    ".gitignore", "README.md", "requirements.txt", "app.py",
    "USCODE23_LLCP_021924.HTML", "CIA 3 @Machine learning .docx.pdf",
    "scripts", "notebooks", "reports", "figures", "models", "data", "output", "DATA_DOWNLOAD.md",
    "SUBMISSION_CHECKLIST.md",
]


def main() -> None:
    if DESTINATION.exists():
        shutil.rmtree(DESTINATION)
    DESTINATION.mkdir()
    for name in ITEMS:
        source = PROJECT / name
        if not source.exists():
            raise FileNotFoundError(source)
        target = DESTINATION / name
        shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.inspect.ndjson")) if source.is_dir() else shutil.copy2(source, target)


if __name__ == "__main__":
    main()
