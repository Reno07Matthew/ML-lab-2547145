# BRFSS 2023 data setup

The audited `LLCP2023.csv` is 461 MB and is intentionally excluded from the submission folder. Place it at the project root before reproduction.

Exact audited-file checksum:

```text
SHA-256  e34a22559c581147cf1a622fc470d70dc02ccc1800fb40fe2e3f32420e8ba244  LLCP2023.csv
```

Verify it with:

```bash
sha256sum LLCP2023.csv
```

Official CDC source: https://www.cdc.gov/brfss/annual_data/annual_2023.html

The current February 2025 SAS Transport download is:
https://www.cdc.gov/brfss/annual_data/2023/files/LLCP2023XPT.zip

After extracting `LLCP2023.XPT`, a CSV can be created with pandas:

```bash
.venv/bin/python -c "import pandas as pd; pd.read_sas('LLCP2023.XPT', format='xport', encoding='latin1').to_csv('LLCP2023.csv', index=False)"
```

Important: the current official XPT has 345 variables, while the supplied and audited CSV has 350 columns. The project documents that version difference in `reports/column_discrepancy.md`. Exact numerical reproduction requires the audited CSV matching the checksum above; converting the revised official XPT is suitable for rerunning the method but may not reproduce the frozen artifacts byte-for-byte.
