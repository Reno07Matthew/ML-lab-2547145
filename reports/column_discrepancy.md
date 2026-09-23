# 350-versus-345 column discrepancy

- Supplied `LLCP2023.csv`: 350 columns.
- Current CDC February 2025 SAS Transport release: 345 variables.
- CSV-only fields: `BIRTHSEX`, `CELSXBRT`, `LNDSXBRT`, `RCSGEND1`, `RCSXBRTH`, `TRNSGNDR`.
- Current-layout field absent from the CSV: `RCSBORG1` (Sex of child: boy/girl).
- Net difference: six legacy fields removed and one replacement field added, producing 350 - 6 + 1 = 345.

CDC states that the February 2025 file was modified to comply with Presidential executive orders and that removed questions can cause apparently inconsistent missing values. No field-by-field CDC release note was located. The six CSV-only names are sex-at-birth/gender-related items, so that dataset-level explanation is consistent with the observed difference; this is an inference, not a more specific CDC statement.

All modelling uses an explicit whitelist of requested variables that appear in the current documented layout. Every CSV-only field, `RCSBORG1`, `_LLCPWT`, and all forbidden leakage fields are excluded from predictors.

Sources:
- https://www.cdc.gov/brfss/annual_data/annual_2023.html
- https://www.cdc.gov/brfss/annual_data/2023/llcp_varlayout_23_onecolumn.html
