# BRFSS 2023 initial data audit

Status: audit complete; modelling intentionally not started.

## Cohort and target

- Supplied dataset: `LLCP2023.csv` (433,323 rows, 350 columns).
- `_AGE80` filter 18-39: 99,690 rows.
- Valid `_MICHD` cohort: 99,114 rows; 576 young-adult rows have a missing target and are excluded.
- Positive (`_MICHD=1`): 1,050 (1.0594%).
- Negative (`_MICHD=2`): 98,064 (98.9406%).
- Survey-weighted positive percentage using `_LLCPWT`: 1.1692% (descriptive only).
- Positive:negative ratio: 1:93.39.
- Duplicate respondent IDs using `_STATE` + `SEQNO`: 0.
- Full-file `_MICHD` frequencies match the official codebook.
- `_MICHD` values inconsistent with the documented `CVDINFR4`/`CVDCRHD4` derivation: 0.
- Missing young-adult targets arise from source responses: CVDINFR4=2, CVDCRHD4=7 (n=198); CVDINFR4=2, CVDCRHD4=9 (n=10); CVDINFR4=7, CVDCRHD4=2 (n=280); CVDINFR4=7, CVDCRHD4=7 (n=77); CVDINFR4=9, CVDCRHD4=2 (n=3); CVDINFR4=9, CVDCRHD4=9 (n=8).

## Selected-feature missingness after age and target filtering

```csv
feature,usable_n,usable_pct,raw_missing_n,invalid_response_n,effective_missing_n,effective_missing_pct
_AGE80,99114,100.0,0,0,0,0.0
SEXVAR,99114,100.0,0,0,0,0.0
EXERANY2,98923,99.8073,0,191,191,0.1927
_SMOKER3,93635,94.472,0,5479,5479,5.528
DRNKANY6,91899,92.7205,0,7215,7215,7.2795
_RFBING6,90869,91.6813,0,8245,8245,8.3187
_RFDRHV8,90966,91.7792,0,8148,8148,8.2208
_BMI5,88822,89.616,10292,0,10292,10.384
GENHLTH,98930,99.8144,1,183,184,0.1856
PHYSHLTH,97435,98.306,0,1679,1679,1.694
MENTHLTH,97509,98.3807,0,1605,1605,1.6193
BPHIGH6,98732,99.6146,0,382,382,0.3854
TOLDHI3,68238,68.848,30454,422,30876,31.152
CVDSTRK3,98996,99.8809,0,118,118,0.1191
CHCKDNY2,98925,99.8093,0,189,189,0.1907
DIABETE4,98917,99.8012,0,197,197,0.1988
```

`effective_missing_n` combines blank values with official refused, unknown/not-sure, and derived missing codes. Exact observed codes and labels are in `feature_audit.csv`.

## Planned feature configurations

- Lifestyle-only: `EXERANY2`, `_SMOKER3`, `DRNKANY6`, `_RFBING6`, `_RFDRHV8`.
- Lifestyle plus low-cost non-invasive health/history: all lifestyle fields plus `_BMI5`, `GENHLTH`, `PHYSHLTH`, `MENTHLTH`, `BPHIGH6`, `TOLDHI3`, `CVDSTRK3`, `CHCKDNY2`, `DIABETE4`.
- `_AGE80` and `SEXVAR` are retained for cohort definition and subgroup evaluation, not included in the literal lifestyle-only predictor list.

## Leakage and interpretation controls

- `CVDINFR4` and `CVDCRHD4` are excluded because the official codebook states that `_MICHD` is derived from them.
- `HASYMP1`-`HASYMP6` are excluded because they measure knowledge of possible symptoms, not experienced symptoms.
- `_MICHD` is prevalent self-reported history of CHD/MI in a cross-sectional survey. It is not a future-event label and cannot establish future heart-attack risk or causality.
- No train/validation/test split, preprocessing fit, resampling, or model training has been run in this phase.
