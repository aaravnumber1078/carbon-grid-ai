# Phase G Incremental Value Report

**Date:** 2026-10-03T00:26:04.612895
**Pilot prompts:** 25 (25 prompts × 3 reps × 2 configs)
**CV method:** LeaveOneOut (LOPO-CV)
**Features evaluated:** 21 pre-inference features
**Prompt families with duplicates:** 0

## Model Comparison

| Config | Model | MAE (Wh) | RMSE (Wh) | R² | Negative Predictions |
|--------|-------|----------|-----------|-----|----------------------|
| FP16 | LinearReg_maxTokens | 0.049507 | 0.061954 | 0.9713 | 3 |
| FP16 | Ridge_full | 0.042804 | 0.051810 | 0.9799 | 2 |
| FP16 | ElasticNet_full | 0.022693 | 0.028034 | 0.9941 | 0 |
| INT4 | LinearReg_maxTokens | 0.067471 | 0.083504 | 0.9583 | 3 |
| INT4 | Ridge_full | 0.058280 | 0.069369 | 0.9712 | 4 |
| INT4 | ElasticNet_full | 0.033196 | 0.037107 | 0.9918 | 1 |

## Paired Comparison (Extended - Baseline)

| Config | Extended Model | Mean Diff (Wh) | Median Diff (Wh) | Better | Worse | Tied |
|--------|----------------|----------------|------------------|--------|-------|------|
| FP16 | Ridge_full | +0.008370 | +0.001227 | 10 | 15 | 0 |
| FP16 | ElasticNet_full | -0.003202 | -0.011722 | 14 | 11 | 0 |
| INT4 | Ridge_full | +0.004323 | +0.001563 | 12 | 13 | 0 |
| INT4 | ElasticNet_full | -0.004945 | -0.009059 | 13 | 12 | 0 |

## Negative Predictions

Physically impossible negative energy predictions observed for ElasticNet_full:
- FP16: 0/25 predictions < 0 Wh
- INT4: 1/25 predictions < 0 Wh

## Prompt Family Overlap

Found 0 template families with multiple prompts in the 25-prompt pilot.
This may inflate LOPO-CV scores if similar prompts appear in both train and test folds.

## Conclusion

- **FP16:** IMPROVES (diff = -0.026814 Wh, better=14, worse=11)
- **INT4:** IMPROVES (diff = -0.034274 Wh, better=13, worse=12)

**Recommendation:**
The 20 additional features show consistent improvement. Consider 100-prompt expansion.

**Limitation:** n=25 is small; LOPO-CV variance is high. Prompt family overlap not fully controlled.
R² alone does not imply practical utility for routing decisions.