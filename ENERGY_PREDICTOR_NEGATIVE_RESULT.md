# ENERGY PREDICTOR NEGATIVE RESULT

## Phase F.1 Audit Summary

**Audit Date:** 2026-09-26  
**Status:** FAIL — Do Not Proceed to Phase F.2  
**Experiment Preserved:** All F.1 artifacts preserved for reproducibility

---

## Executive Summary

**Audit Verdict: FAIL — Do Not Proceed to Phase F.2**

The Phase F.1 Energy Predictor audit revealed critical methodological issues. The reported **100% directional accuracy** is an artifact of the B2 dataset composition (all 37 test samples have INT4 < FP16), not evidence of model quality. A critical feature leakage was identified: `task_type` is dataset-generation metadata unavailable at inference time.

**Verdict: Do not proceed to Phase F.2.**

---

## Critical Findings

### 1. Task Type Feature Leakage (CRITICAL)
- **Feature:** `task_type` used as categorical predictor
- **Problem:** `task_type` is dataset-generation metadata, unavailable at inference time for new prompts
- **Location:** `train_energy_predictor.py:121` → `row["task_type"] = r.get("task_type", "unknown")`
- **Impact:** Model cannot be deployed; feature unavailable at inference time for new prompts

### 2. Dataset Lacks Target Variation
- **B2 Dataset (182 samples):** ALL samples have INT4 < FP16 (negative energy difference)
- **Sign distribution:** INT4 < FP16: 182, INT4 > FP16: 0, INT4 = FP16: 0
- **Energy difference range:** -0.024976 to -0.001265 Wh (all negative)
- **Mean difference:** -0.0116 Wh (INT4 uses less energy)

### 3. 100% Directional Accuracy Is an Artifact
- All 37 test samples have INT4 < FP16 (same sign)
- Model predicts "INT4 lower energy" for all 37 → 100% accuracy
- This measures dataset composition, not model quality

### 5. Baseline MAE Correction
| Metric | Previously Reported | Corrected |
|--------|-------------------|-----------|
| Baseline MAE | 0.0119 Wh | **0.0121 Wh** |
| Directional Accuracy | 0% | 0% |

---

## Key Metrics

### Corrected Baseline (Static Phase 0.5 Profile)
| Metric | Value |
|--------|-------|
| MAE | 0.0121 Wh |
| Directional Accuracy | 0% |
| R² | -8.46 |

### F.1 Predictor (37 test samples)
| Metric | Value |
|--------|-------|
| MAE | 0.00193 Wh |
| RMSE | 0.00266 Wh |
| R² | 0.581 |
| Directional Accuracy | 100% (artifact) |
| Within 10% | 37.8% |

### Improvement Over Baseline
| Metric | Baseline | Predictor | Improvement |
|--------|----------|-----------|-------------|
| MAE | 0.0121 Wh | 0.0019 Wh | -84% |
| Dir. Acc | 0% | 100% | +100%* |
| R² | -8.46 | 0.58 | +9.04 |

*Directional accuracy improvement is misleading — all test samples have same sign*

---

## Key Conclusions

### 1. Dataset Fundamentally Lacks Variation
- All 182 B2 samples: INT4 < FP16 (negative energy difference)
- No positive energy difference samples exist in B2
- B1 has 3/50 positive (only factual task, low complexity)

### 2. 100% Directional Accuracy Is Misleading
- All 37 test samples have same sign (INT4 < FP16)
- Model predicts negative for all → 100% accuracy by construction
- This measures dataset composition, not model quality

### 3. Static Baseline Is Wrong for B2 Workload
| Profile | FP16 | INT4 | INT4-FP16 |
|---------|------|------|-----------|
| Phase 0.5 | 0.0488 | 0.0493 | +0.0005 |
| B2 Mean | 0.105 | 0.094 | -0.011 |

Static profile reverses the actual relationship for B2 workload.

### 5. Sample Size Insufficient
- 182 samples, ~25 features (17 numeric + 8 task types)
- Rule of thumb: 10-20 samples/feature for linear models
- Required: 250-500 samples for 25 features
- Status: INSUFFICIENT for complex models

---

## Recommendations

### Immediate Actions Required
1. **Remove `task_type` feature** — dataset-generation metadata unavailable at inference
2. **Remove DEBUG print statements** from production code
3. **Fix baseline MAE documentation** → 0.0121 Wh (not 0.0119)
4. **Document limitation:** 100% directional accuracy is dataset artifact

### Future Work Requirements
Before any production deployment:
1. Collect independent workload/hardware dataset with genuine FP16/INT4 variation
2. Remove `task_type` feature or implement runtime task classification
3. Collect mixed-sign dataset (both INT4<FP16 and INT4>FP16 cases)
4. Validate on independent hardware/workload combinations

---

## Artifacts Preserved (Do Not Modify)

| Artifact | Path |
|----------|------|
| Trained predictor | `models/energy_predictor/energy_predictor.joblib` |
| Feature extractor | `models/energy_predictor/feature_extractor.joblib` |
| Training results | `data/evaluation/phase_f/energy_predictor_results.json` |
| Evaluation results | `data/evaluation/phase_f/energy_predictor_evaluation.json` |
| Test split | `data/evaluation/phase_f/energy_predictor_test_split.json` |
| B2 dataset | `data/processed/b2_dataset.json` |
| Phase 0.5 validation | `phase0_5_validation_results.json` |

---

## Final Recommendation

**Do NOT proceed to Phase F.2.**

**Recommended path:**
1. **Option A (if continuing):** Remove `task_type`, retrain on B2+B1 with separate FP16/INT4 energy heads, evaluate on mixed-sign data
2. **Option C (recommended):** Document as negative result, move to CarbonGrid evaluation/demo

The dataset fundamentally lacks variation in the target sign. Any "100% directional accuracy" claim is an artifact of the dataset, not model quality.

---

**Audit completed: 2026-09-26**  
**Audit verdict: FAIL — Do not proceed to Phase F.2**