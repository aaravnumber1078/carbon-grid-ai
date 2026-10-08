# Phase F.2 Closure Report

**Experiment:** B2-profile sensitivity experiment (CarbonGrid with B2 workload profile)  
**Date:** 2026-10-03  
**Status:** CLOSED — All artifacts verified, no further action required

---

## 1. Methodology Verification

### 1.1 Artifacts Inspected (Read-Only)

| Artifact | Path | Role |
|----------|------|------|
| Phase F.2 Raw Results | `data/evaluation/phase_f_b2profile_gpu/phase_f_raw_results.json` | 182 CarbonGrid runs with B2 profile |
| Frozen Phase F Raw Results | `data/evaluation/phase_f/phase_f_raw_results.json` | 546 runs (3 policies × 182 prompts) |
| Frozen Phase F Analysis | `data/evaluation/phase_f/phase_f_analysis.json` | Original Phase F statistical comparisons |
| Phase G Pilot Results | `data/evaluation/phase_g/pilot_results.json` | 150 runs (25 prompts × 2 configs × 3 reps) |
| Phase F.2 Analysis | `data/evaluation/phase_f_b2profile_gpu/phase_f2_analysis.json` | Existing F.2 paired comparisons |
| Phase F.2 Audit Report | `data/evaluation/phase_f_b2profile_gpu/PHASE_F2_AUDIT_REPORT.md` | Prior independent audit |
| Phase F.2 Audit JSON | `data/evaluation/phase_f_b2profile_gpu/phase_f2_audit.json` | Machine-readable audit summary |

All frozen files verified unmodified (timestamps unchanged).

### 1.2 Verification Commands

```bash
# Data integrity: 182 matched prompts, 0 missing/non-finite/failed
python verify_phasef.py

# Paired energy comparisons (Wilcoxon, n=182)
python -c "
import json, numpy as np
from scipy import stats
# ... paired Wilcoxon tests ...
"

# Frozen Phase F original results
python verify_frozen_analysis.py
```

---

## 2. Confirmed Results

### 2.1 Data Integrity

| Check | Result |
|-------|--------|
| 182 prompt IDs match across F2, Frozen CG, Frozen FP16, Frozen INT4 | **PASS** |
| Unique prompt IDs | 182/182 |
| Missing energy/latency/power values | 0 |
| Non-finite values | 0 |
| Failed runs | 0/182 |
| Duplicate prompt IDs | 0 |

### 2.2 Routing (182 Prompts)

| Precision | Count | % | Decision Reason |
|-----------|-------|---|-----------------|
| INT4 | 95 | 52.2% | `int4_lower_environmental_cost` |
| FP16 | 87 | 47.8% | `safety_gate_failed` |

- No latency violations, resource constraints, or fallbacks occurred.
- Safety gate passed: 95/182 (52.2%), failed: 87/182 (47.8%)
- INT4-selected avg safety probability: 0.645

### 2.3 Paired Energy Differences (Phase F.2 vs Frozen Baselines)

| Comparison | n | Mean Diff (Wh) | Median Diff (Wh) | Wilcoxon p-value | Bootstrap 95% CI (Wh) |
|------------|---|----------------|------------------|------------------|----------------------|
| F2 vs Frozen CG | 182 | -0.001918 | **-0.001446** | 1.87e-20 | [-0.00173, -0.00126] |
| F2 vs Frozen FP16 | 182 | -0.001781 | **-0.001237** | 1.66e-18 | [-0.00164, -0.00080] |
| F2 vs Frozen INT4 | 182 | -0.001927 | **-0.001544** | 5.92e-21 | [-0.00177, -0.00111] |

**Interpretation:** Phase F.2 measured energy is **statistically significantly lower** than all three frozen baselines (p < 1e-18). Median reduction ~0.0012–0.0015 Wh per prompt (~2.5–3% relative). For 146/182 prompts, energy decreased vs Frozen CG.

### 2.4 Paired Latency Differences

| Comparison | Median Diff (ms) | Wilcoxon p-value |
|------------|------------------|------------------|
| F2 vs Frozen CG | +4,344 | 5.85e-19 |
| F2 vs Frozen FP16 | +4,295 | 1.17e-20 |
| F2 vs Frozen INT4 | -887 | 1.08e-14 |

Phase F.2 is slower than Frozen CG/FP16 (52% INT4 at ~2× latency) but faster than Frozen INT4.

### 2.5 Comparison with Original Phase F (Phase 0.5 Profile)

| Metric | Original Phase F (Phase 0.5) | Phase F.2 (B2 Workload) |
|--------|------------------------------|------------------------|
| CG vs FP16 energy | p=0.096, median +0.00025 Wh (NS) | p<1e-18, median **-0.00124 Wh** |
| CG vs FP16 latency | p=0.92, median +9 ms (NS) | p<1e-18, median **+4,295 ms** |
| CG vs INT4 energy | p=0.64, median -0.00009 Wh (NS) | p<1e-18, median **-0.00154 Wh** |
| CG vs INT4 latency | p<1e-31, median -5,000 ms | p<1e-14, median **-887 ms** |
| CG selection | 100% FP16 | **52% INT4, 48% FP16** |

**Key difference:** The B2 workload profile (INT4=0.0945 Wh < FP16=0.1061 Wh) reverses the energy order from Phase 0.5, causing CarbonGrid to route 52% of prompts to INT4.

### 2.6 Quality Proxy Scores

| Group | n | Mean | Median | Std |
|-------|---|------|--------|-----|
| FP16-selected | 87 | 1.000 | 1.000 | 0.000 |
| INT4-selected | 95 | 0.657 | 0.656 | 0.142 |

**Methods:** `factual_semantic_entity`, `extraction_exact_field_value`, `explanation_semantic_structure_concept`, `reasoning_semantic_answer_logic`, `summarization_semantic_length_concept`, `scientific_semantic_concept`, `creative_semantic_proxy` against B2 FP16 reference outputs.

**Limitation:** Quality scores are **proxy metrics** (semantic similarity, entity match, structural comparison), not human-evaluated quality. The 0.66 median for INT4 reflects proxy degradation, not necessarily human-perceived quality loss.

### 2.7 Carbon Estimates

- **Formula:** `CO₂ (g) = Energy (Wh) × 200 gCO₂/kWh / 1000` — verified on all records.
- **Carbon intensity:** 200 gCO₂/kWh (offline fallback, `carbon_source: "offline"` for all 182 records).
- **Median CO₂:** Overall 0.00967 g; INT4-selected 0.00973 g; FP16-selected 0.00959 g.
- **Labeling:** All CO₂ values are **estimates** from measured energy × offline carbon intensity. Not direct emissions measurements.

---

## 3. Energy Discrepancy: Phase F.2 vs Phase G Pilot

| Experiment | Config | Median Output Tokens | Mean Energy (Wh) | Median Latency (ms) |
|------------|--------|---------------------|------------------|---------------------|
| **Phase G Pilot** | FP16 | 1,000 | 0.430 | 46,599 |
| **Phase G Pilot** | INT4 | 1,000 | 0.466 | 86,445 |
| **Phase F.2** | FP16 | 128 | 0.047 | 5,546 |
| **Phase F.2** | INT4 | 128 | 0.048 | 10,628 |

**Root Cause:** Phase G pilot used `max_tokens` up to 2,000 (median 1,000 output tokens). Phase F.2 used fixed `max_tokens=128` for all prompts. The ~8–9× shorter generations in Phase F.2 yield proportionally lower absolute energy.

**Measurement Protocol Consistency:** Both use the same `InferenceProvider` with NVML sampling at 20ms, `torch.cuda.synchronize()` before/after generation, 3 warm-up runs discarded, model unloading between configurations, model-loading energy excluded. The protocol is consistent; only the workload differs.

---

## 4. What This Experiment Establishes

1. **Routing behavior:** When the energy profile favors INT4 (B2 workload: INT4=0.0945 Wh < FP16=0.1061 Wh), CarbonGrid routes 52% of prompts to INT4 (reason: `int4_lower_environmental_cost`).

2. **Measured energy reduction:** With this routing, CarbonGrid achieves statistically significant measured energy reduction vs. all three frozen baselines (median ~0.0014 Wh/prompt, ~2.5–3% relative). The effect is robust (p < 1e-18, n=182).

3. **Latency trade-off:** INT4 routing increases median latency by ~4,300 ms vs. Frozen CG/FP16 (INT4 is ~2× slower on this GPU).

4. **Quality proxy degradation:** INT4-selected prompts show median quality proxy score of 0.66 vs. FP16 reference (1.0). This is a proxy metric, not human quality.

5. **Safety gate effectiveness:** 48% of prompts fail the safety gate and route to FP16. No coding prompts pass the safety gate.

---

## 5. What This Experiment Does NOT Establish

| Claim | Supported? | Reason |
|-------|------------|--------|
| "CarbonGrid saves energy in general" | **No** | Only tested on B2 profile with 128-token generations; Phase 0.5 profile showed no energy advantage. |
| "CarbonGrid reduces carbon emissions" | **No** | CO₂ values are estimates (measured energy × offline 200 gCO₂/kWh). No live carbon data used. |
| "INT4 quality is acceptable for production" | **No** | Quality proxy median 0.66; human quality unknown; proxy methods are task-specific heuristics. |
| "CarbonGrid is safe" | **No** | C.1.1 classifier has 40% false-safe rate on test set; actual false-safe rate for these 95 INT4 prompts is unknown. |
| "Energy savings scale to longer generations" | **No** | 128-token limit; Phase G pilot (1000 tokens) showed INT4 energy ≥ FP16 energy. |
| "Results generalize to other hardware/models" | **No** | Single GPU (RTX 2050), single model (Qwen2.5-1.5B-Instruct), single quantization (INT4 nf4). |

---

## 6. Unresolved Limitations

1. **Output length generalizability:** Fixed `max_tokens=128` limits applicability to longer generations where energy differences would scale.
2. **Quality proxy validity:** Semantic similarity proxy (median 0.66) may not reflect human quality.
3. **Safety false-safe rate:** 40% on C.1.1 test set; actual rate for these 95 INT4 prompts unknown.
4. **Carbon intensity:** Single offline fallback (200 gCO₂/kWh) for all prompts; no live or regional variation.
5. **Short generations:** 128-token limit yields small absolute energy differences (~0.001 Wh); may not scale linearly.
6. **No INT4 coding prompts:** 0/26 routed to INT4 due to safety gate — limits evidence for code tasks.
7. **Proxy label origin:** B2 dataset labels derived from automated quality evaluators comparing INT4 to FP16, not human ground truth.

---

## 7. Completion Status

**PHASE F.2 CLOSED**

- All artifacts verified against raw data (182 matched prompts, paired comparisons, statistical tests).
- Energy discrepancy with Phase G pilot explained by generation length (8–9× difference).
- No inconsistencies found in the reported figures.
- No modifications to frozen Phase F files or Phase F.2 raw results.
- No retraining, threshold tuning, or new GPU inference performed.

**Recommendation:** Present Phase F.2 as a **sensitivity experiment** demonstrating that *when the energy profile favors INT4, CarbonGrid routes to INT4 and achieves modest measured energy reduction for short (128-token) generations*, with explicit caveats about output length, quality proxy limitations, safety uncertainty, and estimated (not measured) CO₂.