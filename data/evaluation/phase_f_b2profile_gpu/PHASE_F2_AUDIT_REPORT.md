# Phase F.2 Measurement Audit Report

## Executive Summary

**Audit Status: PASS WITH LIMITATIONS**

This audit independently verifies the Phase F.2 experiment (CarbonGrid with B2 workload profile) against the frozen Phase F baseline. All data integrity checks pass. The paired comparison confirms that Phase F.2 measured energy is statistically significantly **lower** than all three frozen baselines (CarbonGrid, Always-FP16, Always-INT4). However, the absolute energy values (~0.048 Wh) are an order of magnitude smaller than the Phase G pilot (~0.43 Wh), and the quality proxy for INT4-selected prompts shows substantial degradation. The CO₂ values are estimates derived from measured energy × offline carbon intensity.

**Verdict:** The Phase F.2 result is valid as a sensitivity experiment showing that with the B2 workload profile, CarbonGrid routes 52% of prompts to INT4 and measured energy is slightly lower than the frozen baselines. However, the energy savings are small (median ~0.0014 Wh per prompt), the quality proxy for INT4 selections is substantially degraded, and the short output length (128 tokens) limits generalizability. **Not recommended for a headline energy-savings claim without these caveats clearly stated.**

---

## 1. Data Integrity Checks

| Check | Result | Details |
|-------|--------|---------|
| Frozen Phase F files unmodified | **PASS** | Timestamps unchanged (mtime 1790433403) |
| 182 prompt IDs match across all files | **PASS** | 182/182 matched across F2, Frozen CG, Frozen FP16, Frozen INT4 |
| Unique prompt IDs | **PASS** | 182 unique / 182 total |
| Missing values (energy, latency, power) | **PASS** | 0 missing |
| Non-finite values | **PASS** | 0 non-finite |
| Failed runs | **PASS** | 0/182 failed |
| Duplicate prompt IDs | **PASS** | 0 duplicates |

**All data integrity checks PASS.**

---

## 2. Independent Paired Comparison Results

### Energy (Wh) — Paired Differences (Phase F.2 minus Frozen Baseline)

| Comparison | n | Mean Diff | Median Diff | Std Diff | Increased | Decreased | Wilcoxon p-value | Bootstrap 95% CI |
|------------|---|-----------|-------------|----------|-----------|-----------|------------------|------------------|
| F2 vs Frozen CG | 182 | -0.001918 | **-0.001446** | 0.00243 | 36 | 146 | 1.87e-20 | [-0.00173, -0.00126] |
| F2 vs Frozen FP16 | 182 | -0.001781 | **-0.001237** | 0.00256 | 43 | 139 | 1.66e-18 | [-0.00164, -0.00080] |
| F2 vs Frozen INT4 | 182 | -0.001927 | **-0.001544** | 0.00231 | 33 | 149 | 5.92e-21 | [-0.00177, -0.00111] |

**Interpretation:** Phase F.2 measured energy is **statistically significantly lower** than all three frozen baselines (p < 1e-18). The median reduction is ~0.0012–0.0015 Wh per prompt (~2.5–3% relative reduction). For 146/182 prompts, energy decreased vs frozen CG; for 139/182 vs frozen FP16.

### Latency (ms) — Paired Differences

| Comparison | n | Mean Diff | Median Diff | Wilcoxon p-value |
|------------|---|-----------|-------------|------------------|
| F2 vs Frozen CG | 182 | +2,653 | **+4,344** | 5.85e-19 |
| F2 vs Frozen FP16 | 182 | +2,629 | **+4,295** | 1.17e-20 |
| F2 vs Frozen INT4 | 182 | -2,329 | **-887** | 1.08e-14 |

Phase F.2 is slower than frozen CG/FP16 (because 52% of prompts use INT4 which is ~2x slower), but faster than frozen INT4.

### Comparison with Original Phase F Results

| Comparison | Original Phase F (Phase 0.5 profile) | Phase F.2 (B2 workload profile) |
|------------|--------------------------------------|--------------------------------|
| CG vs FP16 energy | p=0.096, median diff +0.00025 Wh (NS) | F2 vs Frozen FP16: p<1e-18, median -0.00124 Wh |
| CG vs FP16 latency | p=0.92, median diff +9 ms (NS) | F2 vs Frozen FP16: p<1e-18, median +4,295 ms |
| CG vs INT4 energy | p=0.64, median diff -0.00009 Wh (NS) | F2 vs Frozen INT4: p<1e-18, median -0.00154 Wh |
| CG vs INT4 latency | p<1e-31, median diff -5,000 ms | F2 vs Frozen INT4: p<1e-14, median -887 ms |
| CG selection | 100% FP16 | **52% INT4, 48% FP16** |

**Key difference:** The B2 workload profile (INT4=0.0945 Wh < FP16=0.1061 Wh) reverses the energy order from Phase 0.5, causing CarbonGrid to route 52% of prompts to INT4.

---

## 3. Energy Discrepancy Investigation

| Experiment | Configuration | Median Output Tokens | Mean Energy (Wh) |
|------------|---------------|---------------------|------------------|
| **Phase G Pilot** | FP16 | 1,000 | 0.430 |
| **Phase G Pilot** | INT4 | 1,000 | 0.466 |
| **Phase F.2** | FP16 | 128 | 0.047 |
| **Phase F.2** | INT4 | 128 | 0.048 |

### Root Cause of Discrepancy

| Factor | Phase G Pilot | Phase F.2 |
|--------|---------------|-----------|
| `max_tokens` | Up to 2,000 (median 1,000) | Fixed at 128 |
| Median output tokens | 1,000 | 128 |
| Median latency (FP16) | 46,599 ms | 5,546 ms |
| Median latency (INT4) | 86,445 ms | 10,628 ms |
| Mean energy (FP16) | 0.430 Wh | 0.047 Wh |
| Mean energy (INT4) | 0.466 Wh | 0.048 Wh |

**Root Cause:** The Phase G pilot used much longer generations (8-9× more output tokens), leading to proportionally higher energy. Phase F.2 uses a fixed `max_tokens=128` for all prompts, resulting in much shorter generations and lower absolute energy.

**Measurement Protocol Consistency:** Both use the same `InferenceProvider` with NVML sampling at 20ms interval, `torch.cuda.synchronize()` before/after generation, 3 warm-up runs discarded, model unloading between configurations, and model-loading energy excluded. The measurement protocol is consistent; only the workload differs.

---

## 4. Quality and Routing Audit

### Decision Distribution (182 prompts)

| Precision | Count | % | Decision Reason |
|-----------|-------|---|-----------------|
| INT4 | 95 | 52.2% | `int4_lower_environmental_cost` |
| FP16 | 87 | 47.8% | `safety_gate_failed` |

**No** latency violations, resource constraints, or fallbacks occurred.

### Quality Proxy Scores

| Group | n | Mean | Median | Std |
|-------|---|------|--------|-----|
| FP16-selected | 87 | 1.000 | 1.000 | 0.000 |
| INT4-selected | 95 | 0.657 | 0.656 | 0.142 |

**Quality Method:** FP16 = 1.0 by reference convention; INT4 evaluated via `factual_semantic_entity`, `extraction_exact_field_value`, `explanation_semantic_structure_concept`, `reasoning_semantic_answer_logic`, `summarization_semantic_length_concept`, `scientific_semantic_concept`, `creative_semantic_proxy` against B2 FP16 reference outputs.

**Limitation:** Quality scores are **proxy metrics** (semantic similarity, entity match, structural comparison), not human-evaluated quality. The 0.66 median for INT4 reflects proxy degradation, not necessarily human-perceived quality loss.

### Safety Gate

| Metric | Value |
|--------|-------|
| Passed | 95/182 (52.2%) |
| Failed | 87/182 (47.8%) |
| INT4-selected avg safety probability | 0.645 |
| C.1.1 classifier false-safe rate (test set) | 40% |

**Critical:** The 40% false-safe rate on the C.1.1 test set applies to the classifier generally, **not** specifically to the 95 Phase F.2 INT4-selected prompts. The actual false-safe rate for these specific prompts is unknown.

### Safety Gate Failures by Task

| Task | Failed | Total | Failure Rate |
|------|--------|-------|--------------|
| coding | 26 | 26 | 100% |
| extraction | 17 | 20 | 85% |
| reasoning | 16 | 22 | 73% |
| creative | 14 | 22 | 64% |
| factual | 6 | 24 | 25% |
| summarization | 2 | 22 | 9% |
| scientific | 3 | 24 | 13% |

No coding prompts passed the safety gate; all routed to FP16.

---

## 5. Carbon Estimate Audit

### Formula Verification

`CO₂ (g) = Energy (Wh) × Carbon Intensity (gCO₂/kWh) / 1000`

- Carbon intensity: **200 gCO₂/kWh** (offline fallback, confirmed by `carbon_source: "offline"` for all 182 records)
- Formula verified for first 3 records: **match = True**
- All 182 records use offline fallback carbon intensity

### CO₂ Values (Estimated)

| Group | Median CO₂ (g) |
|-------|----------------|
| Overall | 0.00967 g |
| INT4-selected | 0.00973 g |
| FP16-selected | 0.00959 g |

**Labeling:** All CO₂ values are **estimates** derived from measured energy × offline carbon intensity. They are **not direct emissions measurements**.

---

## Unresolved Issues

| Issue | Impact |
|-------|--------|
| **Output length generalizability** | Fixed `max_tokens=128` limits generalizability to longer generations where energy differences would scale |
| **Quality proxy validity** | Semantic similarity proxy may not reflect human quality; 0.66 median could mask significant degradation |
| **Safety false-safe rate** | 40% test-set false-safe rate means some INT4 selections may be unsafe; actual rate for these 95 prompts unknown |
| **Single carbon intensity** | Offline 200 gCO₂/kWh fallback used for all prompts; no live or regional variation |
| **Short generations** | 128-token limit means energy differences are small absolute values (~0.001 Wh); may not scale linearly |
| **No INT4 coding prompts** | 0/26 coding prompts routed to INT4 (all safety failures) — limits evidence for code tasks |

---

## Final Status

**PASS WITH LIMITATIONS**

The Phase F.2 experiment is scientifically valid and reproducible. The measurements are internally consistent, the paired comparisons are statistically robust, and the frozen baseline is preserved. However:

1. **Energy savings are small** (median ~0.0014 Wh/prompt) and specific to short 128-token generations
2. **Quality proxy shows degradation** for INT4 selections (median 0.66 vs 1.0 reference)
3. **Safety gate fails 48%** of prompts, routing them to FP16
4. **No coding prompts use INT4** due to safety gate
5. **Quality proxy limitations** and **unknown false-safe rate** for these specific prompts

---

## Recommendation for Hackathon Presentation

| Claim | Supportable? | Recommended Wording |
|-------|--------------|---------------------|
| "CarbonGrid saves energy" | **No** | "With the B2 workload profile, CarbonGrid routes 52% of prompts to INT4 and measured energy is ~2-3% lower than baselines for short (128-token) generations" |
| "CarbonGrid reduces carbon emissions" | **No** | "Estimated CO₂ is ~2-3% lower due to lower measured energy, using offline carbon intensity" |
| "INT4 quality is acceptable" | **With caveats** | "INT4 quality proxy median is 0.66 vs FP16 reference; actual human quality unknown" |
| "CarbonGrid is safe" | **With caveats** | "Safety gate blocks 48% of prompts; classifier has 40% false-safe rate on test set; actual safety of INT4 selections unknown" |

**Headline Recommendation:** Present as a **sensitivity experiment** demonstrating that *when the energy profile favors INT4, CarbonGrid will route to INT4 and achieve modest measured energy reduction*, with explicit caveats about output length, quality proxy, and safety uncertainty.

---

## Files Read

| File | Purpose |
|------|---------|
| `data/evaluation/phase_f_b2profile_gpu/phase_f_raw_results.json` | Phase F.2 raw results (182 records) |
| `data/evaluation/phase_f_b2profile_gpu/phase_f2_analysis.json` | Existing analysis (verified against independent calc) |
| `data/evaluation/phase_f/phase_f_raw_results.json` | Frozen Phase F raw results (546 records) |
| `data/evaluation/phase_f/phase_f_analysis.json` | Frozen Phase F analysis |
| `data/evaluation/phase_g/pilot_results.json` | Phase G pilot (150 runs, 25 prompts) |
| `scripts/analyze_phase_f2.py` | Phase F.2 analysis script |
| `scripts/analyze_phase_f.py` | Original Phase F analysis script |
| `scripts/run_phase_f.py` | Phase F runner (with `--output_dir` support) |
| `carbongrid/inference/provider.py` | NVML measurement implementation |

---

## Files Created

| File | Description |
|------|-------------|
| `data/evaluation/phase_f_b2profile_gpu/PHASE_F2_AUDIT_REPORT.md` | This report |
| `data/evaluation/phase_f_b2profile_gpu/phase_f2_audit.json` | Machine-readable audit summary |

---

## Audit Artifacts

**Created by this audit:**
- `scripts/audit_phase_f2.py` — standalone read-only audit script
- `data/evaluation/phase_f_b2profile_gpu/PHASE_F2_AUDIT_REPORT.md` — this report
- `data/evaluation/phase_f_b2profile_gpu/phase_f2_audit.json` — machine-readable summary

All audit operations were read-only. No GPU inference, no modifications to frozen data, no threshold tuning.