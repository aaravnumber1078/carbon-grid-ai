# Phase G Closure Report: Exploratory Energy Prediction

**Date:** 2026-10-03  
**Status:** EXPLORATORY STUDY CLOSED — NOT VALIDATED FOR PRODUCTION USE

---

## 1. Artifacts Inspected (Read-Only)

| Artifact | Path | Role |
|----------|------|------|
| Pilot Prompts (25) | `data/evaluation/phase_g/pilot_prompts.json` | Stratified selection from 340 dev prompts |
| Pilot Results (150 runs) | `data/evaluation/phase_g/pilot_results.json` | 25 prompts × 3 reps × 2 configs (FP16/INT4) |
| Full Prompt Set (400) | `data/evaluation/phase_g/prompts.json` | 340 dev / 60 locked test, stratified by category & length |
| Feasibility Report | `data/evaluation/phase_g/phase_g_feasibility_report.json` | LOPO-CV on 25 prompts, 21 features |
| Incremental Value Report | `data/evaluation/phase_g/phase_g_incremental_value_report.json` | 20 features vs max_tokens baseline |
| Incremental Value MD | `data/evaluation/phase_g/PHASE_G_INCREMENTAL_VALUE_REPORT.md` | Human-readable summary |
| Pilot Selection Script | `scripts/select_phase_g_pilot.py` | Stratified random seed 42 |
| Pilot Measurement Script | `scripts/measure_phase_g_pilot.py` | 3 reps, randomized config order, thermal stabilization |
| Feasibility Script | `scripts/phase_g_feasibility.py` | LOPO-CV with nested GridSearchCV |
| Incremental Value Script | `scripts/phase_g_incremental_value.py` | Paired comparison vs max_tokens-only |
| Audit Results | `data/evaluation/phase_g/audit_results.json` | Data integrity: 400 prompts, no leakage |

All frozen files verified unmodified. No GPU inference, no new training, no test-set access.

---

## 2. Verified Results Summary

### 2.1 Pilot Dataset & Measurement

| Metric | Value | Verified |
|--------|-------|----------|
| Prompts measured | 25 (from 340 dev) | ✅ |
| Repetitions per config | 3 | ✅ |
| Total runs | 150 (all successful) | ✅ |
| Config order | Randomized per prompt | ✅ |
| Warm-up runs | 3 discarded per config | ✅ |
| Thermal stabilization | 30s between configs | ✅ |
| Power sampling | NVML @ 20ms interval | ✅ |
| Model unloading | Between configs | ✅ |
| Max tokens | Per-prompt (50–2000) | ✅ |

### 2.2 Measured Energy & Latency (Per-Run Means)

| Config | Mean Energy (Wh) | Median Latency (ms) | Mean Power (W) | Median Output Tokens |
|--------|------------------|---------------------|----------------|----------------------|
| FP16 | 0.4298 | 46,599 | 32.9 | 1,000 |
| INT4 | 0.4658 | 86,445 | 19.3 | 1,000 |

**Confirmed:** INT4 energy > FP16 energy on average (+8.4% relative).  
**Confirmed:** INT4 latency ≈ 1.85× FP16 latency.  
**Confirmed:** INT4 power ≈ 0.59× FP16 power.

### 2.3 Per-Prompt Energy Comparison

| Comparison | Count | Details |
|------------|-------|---------|
| INT4 lower energy than FP16 | 8/25 prompts | Extraction, scientific, creative, summarization, explanation, multi-turn |
| INT4 higher energy than FP16 | 17/25 prompts | Coding, factual, reasoning, other creative |

**Confirmed:** INT4 used less energy for only 8 of 25 pilot prompts. On the remaining 17, INT4 consumed more energy despite lower power, because latency was ~2× longer.

### 2.4 Feasibility Assessment (LOPO-CV, n=25)

| Config | Baseline (train-mean) MAE | Best Model (ElasticNet) MAE | R² | Beats Baseline? |
|--------|---------------------------|------------------------------|-----|-----------------|
| FP16 | 0.3038 Wh | 0.0227 Wh | 0.9941 | ✅ Yes |
| INT4 | 0.3397 Wh | 0.0332 Wh | 0.9918 | ✅ Yes |

**Interpretation:** Both configs show LOPO-CV MAE well below the fold-specific training-mean baseline. However:
- n=25 is very small for LOPO-CV; variance is high.
- Prompt families with multiple members exist (0 in pilot, but possible in full dev set).
- R² alone does not imply practical utility for routing decisions.

### 2.5 Incremental Value of 20 Additional Features

| Config | Baseline (max_tokens only) MAE | Extended (21 features) MAE | Δ MAE | Per-Prompt (Better/Worse/Tied) |
|--------|-------------------------------|----------------------------|-------|--------------------------------|
| FP16 | 0.0495 Wh | 0.0227 Wh | -0.0268 Wh | 14 / 11 / 0 |
| INT4 | 0.0675 Wh | 0.0332 Wh | -0.0343 Wh | 13 / 12 / 0 |

**Interpretation:** The 20 additional features show consistent improvement in LOPO-CV MAE over the max_tokens-only baseline. However:
- Negative energy predictions observed (FP16: 0/25, INT4: 1/25 for ElasticNet).
- No prompt family overlap in pilot (0/25), but not fully controlled for full dev set.
- n=25 is too small to claim validated generalization.

---

## 3. Phase G Pilot vs Phase F.2 Workload Difference

| Dimension | Phase G Pilot | Phase F.2 (B2 Profile) |
|-----------|---------------|------------------------|
| Max tokens | Per-prompt (50–2000), median 1000 | Fixed 128 |
| Median output tokens | 1,000 | 128 |
| Median latency (FP16) | 46,599 ms | 5,546 ms |
| Median latency (INT4) | 86,445 ms | 10,628 ms |
| Mean energy (FP16) | 0.430 Wh | 0.048 Wh |
| Mean energy (INT4) | 0.466 Wh | 0.048 Wh |
| Energy order | FP16 < INT4 | INT4 < FP16 (B2 profile) |
| CarbonGrid routing | N/A (no routing in pilot) | 52% INT4, 48% FP16 |
| Measurement protocol | Same provider, NVML 20ms | Same provider, NVML 20ms |

**Root Cause of Energy Discrepancy:** Phase G pilot used 8–9× more output tokens (median 1000 vs 128), leading to proportionally higher absolute energy. The measurement protocol (NVML, sync, warm-up, unload) is consistent; only the workload differs.

**Critical Implication:** Phase F.2's B2 profile (INT4 lower energy) is a *different workload regime* from Phase G pilot (FP16 lower energy). Results **do not transfer** between them. Phase F.2 energy savings (~0.0014 Wh/prompt) are specific to short 128-token generations where INT4 profile shows lower energy. Phase G pilot shows INT4 energy *exceeds* FP16 for long generations.

---

## 4. Why the Energy Predictor Is Exploratory, Not Validated

| Limitation | Impact |
|------------|--------|
| **n=25 pilot** | LOPO-CV variance extremely high; single prompt can swing MAE by >0.01 Wh. |
| **No held-out test evaluation** | The 60 locked test prompts in `prompts.json` were **never used** for predictor evaluation. All results are LOPO-CV on the same 25 pilot prompts. |
| **Prompt family overlap not controlled** | Pilot selection (0 duplicates) does not guarantee full 340 dev set is free of near-duplicates. |
| **Negative predictions** | ElasticNet produced physically impossible negative energy values (INT4: 1/25). |
| **Workload-specific** | Pilot used long generations (1000 tokens median); Phase F.2 uses 128 tokens. Predictor trained on pilot would not generalize to F.2 workload. |
| **No external validation** | No independent dataset, no temporal holdout, no hardware/device variation. |
| **R² ≠ routing utility** | High R² (0.99) on pilot does not mean predictions are accurate enough for INT4/FP16 routing decisions. |
| **Baseline is strong** | `energy ≈ a × max_tokens_requested + b` alone explains most variance (R² > 0.97). |

**Conclusion:** The predictor shows *feasibility signal* on a tiny, long-generation pilot, but has **not been validated** on any held-out data, any other workload length, or any routing task.

---

## 5. What Further Data & Evaluation Would Be Needed

| Requirement | Description |
|-------------|-------------|
| **Larger evaluation set** | Minimum 100–200 prompts with measured energy, held out from training. |
| **Workload-matched data** | Training and evaluation must match target deployment (e.g., 128-token for F.2-style routing). |
| **External test set** | The locked 60 prompts in `prompts.json` should be measured and used *once* for final validation. |
| **Negative prediction guard** | Models must be constrained to non-negative outputs (e.g., log-link, clipping, quantile regression). |
| **Routing-relevant metric** | Evaluate MAE at decision boundary, not aggregate MAE. Cost-sensitive loss for routing. |
| **Multi-seed stability** | Report variance across multiple random seeds / CV folds. |
| **Prompt family control** | Explicit de-duplication by template family across train/test splits. |
| **Cross-workload test** | Demonstrate predictor trained on one workload (e.g., long gen) fails on another (short gen). |

---

## 6. Verification Commands

```bash
# 1. Pilot results summary
python -c "
import json, numpy as np
with open('data/evaluation/phase_g/pilot_results.json') as f: p=json.load(f)
s=[r for r in p['runs'] if r['success']]
f16=[r for r in s if r['configuration']=='fp16']
i4 =[r for r in s if r['configuration']=='int4']
print(f'FP16: mean energy={np.mean([r[\"energy_wh\"] for r in f16]):.6f} Wh')
print(f'INT4: mean energy={np.mean([r[\"energy_wh\"] for r in i4]):.6f} Wh')
print(f'FP16 median latency={np.median([r[\"latency_ms\"] for r in f16]):.0f} ms')
print(f'INT4 median latency={np.median([r[\"latency_ms\"] for r in i4]):.0f} ms')
print(f'Median tokens: FP16={np.median([r[\"output_tokens\"] for r in f16])}, INT4={np.median([r[\"output_tokens\"] for r in i4])}')
# Per-prompt INT4 < FP16
pids=set(r['prompt_id'] for r in s)
lower=sum(1 for pid in pids if np.mean([r[\"energy_wh\"] for r in i4 if r[\"prompt_id\"]==pid]) < np.mean([r[\"energy_wh\"] for r in f16 if r[\"prompt_id\"]==pid]))
print(f'INT4 lower energy: {lower}/{len(pids)} prompts')
"

# 2. Feasibility report
python -c "
import json
with open('data/evaluation/phase_g/phase_g_feasibility_report.json') as f: d=json.load(f)
print(f'FP16: baseline MAE={d[\"summary\"][\"fp16\"][\"baseline_mae\"]:.6f}, best={d[\"summary\"][\"fp16\"][\"best_model\"]} MAE={d[\"summary\"][\"fp16\"][\"best_mae\"]:.6f}')
print(f'INT4: baseline MAE={d[\"summary\"][\"int4\"][\"baseline_mae\"]:.6f}, best={d[\"summary\"][\"int4\"][\"best_model\"]} MAE={d[\"summary\"][\"int4\"][\"best_mae\"]:.6f}')
"

# 3. Incremental value
python -c "
import json
with open('data/evaluation/phase_g/phase_g_incremental_value_report.json') as f: d=json.load(f)
for c in ['fp16','int4']:
    b=d['results'][c]['LinearReg_maxTokens']['mae']
    e=d['results'][c]['ElasticNet_full']['mae']
    p=d['paired_comparison'][c]['ElasticNet_full']
    print(f'{c.upper()}: base={b:.6f} ext={e:.6f} diff={e-b:+.6f} better={p[\"n_better\"]} worse={p[\"n_worse\"]}')
"

# 4. Locked test set untouched
python -c "
import json
with open('data/evaluation/phase_g/prompts.json') as f: d=json.load(f)
print(f'Dev: {d[\"metadata\"][\"dev_count\"]}, Test: {d[\"metadata\"][\"test_count\"]}')
print('Test set locked — never used for predictor evaluation.')
"
```

---

## 7. Completion Status

**PHASE G CLOSED AS EXPLORATORY STUDY**

### What Was Established
- Pre-inference text features (especially `max_tokens_requested`) show strong correlation with measured energy on a 25-prompt long-generation pilot.
- ElasticNet with 21 features reduces LOPO-CV MAE vs max_tokens-only baseline on the same 25 prompts.
- Both FP16 and INT4 energy prediction beat a fold-mean baseline in LOPO-CV.

### What Was NOT Established
- **No validation on held-out data** — the 60 locked test prompts were never measured or evaluated.
- **No generalization to short generations** — pilot used 1000-token median; F.2 uses 128 tokens.
- **No routing utility proven** — MAE improvements do not translate to better INT4/FP16 routing decisions.
- **No negative-prediction fix** — ElasticNet produces impossible negative energy values.
- **No cross-workload stability** — predictor trained on long generations is expected to fail on short ones (and vice versa).

### Recommendation
**Do not deploy the energy predictor for routing.** The Phase G pilot demonstrates *feasibility signal* only. If energy prediction is needed for routing, a dedicated validation campaign with 100+ prompts at the target workload length (128 tokens for F.2-style, or longer for G-style) is required, with the locked 60-test-prompt set measured and evaluated exactly once.

---

## 8. Report Path

`data/evaluation/phase_g/PHASE_G_CLOSURE_REPORT.md`

No files modified. No new artifacts created beyond this report. All frozen Phase F/F.2/G artifacts preserved.