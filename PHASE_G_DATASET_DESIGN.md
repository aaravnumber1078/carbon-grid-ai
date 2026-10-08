# PHASE G DATASET DESIGN: ADAPTIVE ENERGY PROFILING

## Executive Summary

This document specifies the design for Phase G.0: a new energy-profiling dataset to replace the static Phase 0.5 energy profile with a request-level energy prediction system. The goal is to collect **400 NEW independent prompt measurements** across diverse workloads to train a request-level energy prediction model that can eventually replace the static Phase 0.5 energy profile.

**Key Principle:** Do not modify CarbonGrid, Phase F, or any existing artifacts. This is a fresh dataset design document.

**Critical Design Principle:** The dataset will not impose a target energy-winner distribution. The sign and magnitude of INT4_energy - FP16_energy will be measured empirically. Do NOT fabricate or synthetically create positive energy differences. If the new dataset again happens to have one configuration consistently lower energy, that is itself a result.

---

## 1. Dataset Scope & Scale

### 1.1 Target Size
- **Total prompts:** 400
- **Rationale:** 400 prompts × 2 configurations (FP16/INT4) = 800 inference runs. With 3 repetitions per configuration for noise estimation: 2,400 inference runs. Feasible on RTX 2050 4GB within 2–3 days.

### 1.2 Comparison to Phase F.1
| Aspect | Phase F.1 (B2) | Phase G.0 |
|--------|----------------|-----------|
| Samples | 182 | 400 |
| Sign variation | 0% positive (all INT4 < FP16) | **No target — empirically measured** |
| Energy diff range | -0.025 to -0.001 Wh | Empirically determined |
| Source | B2 only | Fresh measurements |

---

## 2. Prompt Design & Distribution

### 2.1 Prompt Categories (for Dataset Design & Post-Hoc Analysis Only)

Categories are used for **dataset design, diversity assurance, stratification, and post-hoc analysis**. They are **NOT** automatic predictor features.

| Category | Target Count | % of Total | Rationale |
|----------|-------------|------------|-----------|
| **Factual** | 40 | 10% | Short prompts, low energy variance |
| **Extraction** | 40 | 10% | Structured output, medium energy |
| **Explanation** | 40 | 10% | Medium complexity, reasoning |
| **Reasoning** | 40 | 10% | Long chains, high energy |
| **Summarization** | 40 | 10% | Output-length dependent |
| **Coding** | 40 | 10% | High output tokens, structural |
| **Scientific** | 40 | 10% | Terminology-heavy |
| **Creative** | 40 | 10% | Open-ended, high variance |
| **Instruction-heavy** | 40 | 10% | Constraint-heavy prompts |
| **Multi-turn** | 40 | 10% | Multi-message context |

**Total: 400 prompts**

### 2.2 Input/Prompt Length Distribution (Explicitly Controlled)

| Prompt Length Band | Char Range | Token Estimate | Target Count |
|-------------------|------------|----------------|--------------|
| Very short | < 100 chars | < 25 tokens | 40 |
| Short | 100–300 chars | 25–75 tokens | 80 |
| Medium | 300–800 chars | 75–200 tokens | 120 |
| Long | 800–2000 chars | 200–500 tokens | 100 |
| Very long | > 2000 chars | > 500 tokens | 60 |

### 2.3 Requested Output Length Distribution (Explicitly Controlled)

| Requested Output Band | Token Range | Target Count | Example Hints |
|----------------------|-------------|--------------|---------------|
| Very short | < 50 tokens | 40 | "in 1 sentence", "briefly" |
| Short | 50–150 tokens | 80 | "in 2–3 sentences", "concise" |
| Medium | 150–400 tokens | 120 | "in a paragraph", "detailed" |
| Long | 400–1000 tokens | 100 | "comprehensive", "thorough" |
| Very long | > 1000 tokens | 60 | "exhaustive", "comprehensive" |

### 2.4 Length Combination Matrix (Ensures Workload Variation)

| Input Length \ Output Length | Very Short | Short | Medium | Long | Very Long |
|------------------------------|------------|-------|--------|------|-----------|
| **Very Short** | 8 | 8 | 8 | 8 | 8 |
| **Short** | 8 | 16 | 16 | 16 | 8 |
| **Medium** | 8 | 16 | 24 | 24 | 16 |
| **Long** | 8 | 16 | 24 | 24 | 8 |
| **Very Long** | 4 | 8 | 16 | 16 | 8 |

Each cell shows target prompt count. Total = 400.

### 2.5 Multi-Turn Prompts

Multi-turn prompts are represented as **serialized conversation history** passed as a single model input (e.g., using chat template format). The input token count captures the additional conversation context. No separate inference interface is created.

---

## 3. Measurement Protocol

### 3.1 Hardware & Environment
- **GPU:** NVIDIA RTX 2050 4GB (same as Phase 0.5, Phase F)
- **Model:** Qwen2.5-1.5B-Instruct (same as Phase 0.5, Phase F)
- **Software:** PyTorch, bitsandbytes for INT4, transformers
- **OS:** Windows 10/11 or Linux (consistent across runs)

### 3.2 Warm-Up Protocol (Model-Loading Energy Excluded)
1. **Model load:** Load FP16 and INT4 models once at session start (not measured)
2. **Cache warm-up:** 3 FP16 + 3 INT4 inferences on a standard neutral prompt (discarded)
3. **Thermal stabilization:** Wait 30s between configurations for thermal equilibrium
4. **GPU synchronization:** `torch.cuda.synchronize()` before/after each timed region
5. **NVML sampling:** Start ~20ms before generation, stop ~20ms after

### 3.3 Measurement Repetitions
| Configuration | Repetitions | Purpose |
|---------------|-------------|---------|
| FP16 | 3 | Quantify measurement noise |
| INT4 | 3 | Quantify measurement noise |
| **Total per prompt** | **6** | |

Total inference runs: 400 prompts × 6 = 2,400 runs
Estimated time: ~2–3 days continuous on RTX 2050

### 3.4 Configuration Ordering & Randomization
- **Order:** Randomized per prompt (coin flip: FP16 first vs INT4 first)
- **Rationale:** Eliminate systematic bias from thermal state, memory fragmentation
- **Implementation:** `random.choice([FP16, INT4])` for first config per prompt

### 3.5 GPU Telemetry Sampling
- **Interval:** 20 ms (same as Phase 0.5)
- **Method:** NVML (pynvml) power sampling during generation
- **Metrics recorded per run:**
  - Average power (W)
  - Energy (Wh)
  - Peak VRAM (MB)
  - Latency (ms)
  - Tokens/second
  - Input/output token counts

### 3.6 Outlier Handling
| Condition | Action |
|-----------|--------|
| Generation failure / timeout | Mark as failed, retry once, then flag |
| Energy > 3× median | Flag as outlier, investigate, document |
| Latency > 3× median | Flag as outlier, investigate, document |
| Power readings missing | Mark measurement_source="estimated" |

**Do NOT silently delete measurements.** All raw runs are preserved; flags are metadata.

### 3.7 Per-Prompt Aggregation (From 3 Repetitions)
For each configuration (FP16, INT4) per prompt:
- Mean, median, std, CV of energy
- Mean, median, std, CV of latency
- Mean, median, std, CV of power
- Outlier flags retained

### 3.8 Measurement Uncertainty Analysis (Pre-Registered)
Because energy differences may be small, the 3 repetitions quantify measurement noise:
- For each prompt/config: `energy_cv = std_energy / mean_energy`
- **Flag:** If CV > 10%, mark as high measurement uncertainty
- **Key question:** Is |FP16_energy - INT4_energy| meaningfully larger than measurement variability?
- Report: `energy_wh ± std_wh`, `latency_ms ± std_ms`
- Distinguish explicitly: measurement noise vs. true request-level variation vs. model prediction error

---

## 4. Data Schema

### 4.1 Decision-Time Features (Available BEFORE Inference)
These are the ONLY features the predictor may use.

```json
{
  "prompt_id": "unique_id",
  "prompt_text": "full prompt text (serialized for multi-turn)",
  
  // Length features (computed from prompt_text)
  "char_length": 150,
  "word_count": 25,
  "token_count_estimate": 35,
  "sentence_count": 3,
  "question_count": 1,
  
  // Content indicators (heuristic, computed from prompt_text)
  "has_question_mark": true,
  "has_code_indicator": false,
  "has_scientific_terms": true,
  "scientific_term_count": 2,
  "has_reasoning_indicators": false,
  "reasoning_indicator_count": 0,
  "has_code_keywords": false,
  "code_keyword_count": 0,
  
  // Lexical features
  "avg_word_length": 4.2,
  "unique_word_ratio": 0.85,
  "has_numbers": true,
  "number_count": 2,
  
  // Constraint/structure features
  "requested_length_hint": 150,
  "max_tokens_requested": 200,
  "instruction_count": 2,
  "constraint_count": 1,
  
  // Metadata for dataset design/analysis ONLY (NOT predictor features)
  "category": "factual",
  "input_length_band": "short",
  "output_length_band": "medium",
  "is_multi_turn": false,
  "conversation_turns": 1
}
```

### 4.2 Post-Hoc Telemetry (Available ONLY AFTER Inference)
These are NEVER used as predictor features. Used for target creation, analysis, uncertainty estimation.

```json
{
  "run_id": "uuid",
  "prompt_id": "prompt_001",
  "configuration": "FP16",
  "run_number": 1,
  "latency_ms": 5842.3,
  "input_tokens": 35,
  "output_tokens": 128,
  "tokens_per_second": 21.9,
  "peak_gpu_memory_mb": 2960,
  "avg_power_w": 30.45,
  "energy_wh": 0.0493,
  "success": true,
  "error": null,
  "measurement_source": "measured",
  "temperature_c": 23.5,
  "gpu_clock_mhz": 1402
}
```

### 4.3 Aggregated Prompt Summary (Post-Processing)
```json
{
  "prompt_id": "prompt_001",
  "fp16": {
    "latency_ms": {"mean": 5867, "std": 45, "median": 5845, "n": 3},
    "energy_wh": {"mean": 0.0488, "std": 0.0004, "median": 0.0487, "n": 3},
    "power_w": {"mean": 30.45, "std": 1.2, "median": 30.4, "n": 3},
    "tokens_per_second": {"mean": 21.9, "std": 1.2, "median": 21.8, "n": 3},
    "peak_gpu_memory_mb": 2960
  },
  "int4": {
    "latency_ms": {"mean": 10580, "std": 120, "median": 10550, "n": 3},
    "energy_wh": {"mean": 0.0493, "std": 0.0005, "median": 0.0492, "n": 3},
    "power_w": {"mean": 17.12, "std": 0.8, "median": 17.1, "n": 3},
    "tokens_per_second": {"mean": 12.1, "std": 0.5, "median": 12.1, "n": 3},
    "peak_gpu_memory_mb": 1161
  },
  "energy_difference_wh": -0.0005,
  "energy_ratio": 1.010,
  "percentage_difference": 1.0,
  "latency_ratio": 1.80,
  "measurement_uncertainty": {
    "fp16_energy_cv": 0.008,
    "int4_energy_cv": 0.010,
    "energy_diff_magnitude_vs_noise": 1.25
  }
}
```

---

## 5. Prompt Generation Strategy

### 5.1 Sampling Strategy
1. **Seed pool:** Use existing B2 prompts as inspiration templates only
2. **Variation generation:** LLM-assisted prompt rewriting (temperature 0.7) with explicit length/category constraints
3. **Length control:** Explicit token/character targets in prompt templates (enforced by the matrix in 2.4)
4. **Category balance:** Stratified sampling across 10 categories (40 each)
5. **Deduplication:** Embedding-based deduplication (cosine similarity < 0.85)
6. **Template leakage prevention:** If templates are used, ensure template variants cannot leak structure into both train and test (see §8.3)

### 5.2 Prompt Generation Pipeline
```
1. Template selection (category + input_length_band + output_length_band)
2. LLM variation generation (temperature 0.7) with length constraints
3. Heuristic validation (length, category match, token count)
4. Embedding deduplication (cosine < 0.85)
5. Manual spot-check (5% sample)
6. Final curation → locked prompt list
7. Test set split (see §8)
```

---

## 6. Target Variables

The dataset records BOTH absolute energies. Derived targets are computed post-hoc. The modeling experiment will compare all representations.

| Variable | Type | Description |
|----------|------|-------------|
| `fp16_energy_wh` | Measured (target) | Mean of 3 FP16 repetitions |
| `int4_energy_wh` | Measured (target) | Mean of 3 INT4 repetitions |
| `energy_difference_wh` | Derived | INT4 - FP16 (can be positive or negative) |
| `energy_ratio` | Derived | INT4 / FP16 |
| `percentage_difference` | Derived | (INT4 - FP16) / FP16 × 100 |
| `fp16_latency_ms` | Measured | Mean of 3 FP16 repetitions |
| `int4_latency_ms` | Measured | Mean of 3 INT4 repetitions |

**Do NOT assume beforehand which target representation is best.** The later modeling experiment can compare:
- A. FP16 energy prediction
- B. INT4 energy prediction
- C. Energy-difference prediction
- D. Energy-ratio prediction

---

## 7. Train/Validation/Test Protocol (Fixed Before Data Collection)

### 7.1 Split Strategy
| Split | Size | Purpose |
|-------|------|---------|
| **Locked Test** | 60 (15%) | **Final unbiased evaluation — evaluated ONCE** |
| **Development** | 340 (85%) | Model development, nested CV, feature selection |

**Stratification:** By `(category, input_length_band, output_length_band)` to ensure representation across the matrix.

### 7.2 Nested Cross-Validation (ONLY on Development Set)
- **Outer CV:** 5-fold on 340 development samples
- **Inner CV:** 5-fold for hyperparameter tuning within each outer fold
- **Test set:** Completely locked, never touched during development

### 7.3 No Data Leakage Rules
- Feature engineering (scaling, etc.) fit ONLY on training folds within each CV iteration
- Test set completely held out until final evaluation
- No Phase F / Phase B2 data in training/validation
- Test prompts isolated BEFORE any model development begins

### 7.4 Test Set Integrity
- **Split methodology:** Random stratified split with fixed seed (documented)
- **Template leakage prevention:** If prompts share templates, all variants of a template go to the SAME split
- **Near-duplicate check:** Embedding similarity between train and test < 0.85 threshold enforced
- **Split documentation:** Record split seed, stratification columns, and leakage checks

---

## 8. Predictor Features (Locked)

### 8.1 Decision-Time Features (Runtime-Available)
These 20 features can genuinely be computed from a new user request before inference:

| Feature | Type | Computation |
|---------|------|-------------|
| `char_length` | float | len(prompt_text) |
| `word_count` | float | len(prompt_text.split()) |
| `token_count_estimate` | float | word_count × 1.3 (or tokenizer estimate) |
| `sentence_count` | float | count of . ! ? |
| `question_count` | float | count of ? |
| `has_question_mark` | bool | '?' in prompt_text |
| `has_code_indicator` | bool | heuristic: code fences, keywords |
| `has_scientific_terms` | bool | heuristic: scientific vocabulary list |
| `scientific_term_count` | float | count of matched terms |
| `has_reasoning_indicators` | bool | heuristic: "because", "therefore", etc. |
| `reasoning_indicator_count` | float | count of matched terms |
| `has_code_keywords` | bool | heuristic: def, class, function, etc. |
| `code_keyword_count` | float | count of matched keywords |
| `avg_word_length` | float | mean(len(w)) for w in words |
| `unique_word_ratio` | float | unique_words / total_words |
| `has_numbers` | bool | any digit in prompt_text |
| `number_count` | float | count of numeric tokens |
| `requested_length_hint` | float | parsed from "in N sentences", "briefly", etc. |
| `max_tokens_requested` | float | explicit max_tokens if provided, else default |
| `instruction_count` | float | count of imperative sentences |
| `constraint_count` | float | count of explicit constraints |

**TOTAL: 21 features** (all numeric/boolean, no categorical one-hot)

### 8.2 Features EXCLUDED (Post-Hoc / Metadata)
- `category` / `task_type` — dataset design only
- `input_length_band` / `output_length_band` — design only
- `is_multi_turn` / `conversation_turns` — design only
- Actual generated output tokens — post-hoc
- Actual latency / energy / power — post-hoc
- Peak VRAM — post-hoc

---

## 9. Evaluation Metrics (Pre-Registered)

### 9.1 Primary Regression Metrics (Evaluated on Locked Test Set)
| Metric | FP16 Target | INT4 Target | Difference Target |
|--------|-------------|-------------|-------------------|
| MAE (Wh) | < 0.003 Wh | < 0.003 Wh | < 0.003 Wh |
| RMSE (Wh) | < 0.005 Wh | < 0.005 Wh | < 0.005 Wh |
| R² | > 0.5 | > 0.5 | > 0.5 |
| Pearson r | > 0.7 | > 0.7 | > 0.7 |
| Spearman ρ | > 0.7 | > 0.7 | > 0.7 |

### 9.2 Secondary Metrics
- Error distribution (histogram, Q-Q plot)
- Calibration plots (predicted vs actual)
- Per-category / per-length-band breakdown (analysis only)

### 9.3 Decision-Level Metric (Conditional)
**Configuration-selection accuracy** (which config has lower predicted energy) is ONLY reported if the test set contains genuine variation in which configuration has lower *measured* energy.

If all test prompts have the same lower-energy configuration:
- Explicitly state: "Directional accuracy is not informative — test set lacks sign variation"
- Do NOT present it as evidence of routing intelligence

### 9.4 Baseline Comparisons (Pre-Registered)

**Important distinction:**

- **Phase 0.5 measured static energy values** (used by CarbonGrid for environmental decisions):
  - FP16 = 0.0488 Wh
  - INT4 = 0.0493 Wh
  - These are single-point estimates from Phase 0.5 benchmarking.

- **Phase F.1 / B2 static-prediction MAE** (~0.012 Wh):
  - This is the error of predicting *every prompt* with those constant values, evaluated against the B2 measured per-prompt energies.
  - It reflects the mismatch between the static profile and actual per-prompt variation.

**For Phase G, the static-profile baseline is the constant-energy predictor using the Phase 0.5 values:**

| Baseline | Description |
|----------|-------------|
| **Static Phase 0.5 (Phase G baseline)** | Constant prediction: FP16=0.0488 Wh, INT4=0.0493 Wh for all prompts |
| Global mean | Mean FP16 / INT4 energy across dev set |
| Per-category mean | Mean per category (if category available at runtime — it is NOT) |
| Heuristic | `energy ∝ output_tokens × complexity_proxy` |

**Success requires:** The predictor must materially improve over the **Static Phase 0.5 constant predictor** on the locked test set.

---

## 10. Pre-Registered Success Criteria

The following criteria are defined BEFORE data collection and will not be adjusted after seeing results.

**Reference baseline:** Static Phase 0.5 constant predictor (FP16=0.0488 Wh, INT4=0.0493 Wh). Its MAE on the locked test set will be computed as the reference.

| Criterion | Threshold | Evaluation |
|-----------|-----------|------------|
| **FP16 MAE** | < 0.003 Wh | Locked test set |
| **INT4 MAE** | < 0.003 Wh | Locked test set |
| **FP16 MAE vs Static** | Predictor MAE < Static Phase 0.5 MAE | Locked test set |
| **INT4 MAE vs Static** | Predictor MAE < Static Phase 0.5 MAE | Locked test set |
| **R² (both)** | > 0.5 | Locked test set |
| **Calibration** | Predicted vs actual slope ≈ 1 | Locked test set |

**If criteria are not met:**
- Report the negative result
- Do NOT force integration into CarbonGrid
- Document what the data reveals about predictability

---

## 11. Future Extension: Uncertainty-Aware Routing (NOT Implemented Now)

**Planned for future Phase G.x only.** Do NOT implement now. Do NOT modify existing Decision Engine.

If predicted FP16 and INT4 energy intervals substantially overlap (e.g., |pred_diff| < k × pred_uncertainty), a future CarbonGrid version could:
- Keep safety gate mandatory
- Keep latency constraint mandatory
- If predicted environmental difference is smaller than prediction uncertainty, do not treat the difference as decisive
- Fall back to conservative default (e.g., FP16 for quality, or user preference)

---

## 12. Phase G Timeline (Conceptual)

| Phase | Description | Status |
|-------|-------------|--------|
| **G.0** | Design (this doc) | **CURRENT — UNDER REVIEW** |
| **G.1.1** | Prompt dataset generation | PENDING APPROVAL |
| **G.1.2** | Measurement collection | PENDING |
| **G.1.3** | Predictor development/evaluation | PENDING |
| **G.1.4** | Analysis/report | PENDING |

**No implementation begins until this design is approved.**

---

## 13. Data Quality & Validation Gates

| Check | Threshold | Action |
|-------|-----------|--------|
| Measurement CV (energy) | > 10% | Flag as high uncertainty, document |
| Failed generation | > 0 per prompt | Retry once, then flag permanently |
| Energy outlier | > 3× IQR | Flag, review, document (do not delete) |
| Latency outlier | > 3× IQR | Flag, review, document |
| Missing telemetry | Any | Mark measurement_source="estimated" |

### 13.1 Integrity Checks (Post-Collection)
- [ ] All 400 prompts have 3 FP16 + 3 INT4 runs
- [ ] No missing measurements
- [ ] Energy values physically plausible (> 0, < 1 Wh)
- [ ] Latency > 0, < 120s
- [ ] Token counts consistent with prompt/response
- [ ] Test set leakage check passed
- [ ] All raw runs preserved with flags

---

## 14. Appendix: Measurement Methodology Consistency

This protocol maintains consistency with Phase 0.5 / Phase F wherever possible:
- Same GPU (RTX 2050 4GB)
- Same model (Qwen2.5-1.5B-Instruct)
- Same NVML sampling (~20ms interval)
- Same energy integration method (power × time)
- Same INT4 quantization (bitsandbytes 4-bit)
- Same FP16 precision (torch.float16)
- Model-loading energy excluded (same as Phase 0.5)
- Warm-up runs discarded (same as Phase 0.5)

---

## 15. Summary of Changes from Previous Design

| Aspect | Previous Design | Revised Design |
|--------|----------------|----------------|
| Energy-winner target | 30–50% positive | **Removed — fully empirical** |
| Predictor features | 17 numeric + 10 one-hot task_type | **21 decision-time features only** |
| Train/Val/Test | 70/15/15, CV on train+val | **15% locked test, 85% dev, nested CV on dev only** |
| Measurement protocol | High-level | **Detailed: warm-up, randomization, GPU sync, outlier handling** |
| Feature groups | Mixed | **Explicit: decision-time vs post-hoc telemetry** |
| Length control | Category balance only | **Explicit matrix: input × output length bands** |
| Multi-turn | Not specified | **Serialized conversation history** |
| Target variables | energy_diff primary | **Both absolutes + derived (diff, ratio, %)** |
| Evaluation metrics | MAE/R² + directional | **Pre-registered MAE/RMSE/R²/Pearson/Spearman + conditional directional** |
| Uncertainty analysis | Mentioned | **Pre-registered: CV, noise vs signal, explicit distinction** |
| Success criteria | Not pre-registered | **Pre-registered table with thresholds** |
| Test integrity | Basic stratification | **Template leakage prevention, embedding similarity check** |
| Future routing | Not mentioned | **Uncertainty-aware routing as documented future extension** |

---

## 16. Final Design Summary

| Item | Value |
|------|-------|
| **Total prompts** | 400 |
| **Total inference runs** | 2,400 (400 × 2 configs × 3 reps) |
| **Locked test set** | 60 prompts (15%) |
| **Development set** | 340 prompts (85%) |
| **Predictor features** | 21 decision-time features (no task_type) |
| **Target variables** | fp16_energy, int4_energy, plus derived diff/ratio/% |
| **Primary metrics** | MAE, RMSE, R², Pearson r, Spearman ρ |
| **Success criteria** | MAE < 0.003 Wh, beats static baseline, R² > 0.5 |
| **Measurement uncertainty** | Pre-registered CV analysis, noise vs signal distinction |

---

*End of Phase G.0 Dataset Design Document (Revised)*

---
*Document version: 2.1*  
*Date: 2026-09-27*  
*Status: REVISED — APPROVED*  
*Next step: Begin G.1.1 — Prompt dataset generation*