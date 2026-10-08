# Request-Level Energy Predictor: Feasibility Assessment & Experimental Protocol

## Executive Summary

**Verdict: Feasible with caveats.** The existing data provides a marginal but scientifically defensible foundation for a request-level energy predictor. The critical limitation is **data volume** (182 B2 samples with measured INT4/FP16 energy), which limits model complexity and statistical confidence. A lightweight ridge/elastic-net model with rigorous nested CV is the maximum defensible complexity.

---

## 1. Features Available at Request/Decision Time

### From B2 Dataset (182 samples)
**Prompt-derived features (22 numeric):**
- `char_length`, `word_count`, `token_count_estimate`
- `has_question_mark`, `has_code_indicator`, `question_count`, `sentence_count`
- `has_scientific_terms`, `scientific_term_count`, `has_reasoning_indicators`, `reasoning_indicator_count`
- `has_code_keywords`, `code_keyword_count`
- `avg_word_length`, `unique_word_ratio`, `has_numbers`, `number_count`
- `requested_length_hint`, `max_tokens_requested`, `instruction_count`, `constraint_count`

**TF-IDF features (100):** 100-dim sparse vector (unigrams + bigrams, English stopwords)

**Task type (categorical):** 8 categories (factual, extraction, explanation, reasoning, coding, summarization, scientific, creative)

**complexity_level:** Available in B2 but **excluded from C.1.1 classifier** (dataset-generation metadata). Could be used for energy prediction if available at inference time.

### From Phase 0.5 (12 samples × 2 configs = 24 measurements)
- **Limited to 6 prompt types** (factual, scientific, explanation, summarization, coding, reasoning)
- Only 6 unique prompts × 2 configurations
- Not representative of B2 diversity

### From Phase F (182 samples × 3 policies = 546 runs)
- All B2 prompts evaluated under Always-FP16, Always-INT4, CarbonGrid
- **Measured FP16/INT4 energy & latency** for all 182 prompts
- Decision-time static estimates (Phase 0.5 profile) stored
- **Classifier safety probability** available

---

## 2. Measured Energy/Latency Data Summary

| Source | Samples | FP16 Energy | INT4 Energy | FP16 Latency | INT4 Latency |
|--------|---------|-------------|-------------|--------------|--------------|
| Phase 0.5 | 6 prompts × 2 configs | ✓ | ✓ | ✓ | ✓ |
| B2 Dataset | 182 prompts | ✓ | ✓ | ✓ | ✓ |
| Phase F | 182 × 3 policies | ✓ | ✓ | ✓ | ✓ |

**Key finding:** B2 provides 182 independent prompt-level measurements with both FP16 and INT4 energy measured. This is the primary training data.

---

## 3. Training Data Without Leaking Phase F Test Set

**Safe training data (no leakage):**
- **B2 Dataset (182 samples):** Measured FP16/INT4 energy from independent benchmark runs. These are the *ground truth* labels.
- **Phase 0.5 (12 measurements):** Independent validation runs on different hardware run, different prompts (6 prompt types). Can be used for external validation or domain adaptation check.

**Must NOT use (leakage):**
- **Phase F CarbonGrid runs (182):** Decision-time estimates used in actual routing decisions. Using these for training would leak the decision policy into the energy model.
- **Phase F Always-FP16/Always-INT4 (182 each):** Same hardware runs as CarbonGrid; not independent.

**Recommended training set:** B2 dataset (182 samples) with measured INT4/FP16 energy as targets.

---

## 4. Data Volume Assessment

| Metric | Value | Assessment |
|--------|-------|------------|
| Samples | 182 | **Marginal** for ML; limits to very simple models |
| Features (raw) | 22 numeric + 100 TF-IDF | High-dimensional relative to n |
| Target variables | 2 (FP16 energy, INT4 energy) or 1 (difference) | OK |
| Task types | 8 categories | Sufficient for stratification |
| Complexity levels | 4 levels | Sufficient |

**Verdict:** 182 samples is **marginal but defensible** for ridge/elastic-net regression with strong regularization. Not sufficient for gradient boosting or random forest without severe overfitting risk.

---

## 5. Target Variable Recommendation

**Primary target: `energy_difference = INT4_energy - FP16_energy`**

**Rationale:**
- Directly answers the decision-relevant question: "Which config uses less energy?"
- Magnitude of difference is small (~0.002 Wh mean), so absolute energy prediction has higher relative error
- Difference target has lower variance and is more stable
- Decision engine only needs sign(difference) + magnitude for CO2 calculation

**Alternative targets (if needed):**
- `FP16_energy` and `INT4_energy` as multi-output
- `log(INT4_energy / FP16_energy)` for ratio prediction

---

## 6. Model Recommendations (Ranked by Defensibility)

| Rank | Model | Justification |
|------|-------|---------------|
| 1 | **Ridge/ElasticNet** | Strong regularization handles n<<p; interpretable coefficients; minimal hyperparameters |
| 2 | **Linear Regression + PCA** | Dimensionality reduction before regression; very few effective parameters |
| 3 | **Random Forest (max_depth≤3, min_samples_leaf≥10)** | Only if nested CV shows clear improvement over linear |
| 4 | Gradient Boosting | **Not recommended** — too flexible for n=182 |

**Primary recommendation:** **ElasticNet (α=0.5, λ via nested CV)** with standardized features.

---

## 7. Train/Validation/Test Split Protocol

**Stratified Group K-Fold (no leakage):**
- **Stratify by:** `(task_type, complexity_level)` — 8×4=32 strata
- **Outer CV:** 5-fold (test sets)
- **Inner CV:** 5-fold (hyperparameter tuning)
- **Group by:** `prompt_id` (not needed since 1 sample per prompt)

**Test set holdout:** Fixed 20% (36 samples), stratified, **fixed seed (42)**, defined *before* any analysis.

**No data augmentation.** No synthetic samples. No Phase F data in training.

---

## 8. Feature Set

### Core Features (required)
- 22 prompt-derived numeric features (standardized)
- Task type (one-hot, 8 categories)
- Complexity level (ordinal, 1-4) — *if available at inference time*

### Optional/Experimental
- TF-IDF (100-dim): **Not recommended** — 100 sparse features for n=182 is too high-dimensional; keep only if nested CV shows improvement
- Classifier safety probability: **Available at decision time**, could be informative
- Prompt embedding (SentenceTransformer): **Not recommended** — adds external dependency, high dimension

**Recommended feature set:** 22 numeric + 8 one-hot task = 30 features → ElasticNet handles this well.

---

## 9. Evaluation Metrics

| Metric | Purpose | Target |
|--------|---------|--------|
| **MAE (Wh)** | Absolute error in Wh | < 0.002 Wh (10% of mean diff) |
| **RMSE (Wh)** | Penalize large errors | < 0.003 Wh |
| **R²** | Variance explained | > 0.3 (modest) |
| **Directional Accuracy** | Sign(diff) correct | > 70% |
| **Within-10% Accuracy** | Relative error < 10% | > 50% |

**Primary metric:** MAE + Directional Accuracy (decision-relevant)

---

## 10. Integration Without Contaminating Phase F

**Architecture:**
```
┌─────────────────────────────────────────────────────────────┐
│                     DECISION ENGINE                         │
├─────────────────────────────────────────────────────────────┤
│  Safety Gate (Classifier)  │  Energy Predictor (NEW)        │
│  - Safety prob              │  - FP16 energy prediction      │
│  - Threshold check          │  - INT4 energy prediction      │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
              ┌─────────────────────────┐
              │  Energy Difference      │
              │  INT4_energy - FP16_energy │
              └─────────────────────────┘
                          │
                          ▼
         ┌────────────────┴────────────────┐
         ▼                                 ▼
   Carbon-aware routing              Latency constraint
   (if diff < 0 → INT4)               (check SLA)
```

**Integration rules:**
1. Energy predictor runs **after** safety gate passes
2. Predictor uses **same features** as classifier (no new dependencies)
3. Static profile (Phase 0.5) remains **fallback** if predictor fails
4. **No retraining** of classifier; energy predictor is independent module
5. Phase F evaluation results **unchanged** — predictor is additive

---

## Proposed Experimental Protocol

### Phase F.1: Model Development (Offline)
1. **Lock test set:** 36 samples (20%), stratified by (task_type, complexity), seed=42
2. **Train:** 146 samples, nested 5×5 CV for λ (ElasticNet α=0.5)
3. **Evaluate:** MAE, RMSE, R², Directional Accuracy on locked test set
4. **Baseline comparison:** Static Phase 0.5 profile (always predicts FP16=0.0488, INT4=0.0493)
5. **Success criterion:** MAE < 0.002 Wh AND Directional Accuracy > 70%

### Phase F.2: Ablation Study
- Features: {numeric only, +task_type, +TF-IDF, +safety_prob}
- Models: Ridge vs ElasticNet vs Linear+PCA
- Report full CV results table

### Phase F.3: Integration Test (Offline)
- Replay Phase F CarbonGrid decisions using predictor
- Compare: Static profile vs Predictor decisions
- Metrics: Energy savings, CO2 reduction, safety gate interactions

---

## Data Availability for Predictor Development

| Dataset | Role | Size | Leakage Risk |
|---------|------|------|--------------|
| B2 Measured Energy | Labels (y) | 182 | None |
| B2 Prompt Features | Features (X) | 182 | None |
| Phase 0.5 | External validation | 12 | None |
| Phase F CarbonGrid | **FORBIDDEN** | 182 | **HIGH** — decision policy |
| Phase F Baselines | **FORBIDDEN** | 364 | **HIGH** — same runs |

---

## Recommendation

**Proceed with Phase F.1 (Offline Model Development)** under these conditions:

1. **Use only B2 data (182 samples)** for training/validation
2. **ElasticNet with nested 5×5 CV** as primary model
3. **MAE + Directional Accuracy** as primary metrics
4. **Success gate:** MAE < 0.002 Wh AND Directional Accuracy > 70% on locked test set
5. **If failed:** Do not integrate; report negative result honestly
6. **No Phase F data** in any training/validation step

**Scientific honesty requirement:** If model fails to beat static profile, report negative result. Do not ship a predictor that doesn't improve on static baseline.

---

## Files to Create (If Proceeding)

```
scripts/train_energy_predictor.py      # Training script with nested CV
scripts/evaluate_energy_predictor.py   # Evaluation on locked test set
scripts/analyze_energy_predictor.py    # Analysis & plotting
tests/test_energy_predictor.py         # Unit tests
models/energy_predictor/               # Model artifacts
data/evaluation/phase_f/energy_predictor_results.json
```

---

## Conclusion

**The data barely supports this extension.** 182 samples is the minimum for a regularized linear model. The static Phase 0.5 profile is demonstrably wrong for B2 workload (reverses FP16/INT4 order), so there is signal to learn. However, the effect size is small (3.6% mean difference), and 182 samples limits model complexity.

**Recommendation:** Proceed with Phase F.1 as a **time-boxed experiment (1-2 weeks)**. If the predictor fails to beat the static profile on the locked test set, **abandon the extension** and document the negative result. Do not proceed to integration without a clear positive result.

**Do not modify CarbonGrid, classifier, or Phase F evaluation.** The energy predictor is an independent, optional module.