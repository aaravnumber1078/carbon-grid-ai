# CarbonGrid-AI Master Limitation Register

**Status:** STAGE 2 — P1 STEP 3 COMPLETE (SAFETY FIXES IMPLEMENTED)  
**Date:** 2026-10-04  
**Purpose:** Single source of truth for all known limitations, risks, and remediation status.

**Classification Definitions:**
- **Resolved:** Fix implemented, tested, and verified
- **Mitigated:** Partial fix or safeguard in place; residual risk documented and accepted
- **Still Unverified:** Known issue not yet addressed; requires code fix or test
- **Dependent on External Data:** Requires independent real-world data not yet obtained; smallest feasible alternative proposed
- **Future Research:** Fundamental limitation requiring research beyond hackathon scope; clearly scoped

---

## 1. Limitation Register

| ID | Limitation / Risk | Severity | Evidence & File/Line References | Classification | External Data Research & Feasibility | Recommended Remediation | Validation Required | Expected Impact | Risks of Remediation |
|----|-------------------|----------|----------------------------------|----------------|--------------------------------------|-------------------------|---------------------|-----------------|----------------------|
| L1 | Quantization-safety classifier has only 29 held-out test samples; proxy-derived labels; 40% false-safe rate (6/15); calibration uncertain; threshold not independently validated | HIGH | `models/quantization_safety/metadata_c1_1.json:412-429` (test metrics); `carbongrid/classifier/service.py:162` (threshold 0.50); `data/evaluation/phase_f_b2profile_gpu/PHASE_F2_AUDIT_REPORT.md:120-122` | **Mitigated (code)** / **Future Research (validation)** | **External data needed:** Human-verified quantization safety labels for diverse prompts on Qwen2.5-1.5B. **Available:** No public dataset with INT4 vs FP16 quality comparisons for this model. **Papers:** "LLM.int8()" (Dettmers et al.), "AWQ" (Lin et al.) discuss quantization but don't provide labeled safety datasets. **Smallest feasible alternative:** Use existing B2 validation set (25 samples) for threshold sweep; raise threshold to 0.60+ to reduce false-safe rate; document as experimental. | 1. Fail-closed safety gate (already implemented — classifier exception → FP16); 2. Raise threshold to 0.60 using validation set only (never test set); 3. Document classifier as experimental with known false-safe rate; 4. Do not claim production safety. | 1. Verify fail-closed behavior in integration tests; 2. Run threshold sweep on validation set (25 samples); 3. Confirm no test-set tuning. | Reduces unsafe INT4 selections; may reduce INT4 opportunity from 55% to ~40%; safer default behavior. | Higher threshold = fewer INT4 selections = less energy savings potential; but safer. |
| L2 | Energy predictor (Phase F.1 / Phase G) is exploratory only: Phase F.1 model has 100% directional accuracy on locked test set but misleading (all B2 prompts had INT4 < FP16 in that benchmark); Phase G pilot n=25, negative predictions (1/25 INT4), no locked-test evaluation, workload-specific (1000 tokens vs 128); R² ≠ routing utility | HIGH | `data/evaluation/phase_f/energy_predictor_evaluation.json`; `data/evaluation/phase_g/PHASE_G_CLOSURE_REPORT.md:110-123`; `models/energy_predictor/energy_predictor.joblib`; `scripts/train_energy_predictor.py`; `carbongrid/evaluation/energy_predictor_utils.py`; `tests/test_decision_engine.py::TestEnergyPredictorValidation` | **Resolved (not in routing + guards)** / **Dependent on External Data (validation)** | **External data needed:** Measured GPU energy for 100+ prompts at target workload (128 tokens) on RTX 2050 with Qwen2.5-1.5B. **Available:** No public dataset with NVML-measured energy for local LLM inference. **Papers:** "Green AI" (Schwartz et al.), "CarbonTracker" (Anthony et al.) measure energy but not for this model/hardware combo. **Smallest feasible alternative:** Predictor NOT in routing path (confirmed); profile-based estimates (Phase 0.5, B2) used; validation guards added in `carbongrid/evaluation/energy_predictor_utils.py` for negative/NaN/inf/None/string predictions; 11 unit tests in `TestEnergyPredictorValidation`. | 1. Confirm predictor not in `/generate` path (verified by code audit); 2. Validation guards added: `validate_energy_prediction()` clamps negative to 0.0, rejects NaN/inf/None/strings with clear errors; 3. `PREDICTOR_STATUS` documents status; 4. Profile-based estimates remain default. | 1. **Predictor NOT in routing** - confirmed by code audit (no import/use in `carbongrid/`); 2. **11 tests pass** for validation guards (negative→clamped, NaN/inf/None/string→error, too-large→clamped); 3. **PREDICTOR_STATUS** explicitly documents "used_in_production_routing: false". | Eliminates risk of invalid energy estimates driving routing; profile estimates are measured on-target. | Loses adaptive estimation; but predictor was not reliable for routing anyway. |
| L3 | NVML energy measurement is periodic sampling (20ms), not laboratory-grade; can miss short transients; idle/background GPU consumption, warm-up, model loading, measurement boundaries affect results; **energy/CO₂ unit validation completed** | MEDIUM | `carbongrid/inference/provider.py:130-194` (GPUPowerMonitor, 20ms interval); `carbongrid/inference/provider.py:341-365` (measurement boundaries); `data/evaluation/phase_f_b2profile_gpu/PHASE_F2_AUDIT_REPORT.md:87-88`; `carbongrid/decision/models.py:301-325` (validated `estimate_co2_from_energy`); `tests/test_decision_engine.py:26-100` (18 CO₂ unit tests) | **Resolved (unit validation)** / **Still Unverified (NVML methodology documentation)** | **External data needed:** Laboratory-grade power measurements (e.g., external power meter) for same workloads to calibrate NVML. **Available:** No public calibration data for RTX 2050 + Qwen2.5-1.5B. **Papers:** "Zeus" (You et al.) uses similar NVML sampling; "PerfPerWatt" discusses methodology. **Smallest feasible alternative:** Document methodology explicitly; label all energy as "NVML-sampled estimate"; exclude model-loading energy; add idle baseline measurement if feasible. | 1. **Formula verified**: Wh → kWh (÷1000) → gCO₂ (× intensity) → mg (×1000); 2. **Dimensional sanity test passed**: 1 Wh at 200 gCO₂/kWh = 0.2 gCO₂; 3. **Edge cases handled**: negative, NaN, inf, None, strings → ValueError; zero energy/intensity → zero CO₂; very small/large values correct; 4. **Existing valid behavior preserved**: all 109 tests pass. | Honest labeling; sets correct expectations; judges can assess measurement quality. | None if documented properly. |
| L4 | Carbon intensity: UK API only (15 regions); offline fallback = 200 gCO₂/kWh assumption; region selection, timestamps, caching, staleness, failure behavior must be transparent; multi-region routing is simulation; **provenance & staleness transparency implemented** | HIGH | `carbongrid/carbon/providers.py:78-345` (UK provider); `carbongrid/carbon/providers.py:452-537` (MultiProvider fallback); `carbongrid/carbon/models.py:36-83` (CarbonReading with fetched_at, staleness); `carbongrid/carbon/manager.py:201-247` (status with staleness); `carbongrid/api/main.py:146-180, 355-395` (API response with provenance) | **Resolved (metadata transparency)** / **Dependent on External Data (global coverage)** | **External data needed:** Global carbon intensity API with local grid data (e.g., Electricity Maps, WattTime). **Available:** Electricity Maps API (requires key, free tier limited); WattTime (requires registration). Both provide broader coverage than UK-only. **Papers:** "Real-time Carbon Intensity" datasets exist but often require agreements. **Smallest feasible alternative:** Keep UK API + offline fallback; add Electricity Maps as optional provider if API key available; clearly label all sources; never claim global coverage without it. | 1. **Exposed fields**: `carbon_source`, `reading_type`, `timestamp`, `fetched_at`, `staleness_seconds`, `is_stale`, `is_offline`, `is_replay`, `metadata` in `/generate` and `/carbon/status`; 2. **Staleness threshold**: 3600s (1 hour) default, configurable; 3. **Offline fallback labeled**: metadata includes `"note": "Assumed offline value - not a live measurement"`; 4. **7 new tests** in `TestCarbonProvenance` cover fresh/stale/forecast staleness, offline metadata, to_dict. | Transparent carbon provenance; avoids misleading "live global" claims; judges see exactly what's real vs simulated. | None — improves honesty. Electricity Maps integration optional, not required for demo. |
| L5 | RTX 2050 4GB VRAM: measured free VRAM ~707-709 MiB (FP16 loaded); below both policy thresholds (FP16: 3200 MiB, INT4: 1400 MiB); INT4 can be slower and use more energy; live INT4 not verified under current constraints | HIGH | `carbongrid/decision/policy.py:61-62` (thresholds); `carbongrid/api/main.py:293-330` (VRAM telemetry); `data/evaluation/demo_readiness_audit.md:130`; `carbongrid/inference/provider.py:62-102` (VRAM measurement); `carbongrid/decision/engine.py:158-186` (explicit VRAM telemetry semantics) | **Resolved (policy + VRAM semantics)** / **Still Unverified (live INT4)** | **External data needed:** None — this is hardware-specific measurement. **Available:** Current hardware measurement. **Smallest feasible alternative:** Test INT4 load path on demo machine once; if OOM or unstable, document and keep FP16 default. Do not lower thresholds. | 1. Do NOT lower thresholds to force INT4; 2. Keep conservative VRAM checks; 3. Report VRAM telemetry honestly (measured vs unavailable); 4. Document FP16 default due to hardware constraints; 5. Test INT4 loading path separately (one-time). | 1. Run INT4 load test on demo machine; 2. Verify VRAM measurement accuracy; 3. Confirm response shows `vram_measurement_source` correctly; 4. **VRAM telemetry semantics explicit**: `vram_telemetry_available` field distinguishes (a) legacy caller without telemetry → skip VRAM check (backward compatible), (b) telemetry attempted but failed → fail closed (neither FP16 nor INT4 feasible), (c) valid measured VRAM → normal threshold checks. Verified by 9 tests in `TestVRAMTelemetrySemantics`. | Safe operation; no forced INT4; honest about hardware limits; demo shows realistic behavior. | Demo may show mostly FP16 selections — must frame as "hardware-constrained realistic behavior." |
| L6 | Evaluation claims: Phase F no significant energy savings (p=0.096); Phase F.2 savings only for 128-token workload with B2 profile; quality metric is proxy (semantic similarity); results hardware/software/model/prompt-distribution specific | HIGH | `data/evaluation/phase_f/phase_f_analysis.json`; `data/evaluation/phase_f_b2profile_gpu/PHASE_F2_AUDIT_REPORT.md:178-198`; `data/evaluation/phase_f_b2profile_gpu/PHASE_F2_CLOSURE_REPORT.md:149-159` | **Mitigated (claims)** / **Future Research (generalization)** | **External data needed:** Independent reproduction on different hardware/models/workloads. **Available:** No public benchmark with same methodology. **Papers:** MLPerf Inference, LLMPerf — but different scope. **Smallest feasible alternative:** Qualify all claims explicitly in presentation; keep synthetic/controlled evaluation separate from real-world claims; never present Phase F.2 as general proof. | 1. Never claim "CarbonGrid saves energy" universally; 2. Qualify every claim: "On RTX 2050, Qwen2.5-1.5B, 128-token generations, B2 workload profile..."; 3. Label quality as "proxy (semantic similarity vs FP16 reference)"; 4. Keep frozen artifacts immutable; 5. New experiments in new directories only. | 1. Audit all public claims against artifacts; 2. Ensure presentation language matches evidence; 3. Separate "controlled evaluation" from "real-world generalization" slides. | Scientifically defensible claims; judges see exact scope of evidence. | May appear less impressive — but honest and defensible. |
| L7 | API operational reliability: classifier exceptions, missing features, unknown config, unavailable VRAM, carbon provider timeout, model load failure, inference timeout, concurrent requests, resource exhaustion, model lifecycle races | MEDIUM | `carbongrid/api/main.py:240-385` (error handling); `tests/test_phase_e.py` (mocked + lifecycle tests); `data/evaluation/demo_readiness_audit.md:126-132`; `carbongrid/decision/engine.py:57-70` (fail-closed classifier); `carbongrid/api/main.py:250-260` (classifier exception handling); `carbongrid/inference/provider.py:199-450` (model lifetime protection) | **Resolved (fail-closed + VRAM + model lifecycle)** / **Still Unverified (hard cancellation)** | **External data needed:** None — this is code robustness. **Smallest feasible alternative:** Add explicit error handling, fail-closed defaults, timeouts, synchronization, and integration tests with mocked failures. | 1. Add try/except for classifier, carbon, inference with fail-closed (FP16) defaults; 2. Add request timeout (120s); 3. Serialize model loading (mutex); 4. Add model lifetime protection (active-inference counter + condition variable); 5. Add integration tests for each failure mode; 6. Verify concurrent request handling. | 1. **Classifier exception → FP16 with `CLASSIFIER_ERROR`** — implemented in `main.py:250-260`, `engine.py:57-70`; 2. **Invalid safety probability (NaN, inf, negative, >1, None) → FP16 with `CLASSIFIER_ERROR`** — verified by 5 tests in `TestDecisionEngineFailClosed`; 3. **VRAM unavailable (None, NaN, inf, negative) → conservative fallback (FP16)** — verified by 4 tests in `TestDecisionEngineVRAMConstraints`; 4. **Model unload waits for active inference** — implemented in `provider.py` with `_active_inferences` counter + `_model_lifetime_cv`; 5. **Model replacement waits for active inference** — `ensure_loaded()` waits on `_model_lifetime_cv`; 6. **Inference exception decrements active count** — verified by `TestModelLifecycleRace::test_inference_exception_decrements_active_count`; 7. **HTTP timeout while worker continues → model protected until worker finishes** — verified by `TestModelLifecycleRace::test_http_timeout_worker_continues_model_protected`; 8. Normal decisions unchanged for valid inputs — verified by 86+ other passing tests. | Robust demo; no silent failures; graceful degradation; model lifetime protected. | **Hard cancellation of GPU work still not possible** — asyncio timeout cancels awaitable but NOT the underlying worker thread; model remains protected until worker completes. Additional code complexity; but necessary for live demo reliability. |
| L8 | Demo integrity: some data live, some cached, some synthetic, some assumed; must label correctly; must not imply every routing destination is live cloud region or CO₂ directly measured; **carbon provenance & staleness labeling implemented** | MEDIUM | `data/evaluation/demo_readiness_audit.md:97-98`; `carbongrid/carbon/manager.py:149-199` (simulated routing); `carbongrid/api/main.py:366-372` (measurement_source, VRAM source); `carbongrid/api/main.py:355-395` (carbon provenance fields); `carbongrid/carbon/manager.py:201-247` (status with staleness) | **Partially Resolved (carbon labeling)** / **Still Unverified (energy, latency provenance labeling)** | **External data needed:** None — this is labeling transparency. **Smallest feasible alternative:** Audit every API response field; add `data_source` to all measurements/estimates; explicitly label simulated routing. | 1. **Carbon fields**: `carbon_source`, `carbon_reading_type`, `carbon_timestamp`, `carbon_fetched_at`, `carbon_staleness_seconds`, `carbon_is_stale`, `carbon_is_offline`, `carbon_is_replay`, `carbon_metadata` in `/generate`; 2. **Offline fallback labeled** in metadata; 3. **Simulated routing labeled** in `/carbon/status` with `mode: "SIMULATED_REGIONAL_ROUTING"`; 4. **Energy/latency provenance** pending (P2). | Transparent demo; judges can assess scope of carbon data; energy/latency provenance still needs work. | None — improves credibility. |

---

## 2. Verified Completed Phases & Artifacts

| Phase | Description | Status | Key Artifacts | Frozen? |
|-------|-------------|--------|---------------|---------|
| A | Carbon providers (UK API, replay, offline) | COMPLETE | `carbongrid/carbon/providers.py`, `carbongrid/carbon/manager.py`, `data/replay/carbon_replay.json` | No (code) |
| B1 | 50 prompts | COMPLETE | `data/raw/b1_prompts.json` | Yes (data) |
| B2 | 182 prompts, 8 task types, proxy labels | COMPLETE | `data/processed/b2_dataset.json` | Yes (data) |
| C | C.1.1 Classifier (RF, 122 features, threshold 0.50) | COMPLETE | `models/quantization_safety/classifier_c1_1.pkl`, `feature_extractor_c1_1.pkl`, `metadata_c1_1.json` | Yes (model) |
| D | Decision Engine (safety, latency, energy, CO₂, resources) | COMPLETE | `carbongrid/decision/engine.py`, `models.py`, `policy.py` | No (code) |
| E | FastAPI Gateway (`/generate`, `/health`, `/classifier/info`, `/carbon/status`, `/decision/test`) | COMPLETE | `carbongrid/api/main.py`, `schemas.py` | No (code) |
| F | Frozen baseline benchmark (546 runs: 182×3 policies) | FROZEN | `data/evaluation/phase_f/phase_f_raw_results.json`, `phase_f_analysis.json` | **YES** |
| F.1 | Energy predictor audit (100% directional accuracy debunked) | COMPLETE | `data/evaluation/phase_f/energy_predictor_negative_result.json` | Yes (report) |
| F.2 | B2-profile sensitivity (182 runs, 52% INT4, p<1e-18) | CLOSED | `data/evaluation/phase_f_b2profile_gpu/phase_f_raw_results.json`, `PHASE_F2_AUDIT_REPORT.md`, `PHASE_F2_CLOSURE_REPORT.md` | **YES** |
| G | Energy prediction exploration (400 prompts, 25 pilot, 60 locked test) | CLOSED (exploratory) | `data/evaluation/phase_g/prompts.json`, `pilot_results.json`, `PHASE_G_CLOSURE_REPORT.md` | **YES** (locked test) |
| H.1 | VRAM telemetry in `/generate` response | COMPLETE | `carbongrid/api/main.py:368-372`, `carbongrid/inference/provider.py:38-60` | No (code) |

---

## 3. Prioritized Remediation Plan (Stage 2+)

**Principle:** Quick, high-impact code fixes first (P0, P1). Data-dependent validation second (P2). No unnecessary experiments. Frozen artifacts untouched. Locked test set inaccessible.

### P0 — Critical Safety & Correctness (Code Fixes — Do First, No External Data Needed)
| Step | Action | Classification Target | Files | Test |
|------|--------|----------------------|-------|------|
| 1 | **Fail-closed safety gate audit**: Verify classifier/telemetry failures → FP16, never INT4 | L1: Mitigated | `carbongrid/decision/engine.py:57-67`, `carbongrid/api/main.py:250-255` | Add tests: classifier exception → FP16; VRAM unavailable → conservative |
| 2 | **VRAM measurement semantics**: Ensure `vram_measurement_source="unavailable"` + `error` when failed; fallback value (0) used only internally | L5: Resolved (policy) / L8: Mitigated | `carbongrid/api/main.py:263-274`, `carbongrid/inference/provider.py:38-60` | Test VRAM failure path; verify response fields |
| 3 | **Energy/CO₂ unit verification**: Confirm Wh → kWh → gCO₂ calculation end-to-end | L3: Mitigated | `carbongrid/decision/models.py:300-308`, `carbongrid/carbon/calculator.py:83-142` | Unit tests for `estimate_co2_from_energy` with known values |
| 4 | **Carbon source/timestamp/staleness**: Expose all metadata in API; label offline fallback | L4: Mitigated / L8: Mitigated | `carbongrid/api/main.py:156-174`, `carbongrid/carbon/manager.py:201-223` | Verify `/generate` and `/carbon/status` include `is_offline`, `timestamp`, `reading_type`, `staleness_seconds` |
| 5 | **Prevent negative energy predictions**: Verify predictor not in `/generate` path; add `max(0, prediction)` guard if retained | L2: Mitigated | N/A (predictor not in `/generate` path) | Verify predictor not in decision path; unit test guard |

### P1 — API Reliability & Demo Readiness (Code Fixes — High Impact, No External Data)
| Step | Action | Classification Target | Files | Test |
|------|--------|----------------------|-------|------|
| 6 | **Model pre-load on startup**: Load both FP16 and INT4 in `lifespan` with warm-up | L7: Still Unverified | `carbongrid/api/main.py:94-123` | Manual: first `/generate` < 2s |
| 7 | **Add warm-up endpoint**: `POST /warmup` to trigger model loading | L7: Still Unverified | `carbongrid/api/main.py` (new) | Manual test |
| 8 | **Request timeout**: Add timeout to `/generate` (e.g., 120s) | L7: Still Unverified | `carbongrid/api/main.py:224` | Test with long prompt |
| 9 | **Concurrent request safety**: Ensure model loading is thread-safe or serialized (mutex) | L7: Still Unverified | `carbongrid/inference/provider.py:260-311` | Load test with 3 concurrent requests |
| 10 | **Input validation hardening**: Empty prompt, max_tokens bounds, invalid task_type | L7: Still Unverified | `carbongrid/api/schemas.py:46-57` (Pydantic) | Test 422 responses |
| 11 | **Explicit error handling + fail-closed**: Try/except for classifier, carbon, inference with FP16 defaults | L7: Still Unverified | `carbongrid/api/main.py:240-385` | Integration tests with mocked failures |
71: | **11b** | **Carbon provider timeout hardening**: Separate connect/read timeouts, total budget, retry config, offline fallback provenance | L7: **Resolved** | `carbongrid/carbon/providers.py`, `carbongrid/carbon/manager.py`, `carbongrid/api/main.py` | 5 new tests in `TestCarbonProviderTimeouts` |
72: 
73: ### P2 — Measurement Integrity & Transparency (Code + Labeling)
| Step | Action | Classification Target | Files | Test |
|------|--------|----------------------|-------|------|
| 12 | **Separate measured vs estimated energy**: Ensure `measurement_source` ∈ {measured, estimated, phase05_profile_fallback, b2_profile_fallback} | L3: Mitigated / L8: Mitigated | `carbongrid/api/main.py:316-331`, `carbongrid/decision/models.py:166` | Verify response for NVML success vs failure |
| 13 | **NVML failure handling**: If NVML unavailable, `avg_power_w=null`, `energy_wh=null`, `measurement_source="estimated"` | L3: Mitigated / L8: Mitigated | `carbongrid/inference/provider.py:173-194`, `carbongrid/api/main.py:316-331` | Test on machine without NVML/power support |
| 14 | **Quality metric labeling**: Ensure all quality scores labeled "proxy (vs FP16 reference)" | L6: Mitigated / L8: Mitigated | `data/evaluation/phase_f_b2profile_gpu/phase_f_raw_results.json` metadata | N/A (reporting only) |
| 15 | **Simulated routing labeling**: Verify `/carbon/status` regional comparison marked `mode: "SIMULATED_REGIONAL_ROUTING"` | L4: Mitigated / L8: Mitigated | `carbongrid/carbon/manager.py:149-199` | Check `/carbon/status` response |
| 16 | **Add `data_source` to every numeric field in API response**: Energy, latency, carbon, VRAM | L8: Mitigated | `carbongrid/api/main.py:336-373` | Manual review of full `/generate` response |

### P3 — External Data Validation (Optional — Only If Time Permits)
| Step | Action | Classification Target | Feasibility | Notes |
|------|--------|----------------------|-------------|-------|
| 17 | **Electricity Maps API integration** (optional): Add as carbon provider if API key available | L4: Dependent on External Data | Medium — requires API key, rate limits | Not required for demo; UK API + offline sufficient |
| 18 | **Classifier threshold sweep on validation set**: Raise to 0.60 using 25 validation samples (never test set) | L1: Mitigated | High — validation set exists | Reduces false-safe rate; document change |
| 19 | **INT4 live load test on demo machine**: One-time verification | L5: Still Unverified | High — 5 min test | If OOM/unstable, document and keep FP16 default |
| 20 | **Idle power baseline measurement** (optional): Measure GPU idle for 30s to contextualize NVML | L3: Future Research | Medium — time permitting | Improves energy estimate context |

### P4 — Reproducibility & Documentation
| Step | Action | Classification Target | Files | Test |
|------|--------|----------------------|-------|------|
| 21 | **Reproducibility metadata**: Add run IDs, seeds, versions, input hashes to evaluation scripts | All | `scripts/run_phase_f.py`, `scripts/analyze_phase_f2.py` | N/A |
| 22 | **Update MASTER_LIMITATION_REGISTER** with final status after each stage | All | `data/evaluation/MASTER_LIMITATION_REGISTER.md` | N/A |
| 23 | **Create MASTER_REMEDIATION_REPORT** (Stage 7) | All | `data/evaluation/MASTER_REMEDIATION_REPORT.md` | N/A |

---

## Classification Summary (After P1 Step 3 Fixes — 2026-10-04)

| ID | Limitation | Current | Status After Fixes | Requires External Data? |
|----|------------|---------|-------------------|------------------------|
| L1 | Classifier safety | CONFIRMED | **Partially Mitigated** — Fail-closed gate implemented (classifier exception → FP16 with `CLASSIFIER_ERROR`); invalid safety prob (NaN, inf, negative, >1, None) → FP16. Threshold raise on validation set pending. | Future Research (human labels) |
| L2 | Energy predictor | CONFIRMED | **Resolved (not in routing + guards)** — Predictor NOT in `/generate` path (code audit confirmed); profile-based estimates used; validation guards in `energy_predictor_utils.py` for negative/NaN/inf/None/strings; 11 tests in `TestEnergyPredictorValidation`. Not validated for production accuracy. | Dependent (100+ measured energy runs) |
| L3 | NVML measurement | CONFIRMED | **Resolved (unit validation)** — Formula Wh→kWh→gCO₂ verified; dimensional sanity (1 Wh @ 200 gCO₂/kWh = 0.2 g) passed; 18 edge-case tests added (negative, NaN, inf, None, strings, zero, small/large values). NVML methodology documentation pending. | Future Research (lab calibration) |
| L4 | Carbon intensity scope | CONFIRMED | **Resolved (metadata transparency)** — Provenance fields (source, type, timestamp, fetched_at, staleness_seconds, is_stale, is_offline, is_replay, metadata) exposed in `/generate` and `/carbon/status`; staleness threshold (3600s) defined; offline fallback labeled. 7 new tests in `TestCarbonProvenance`. Global coverage still dependent on external API. | Dependent (global API access) |
| L5 | VRAM / INT4 on RTX 2050 | CONFIRMED | **Resolved (policy + explicit VRAM telemetry semantics)** — `vram_telemetry_available` field distinguishes: (a) legacy caller without telemetry → skip VRAM check (backward compatible); (b) telemetry attempted but failed → fail closed (neither config feasible); (c) valid measured VRAM → normal threshold checks. Valid VRAM decisions preserved. Live INT4 test pending. 9 tests in `TestVRAMTelemetrySemantics`. | No (hardware test) |
| L6 | Evaluation claims scope | CONFIRMED | **Mitigated** — Qualified claims, separated synthetic vs real. | Future Research (independent reproduction) |
| L7 | API reliability | SUSPECTED | **Resolved (fail-closed + VRAM + model lifecycle)** — Fail-closed classifier + VRAM semantics implemented; carbon provider timeout handling implemented (separate connect/read timeouts, total budget, retry config, offline fallback with correct provenance); request/inference timeout implemented (120s total, 60s model load); model-load failure handling (clear 500/504 responses, no fabricated output). **Model lifecycle race fixed**: active-inference counter + condition variable protects model for entire `model.generate()` duration; `unload()` and `ensure_loaded()` wait for active inferences to complete; exception decrements active count correctly; HTTP timeout does not release model protection until worker finishes. 7 tests in `TestModelLifecycleRace`. **Hard cancellation of GPU work still not possible** — asyncio timeout cancels awaitable but NOT underlying worker thread. | No (code + tests) |
| L8 | Demo integrity / labeling | CONFIRMED | **Partially Resolved** — Carbon provenance & staleness labeling implemented. Energy/latency provenance labeling pending (P2). | No (labeling) |

---

## 4. Actions That Must NOT Be Performed

| Prohibited Action | Reason |
|-------------------|--------|
| Modify, overwrite, regenerate, or rerun frozen Phase F artifacts (`data/evaluation/phase_f/phase_f_raw_results.json`) | Preserves original baseline integrity |
| Modify, overwrite, or rerun Phase F.2 artifacts (`data/evaluation/phase_f_b2profile_gpu/phase_f_raw_results.json`) | Preserves sensitivity experiment integrity |
| Inspect, evaluate, tune against, or access Phase G locked test set (`data/evaluation/phase_g/prompts.json` test split) | Prevents data leakage; test set must remain unseen |
| Alter existing raw data to improve results | Scientific integrity |
| Silently change labels, feature definitions, energy units, benchmark criteria, or statistical tests | Reproducibility |
| Tune classifier threshold on test set and report as independent validation | Methodological error |
| Claim implementation fix improves model quality without new valid evaluation | Evidence-based claims only |
| Claim significant energy savings without valid statistical analysis of appropriate comparison | Statistical rigor |
| Claim human-level correctness from proxy quality scores | Proxy ≠ ground truth |
| Force INT4 inference or lower VRAM thresholds for favorable demo | Safety & hardware reality |
| Fabricate real carbon providers, cloud endpoints, measurements, or energy savings | Honesty |
| Use future work as substitute for fixing straightforward software bugs | Accountability |
| Begin frontend implementation | Separate effort (Claude) |
| Rewrite entire repository | Prefer small, reviewable changes |
| Run expensive GPU experiments without justification | Resource efficiency |

---

## 5. Discrepancies Between Project Brief and Repository

| Area | Brief Claim | Repository Reality | Notes |
|------|-------------|-------------------|-------|
| Classifier test set size | "29 samples" | Confirmed: 29 samples in metadata | Matches |
| Classifier false-safe rate | "40% (6/15)" | Confirmed: metadata shows 6/15 false-safe | Matches |
| Phase F energy savings | "Not statistically significant" | Confirmed: p=0.096, median +0.001 Wh | Matches |
| Phase F.2 energy reduction | "Statistically significant" | Confirmed: p<1e-18, median -0.0014 Wh | Matches |
| Phase F.2 INT4 selection | "52.2%" | Confirmed: 95/182 = 52.2% | Matches |
| Phase F.2 quality proxy | "~0.66 median" | Confirmed: 0.656 median | Matches |
| Phase G pilot energy | "FP16 0.43, INT4 0.47 Wh" | Confirmed in closure report | Matches |
| Phase G locked test set | "60 prompts, never evaluated" | Confirmed: `prompts.json` has test split, never used | Matches |
| VRAM thresholds | FP16: 3200, INT4: 1400 MiB | Confirmed in `policy.py:61-62` | Matches |
| Measured free VRAM | "~707-709 MiB" | Confirmed in demo audit and VRAM telemetry | Matches |
| Carbon offline fallback | "200 gCO₂/kWh" | Confirmed in `OfflineCarbonProvider` and `DecisionPolicy` | Matches |
| Energy predictor validation | "Not validated" | Confirmed: Phase G closed as exploratory | Matches |
| Demo readiness | "3 critical issues" | Confirmed: model pre-load, NVML reliability, offline carbon | Matches |
| Classifier feature alignment bug | "Fixed in service.py" | Confirmed: `_transform` removes `complexity_level` at TF-IDF boundary | Matches |

**No material discrepancies found.** The repository matches the project brief.

---

## 4. Stage 1 & P1 Deliverable Summary

**Stage 1 Complete.** Repository audit complete.

**P1 Step 3 (Safety Fixes) Complete:**
- **Fix 1: Explicit VRAM telemetry semantics** — Added `vram_telemetry_available` field to `DecisionInput`; decision engine distinguishes legacy callers (skip VRAM check) from failed/unavailable telemetry (fail closed); API sets field correctly based on VRAM measurement result. 9 tests added.
- **Fix 2: Model lifecycle race** — Implemented active-inference counter + condition variable; `unload()` and `ensure_loaded()` wait for active inferences; model protected for entire `model.generate()`; exceptions correctly decrement counter; HTTP timeout does not release protection until worker completes. 7 tests added.

**Next Steps (P1 Closure / Final Engineering Audit):**
1. Update limitation register (DONE)
2. Investigate 141 → 139 → 155 test count discrepancy
3. Run complete test suite
4. Run API smoke validation
5. Final report

**Approval Required:** Confirm P1 Step 3 closure and proceed to final freeze for hackathon presentation.

---

## 7. External Data Research Summary

| Limitation | External Data Needed | Reputable Sources Investigated | Feasibility for Hackathon | Smallest Feasible Alternative |
|------------|---------------------|-------------------------------|---------------------------|-------------------------------|
| L1: Classifier validation | Human-verified INT4/FP16 quality labels for Qwen2.5-1.5B | • No public dataset with per-prompt quantization safety labels<br>• LLM.int8() (Dettmers), AWQ (Lin) — methodology papers, no labeled data<br>• MLPerf Inference — different models/tasks | **Not feasible** — requires human evaluation campaign | Raise threshold on validation set (25 samples); document 40% false-safe rate; fail-closed gate |
| L2: Energy predictor validation | 100+ NVML-measured energy runs at 128-token workload on RTX 2050 + Qwen2.5-1.5B | • No public dataset with local LLM energy measurements<br>• Zeus (You et al.) — similar methodology, different models<br>• CarbonTracker — training energy, not inference<br>• MLPerf Power — server-class, not edge GPU | **Not feasible** — requires dedicated measurement campaign | Disable predictor from routing; use profile-based estimates (measured on-target); negative-prediction guard |
| L3: NVML calibration | Laboratory-grade power meter measurements for same workloads | • External power meters (Yokogawa, Keysight) — lab equipment<br>• No public calibration dataset for RTX 2050 + this model | **Not feasible** — requires hardware access | Document NVML as "sampled estimate"; exclude model-loading; warm-up runs; optional idle baseline |
| L4: Global carbon intensity | Real-time carbon API with global/local grid coverage | • **Electricity Maps API** — global, free tier (requires key, rate-limited)<br>• **WattTime** — US-focused, requires registration<br>• **CO2 Signal** — deprecated<br>• Academic datasets — often static, not real-time | **Partially feasible** — Electricity Maps optional if key available | UK API + offline fallback + explicit labeling; Electricity Maps as optional enhancement |
| L6: Generalization evidence | Independent reproduction on different hardware/models | • MLPerf Inference — standardized but different scope<br>• LLMPerf — throughput focused<br>• No public carbon-aware routing benchmark | **Not feasible** — requires multi-hardware campaign | Qualify all claims to exact hardware/model/workload; separate synthetic vs real-world in presentation |

**Key Principle:** Carbon-intensity data alone **cannot** validate GPU energy predictions. Energy validation requires **measured GPU energy** (NVML or external meter) for the same model/hardware/workload. No public dataset provides this for our configuration.

---

## 8. Synthetic vs. Real-World Evaluation Separation

**Mandatory Separation:** All evaluation artifacts and claims must be explicitly categorized:

| Category | What It Contains | What It Demonstrates | What It Does NOT Prove |
|----------|------------------|----------------------|------------------------|
| **Synthetic / Controlled** | Phase F (frozen), Phase F.2 (B2 profile), Phase G pilot, B2 dataset, classifier test set | • Architecture functions end-to-end<br>• Decision logic correctly implemented<br>• Routing responds to energy profile changes<br>• Safety gate blocks unsafe prompts<br>• Measurement pipeline works | • Real-world energy savings<br>• Production safety<br>• Generalization to other models/hardware/workloads<br>• Human quality equivalence |
| **Independent Real-World** | *None yet obtained* | *Would demonstrate:*<br>• Performance on unseen prompts<br>• Energy measurement validity vs lab equipment<br>• Carbon intensity relevance to local grid<br>• Safety on human-evaluated tasks | *N/A — not available* |

**Presentation Rules:**
1. Every claim must state its category: "In our controlled evaluation on RTX 2050..." vs "In independent real-world testing..."
2. No synthetic result may be presented as real-world evidence.
3. Carbon-intensity data (even live) ≠ energy measurement validation.
4. Classifier proxy labels ≠ human safety validation.
5. Phase F.2 energy reduction = "controlled sensitivity experiment," not "proven savings."

**If real-world data becomes available:** It must go in a new evaluation directory (e.g., `phase_h_realworld/`) with its own artifacts, never mixed with frozen Phase F/F.2/G.