"""
CarbonGrid-AI FastAPI Gateway.

Main application that integrates:
- C.1.1 Quantization-Safety Classifier
- Phase A Carbon Provider
- Phase D Decision Engine
- Qwen2.5-1.5B-Instruct Inference Provider (FP16/INT4)
"""

import os
import sys
import time
import uuid
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from carbongrid.api.schemas import (
    GenerationRequest,
    GenerationResponse,
    HealthResponse,
    Precision,
    DataSourceType,
)
from carbongrid.classifier.service import ClassifierService, get_classifier_info
from carbongrid.carbon.manager import CarbonManager, create_carbon_manager, CarbonMode
from carbongrid.decision import (
    DecisionEngine,
    DecisionInput,
    DecisionPolicy,
    Precision as DecisionPrecision,
    DataSourceType as DecisionDataSourceType,
    DecisionReason,
    FallbackPolicy,
    DEFAULT_PROFILES,
    get_default_profile,
)
from carbongrid.inference.provider import get_free_vram_mb, VRAMMeasurementResult


# Configuration from environment variables with defaults
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "7860"))
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
DEVICE_INDEX = int(os.getenv("DEVICE_INDEX", "0"))
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "1"))

CARBON_MODE = os.getenv("CARBON_MODE", "live")
CARBON_CONNECT_TIMEOUT = float(os.getenv("CARBON_CONNECT_TIMEOUT", "3.0"))
CARBON_READ_TIMEOUT = float(os.getenv("CARBON_READ_TIMEOUT", "7.0"))
CARBON_MAX_TOTAL_TIME = float(os.getenv("CARBON_MAX_TOTAL_TIME", "15.0"))

FP16_VRAM_MB = int(os.getenv("FP16_VRAM_MB", "3200"))
INT4_VRAM_MB = int(os.getenv("INT4_VRAM_MB", "1400"))

DECISION_PROFILE = os.getenv("DECISION_PROFILE", "phase05")

GENERATE_TIMEOUT_SECONDS = 120.0  # Total request timeout
MODEL_LOAD_TIMEOUT_SECONDS = 60.0  # Model loading timeout

# Global service instances
classifier_service: Optional[ClassifierService] = None
carbon_manager: Optional[CarbonManager] = None
decision_engine: Optional[DecisionEngine] = None
inference_provider: Optional[Any] = None  # Will import lazily
decision_policy: Optional[DecisionPolicy] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - handles startup and shutdown."""
    global classifier_service, carbon_manager, decision_engine, inference_provider, decision_policy
    
    print("=" * 70)
    print("CARBONGRID-AI STARTUP")
    print("=" * 70)
    
    # 1. Load C.1.1 Classifier
    print("\n[1/4] Loading C.1.1 Classifier...")
    try:
        classifier_service = ClassifierService().load()
    except Exception as e:
        print(f"  WARNING: Failed to load classifier: {e}")
        classifier_service = None
    
    # 2. Initialize Carbon Manager
    print("\n[2/4] Initializing Carbon Manager...")
    try:
        carbon_manager = create_carbon_manager(
            CARBON_MODE,
            connect_timeout=CARBON_CONNECT_TIMEOUT,
            read_timeout=CARBON_READ_TIMEOUT,
            max_total_time_seconds=CARBON_MAX_TOTAL_TIME,
        )
        print(f"  Mode: {CARBON_MODE} (with replay/offline fallback)")
        print(f"  Timeouts: connect={CARBON_CONNECT_TIMEOUT}s, read={CARBON_READ_TIMEOUT}s, total_budget={CARBON_MAX_TOTAL_TIME}s")
    except Exception as e:
        print(f"  WARNING: Failed to create carbon manager: {e}")
        # Fallback to offline
        carbon_manager = create_carbon_manager("offline")
    
    # 3. Initialize Decision Engine with default policy
    print("\n[3/4] Initializing Decision Engine...")
    decision_policy = DecisionPolicy(
        profile_name=DECISION_PROFILE,
        safety_threshold=0.50,   # C.1.1 threshold
        min_vram_mb_for_fp16=FP16_VRAM_MB,
        min_vram_mb_for_int4=INT4_VRAM_MB,
    )
    decision_engine = DecisionEngine(decision_policy)
    print(f"  Policy: {decision_policy.version}")
    print(f"  Safety threshold: {decision_policy.safety_threshold}")
    print(f"  Profile: {decision_policy.profile_name}")
    print(f"  VRAM thresholds: FP16={FP16_VRAM_MB}MB, INT4={INT4_VRAM_MB}MB")
    
    # 4. Initialize Inference Provider (NO pre-load - decision happens first)
    print("\n[4/4] Initializing Inference Provider...")
    try:
        # Import here to avoid loading transformers at startup
        from carbongrid.inference.provider import InferenceProvider, create_inference_provider
        inference_provider = create_inference_provider(
            model_id=MODEL_ID,
            device_index=DEVICE_INDEX,
            max_concurrent_inferences=MAX_CONCURRENT,
        )
        print(f"  Model: {MODEL_ID}")
        print(f"  Device: CUDA (index {DEVICE_INDEX})")
        print(f"  Max concurrent: {MAX_CONCURRENT}")
        print(f"  Model loading: LAZY (on first request after decision)")
    except Exception as e:
        print(f"  WARNING: Failed to create inference provider: {e}")
        inference_provider = None
    
    print("\n" + "=" * 70)
    print("CARBONGRID-AI READY")
    print("=" * 70)
    
    yield
    
    # Shutdown
    print("\nShutting down CarbonGrid-AI...")
    if inference_provider:
        inference_provider.unload()
    print("Shutdown complete.")


app = FastAPI(
    title="CarbonGrid-AI",
    description="Carbon-aware middleware for local LLM inference",
    version="0.1.0",
    lifespan=lifespan,
)

# Allow the console to call the API even when index.html is opened from disk (file://)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_INDEX = Path(__file__).parent.parent.parent / "frontend" / "index.html"


def get_current_carbon_intensity() -> tuple:
    """
    Get current carbon intensity from the carbon manager.
    
    Returns:
        (intensity_gco2_per_kwh, data_source_type, zone, reading_type, timestamp, fetched_at, staleness_seconds, is_stale, is_offline, is_replay, metadata)
    """
    if carbon_manager is None:
        return (200.0, DecisionDataSourceType.OFFLINE, "national", "actual", None, None, None, None, True, False, {"fallback": True})
    
    reading = carbon_manager.get_current_carbon("national")
    if reading:
        # Map DataSource to DecisionDataSourceType
        source_map = {
            "uk_carbon_intensity": DecisionDataSourceType.LIVE,
            "replay": DecisionDataSourceType.REPLAY,
            "offline": DecisionDataSourceType.OFFLINE,
        }
        source_type = source_map.get(reading.data_source.value, DecisionDataSourceType.UNKNOWN)
        
        now = datetime.now(timezone.utc)
        staleness = reading.staleness_seconds(now)
        is_stale = reading.is_stale(3600, now)
        fetched_at = reading.fetched_at.isoformat() if reading.fetched_at else None
        
        return (
            reading.carbon_intensity_gco2_per_kwh,
            source_type,
            reading.zone,
            reading.reading_type.value,
            reading.timestamp.isoformat(),
            fetched_at,
            round(staleness, 1) if staleness is not None else None,
            is_stale,
            reading.is_offline,
            reading.is_replay,
            reading.metadata,
        )
    else:
        return 200.0, DecisionDataSourceType.OFFLINE, "national", "actual", None, None, None, None, True, False, {"fallback": True}


def get_energy_estimates(profile_name: str = "phase05") -> Dict[str, float]:
    """Get energy and latency estimates from the specified profile."""
    profile = get_default_profile(profile_name)
    return {
        "fp16_energy_wh": profile["fp16"]["energy_wh"],
        "int4_energy_wh": profile["int4"]["energy_wh"],
        "fp16_latency_ms": profile["fp16"]["latency_ms"],
        "int4_latency_ms": profile["int4"]["latency_ms"],
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    classifier_info = get_classifier_info()
    
    carbon_status = {}
    if carbon_manager:
        carbon_status = carbon_manager.get_status()
    
    inference_status = {}
    if inference_provider:
        inference_status = inference_provider.get_status()
    
    return HealthResponse(
        status="healthy" if classifier_service and classifier_service.is_loaded() else "degraded",
        version="0.1.0",
        classifier_loaded=classifier_service is not None and classifier_service.is_loaded(),
        carbon_provider_mode=carbon_status.get("mode", "unknown"),
        inference_provider_status=inference_status,
    )


@app.get("/classifier/info")
async def classifier_info():
    """Get C.1.1 classifier information."""
    return get_classifier_info()


@app.get("/carbon/status")
async def carbon_status():
    """Get carbon provider status."""
    if carbon_manager is None:
        raise HTTPException(status_code=503, detail="Carbon manager not initialized")
    return carbon_manager.get_status()


@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    """
    Generate text with carbon-aware precision selection.
    
    Pipeline:
    1. Extract classifier features from prompt
    2. Get safety probability from C.1.1 classifier
    3. Get current carbon intensity
    4. Run Decision Engine
    5. Execute inference at selected precision
    6. Return response with full telemetry
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    # Validate services
    if classifier_service is None or not classifier_service.is_loaded():
        raise HTTPException(status_code=503, detail="Classifier not loaded")
    if decision_engine is None:
        raise HTTPException(status_code=503, detail="Decision engine not initialized")
    if inference_provider is None:
        raise HTTPException(status_code=503, detail="Inference provider not initialized")
    
    async def _generate_with_timeout():
        """Inner generation logic with timeout support."""
        # Step 1: Get safety probability from classifier
        try:
            safety_result = classifier_service.predict_safety(
                prompt=request.prompt,
                task_type=request.task_type or "unknown",
            )
            safety_probability = safety_result["safety_probability"]
            safety_threshold = safety_result["threshold"]
        except Exception as e:
            # Classifier failed - fail-closed to FP16 with explicit reason
            import logging
            logging.getLogger(__name__).warning(f"Classifier prediction failed: {e}. Fail-closed to FP16.")
            safety_probability = 0.0  # Will trigger CLASSIFIER_ERROR in decision engine
            safety_threshold = 0.50
        
        # Step 2: Get current carbon intensity with provenance
        (carbon_intensity, carbon_source, carbon_zone, carbon_reading_type, 
         carbon_timestamp, carbon_fetched_at, carbon_staleness_seconds, 
         carbon_is_stale, carbon_is_offline, carbon_is_replay, carbon_metadata) = get_current_carbon_intensity()
        
        # Step 3: Measure free VRAM for resource feasibility check
        free_vram_mb: Optional[int] = None
        vram_telemetry_available: bool = False
        if inference_provider is not None:
            vram_result = get_free_vram_mb(inference_provider.device_index)
            if vram_result.is_available:
                free_vram_mb = vram_result.free_mb
                vram_telemetry_available = True
            elif vram_result.is_unavailable:
                # VRAM measurement failed - explicitly mark as unavailable
                # This triggers fail-closed behavior in decision engine
                import logging
                logging.getLogger(__name__).warning(f"VRAM measurement unavailable: {vram_result.error}. Fail-closed: neither FP16 nor INT4 will be considered resource-feasible.")
                free_vram_mb = None
                vram_telemetry_available = True  # Telemetry was attempted but failed
            else:
                free_vram_mb = None
                vram_telemetry_available = True  # Telemetry was attempted but failed
        
        # Step 4: Get energy/latency estimates from profile
        estimates = get_energy_estimates(decision_policy.profile_name)
        
        # Step 5: Build DecisionInput
        decision_input = DecisionInput(
            safety_probability=safety_probability,
            safety_threshold=safety_threshold,
            carbon_intensity_gco2_per_kwh=carbon_intensity,
            carbon_data_source=carbon_source,
            carbon_zone=carbon_zone,
            carbon_reading_type=carbon_reading_type,
            carbon_timestamp=carbon_timestamp,
            fp16_estimated_energy_wh=estimates["fp16_energy_wh"],
            int4_estimated_energy_wh=estimates["int4_energy_wh"],
            fp16_estimated_latency_ms=estimates["fp16_latency_ms"],
            int4_estimated_latency_ms=estimates["int4_latency_ms"],
            energy_data_source=DecisionDataSourceType.ESTIMATED,
            latency_data_source=DecisionDataSourceType.ESTIMATED,
            latency_requirement_ms=request.latency_requirement_ms,
            fallback_policy=decision_policy.fallback_policy,
            default_precision=decision_policy.default_precision,
            request_id=request_id,
            decision_policy_version=decision_policy.version,
            gpu_vram_free_mb=free_vram_mb,
            vram_telemetry_available=vram_telemetry_available,
        )
        
        # Step 6: Run Decision Engine
        decision = decision_engine.decide(decision_input)
        
        # Step 7: Execute inference at selected precision with model-load timeout
        selected_precision = Precision(decision.selected_precision.value)
        
        # Ensure model is loaded with timeout
        try:
            from carbongrid.inference.provider import Precision as InferencePrecision
            inf_precision = InferencePrecision(decision.selected_precision.value)
            
            # Use asyncio.to_thread for model loading with timeout
            # This prevents blocking the event loop during model loading
            await asyncio.wait_for(
                asyncio.to_thread(inference_provider.ensure_loaded, inf_precision),
                timeout=MODEL_LOAD_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail=f"Model loading timed out after {MODEL_LOAD_TIMEOUT_SECONDS}s. Selected precision: {decision.selected_precision.value}"
            )
        except Exception as e:
            # Model loading failed - return clear error
            raise HTTPException(
                status_code=500,
                detail=f"Model loading failed for {decision.selected_precision.value}: {str(e)}"
            )
        
        # Step 8: Execute inference with overall request timeout
        inf_start = time.time()
        try:
            metrics = await asyncio.wait_for(
                asyncio.to_thread(
                    inference_provider.run_inference,
                    prompt=request.prompt,
                    max_new_tokens=request.max_new_tokens,
                    precision=selected_precision,
                ),
                timeout=GENERATE_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            # Request timed out - return explicit timeout error
            raise HTTPException(
                status_code=504,
                detail=f"Inference timed out after {GENERATE_TIMEOUT_SECONDS}s. Max tokens: {request.max_new_tokens}, Precision: {decision.selected_precision.value}"
            )
        except Exception as e:
            # Inference failed - return clear error
            raise HTTPException(
                status_code=500,
                detail=f"Inference failed: {str(e)}"
            )
        inf_latency_ms = (time.time() - inf_start) * 1000
        
        # Check if inference was rejected due to concurrency limit
        if not metrics.success:
            if "busy" in metrics.error.lower() or "concurrent" in metrics.error.lower():
                raise HTTPException(
                    status_code=503,
                    detail=f"Inference service busy: {metrics.error}"
                )
            # Other inference failure - return error
            raise HTTPException(
                status_code=500,
                detail=f"Inference failed: {metrics.error}"
            )
        inf_latency_ms = (time.time() - inf_start) * 1000
        
        # Step 9: Calculate actual CO2 if energy was measured
        actual_co2_g = None
        actual_energy_wh = metrics.energy_wh
        measurement_source = DecisionDataSourceType.ESTIMATED
        
        if metrics.success and metrics.energy_wh is not None and metrics.avg_power_w is not None:
            # Energy was measured via NVML
            energy_kwh = metrics.energy_wh / 1000.0
            actual_co2_g = energy_kwh * carbon_intensity
            measurement_source = DecisionDataSourceType.MEASURED
        else:
            # Use estimated energy for CO2 calculation
            estimated_energy_kwh = decision.estimated_energy_wh / 1000.0
            actual_co2_g = estimated_energy_kwh * carbon_intensity
            actual_energy_wh = decision.estimated_energy_wh
            measurement_source = DecisionDataSourceType.ESTIMATED
        
        total_latency_ms = (time.time() - start_time) * 1000
        
        # Build response
        carbon_grid_data = {
            "request_id": request_id,
            "selected_precision": decision.selected_precision.value,
            "decision_reason": decision.decision_reason.value,
            "safety_probability": round(decision.safety_probability, 4),
            "safety_threshold": round(decision.safety_threshold, 2),
            "safety_gate_passed": decision.safety_gate_passed,
            "carbon_intensity_gco2_per_kwh": round(decision.carbon_intensity_gco2_per_kwh, 1),
            "carbon_source": decision.carbon_data_source.value,
            "carbon_zone": decision.carbon_zone,
            "carbon_reading_type": decision.carbon_reading_type,
            "carbon_timestamp": decision.carbon_timestamp,
            "carbon_fetched_at": carbon_fetched_at,
            "carbon_staleness_seconds": carbon_staleness_seconds,
            "carbon_is_stale": carbon_is_stale,
            "carbon_is_offline": carbon_is_offline,
            "carbon_is_replay": carbon_is_replay,
            "carbon_metadata": carbon_metadata,
            "estimated_energy_wh": round(decision.estimated_energy_wh, 6),
            "estimated_co2_g": round(decision.estimated_co2_g, 6),
            "estimated_co2_mg": round(decision.estimated_co2_mg, 3),
            "estimated_latency_ms": round(decision.estimated_latency_ms, 0),
            "latency_constraint_status": decision.latency_constraint_status,
            "latency_requirement_ms": decision.latency_requirement_ms,
            "fallback_used": decision.fallback_used,
            "decision_policy_version": decision.decision_policy_version,
            "energy_data_source": decision.energy_data_source.value,
            # Actual measurements
            "actual_latency_ms": round(inf_latency_ms, 0),
            "actual_energy_wh": round(actual_energy_wh, 6) if actual_energy_wh else None,
            "actual_co2_g": round(actual_co2_g, 6) if actual_co2_g else None,
            "measurement_source": measurement_source.value,
            "tokens_per_second": round(metrics.tokens_per_second, 1) if metrics.success else None,
            "peak_gpu_memory_mb": metrics.peak_gpu_memory_mb if metrics.success else None,
            "avg_power_w": round(metrics.avg_power_w, 2) if metrics.avg_power_w else None,
            "inference_status": "success" if metrics.success else "failed",
            "error": metrics.error,
            "timestamp": decision.timestamp,
            "total_wall_time_ms": round((time.time() - start_time) * 1000, 0),
            # VRAM telemetry
            "free_vram_mb": free_vram_mb,
            "vram_measurement_source": "measured" if vram_result.is_available else "unavailable",
            "vram_measurement_error": vram_result.error if vram_result.is_unavailable else None,
            "fp16_vram_threshold_mb": decision_policy.min_vram_mb_for_fp16,
            "int4_vram_threshold_mb": decision_policy.min_vram_mb_for_int4,
        }
        
        return GenerationResponse(
            response=metrics.response if metrics.success else "",
            carbon_grid=carbon_grid_data,
        )
    
    try:
        return await asyncio.wait_for(_generate_with_timeout(), timeout=GENERATE_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"Request timed out after {GENERATE_TIMEOUT_SECONDS}s. The entire generation pipeline (classifier + carbon + decision + model_load + inference) exceeded the time budget."
        )


@app.post("/decision/test")
async def test_decision(input_data: Dict[str, Any]):
    """Test the Decision Engine directly with custom input."""
    if decision_engine is None:
        raise HTTPException(status_code=503, detail="Decision engine not initialized")
    
    try:
        # Convert string enums to actual enums
        data = input_data.copy()
        data["carbon_data_source"] = DecisionDataSourceType(data.get("carbon_data_source", "estimated"))
        data["energy_data_source"] = DecisionDataSourceType(data.get("energy_data_source", "estimated"))
        data["latency_data_source"] = DecisionDataSourceType(data.get("latency_data_source", "estimated"))
        data["fallback_policy"] = FallbackPolicy(data.get("fallback_policy", "prefer_fp16"))
        data["default_precision"] = DecisionPrecision(data.get("default_precision", "fp16"))
        
        decision_input = DecisionInput(**data)
        decision = decision_engine.decide(decision_input)
        
        return decision.to_dict()
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Decision failed: {str(e)}")


@app.get("/", include_in_schema=False)
async def console():
    """Serve the CarbonGrid console (frontend/index.html)."""
    if FRONTEND_INDEX.exists():
        return FileResponse(FRONTEND_INDEX, media_type="text/html")
    return await root()


@app.get("/api")
async def root():
    """API info endpoint."""
    return {
        "name": "CarbonGrid-AI",
        "version": "0.1.0",
        "description": "Carbon-aware middleware for local LLM inference",
        "endpoints": {
            "generate": "POST /generate - Generate text with carbon-aware precision",
            "health": "GET /health - Health check",
            "classifier_info": "GET /classifier/info - C.1.1 classifier details",
            "carbon_status": "GET /carbon/status - Carbon provider status",
            "decision_test": "POST /decision/test - Test Decision Engine directly",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)