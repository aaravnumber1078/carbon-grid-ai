"""
Inference Provider for CarbonGrid-AI.

Wraps the existing Qwen2.5-1.5B-Instruct FP16 and INT4 inference implementations.
Handles model loading, inference execution, and GPU resource management.

Designed for sequential model loading on 4GB VRAM (RTX 2050).
"""

import sys
import time
import torch
import gc
import logging
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from contextlib import contextmanager

# Try NVML for power monitoring
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

# Initialize NVML once at module load
if NVML_AVAILABLE:
    try:
        pynvml.nvmlInit()
    except Exception:
        NVML_AVAILABLE = False

logger = logging.getLogger(__name__)


class VRAMMeasurementResult:
    """Result of VRAM measurement with explicit status."""
    
    def __init__(self, free_mb: Optional[int] = None, error: Optional[str] = None):
        self.free_mb = free_mb
        self.error = error
    
    @property
    def is_available(self) -> bool:
        return self.free_mb is not None
    
    @property
    def is_unavailable(self) -> bool:
        return self.free_mb is None and self.error is not None
    
    @classmethod
    def success(cls, free_mb: int) -> 'VRAMMeasurementResult':
        return cls(free_mb=free_mb)
    
    @classmethod
    def failure(cls, error: str) -> 'VRAMMeasurementResult':
        return cls(free_mb=None, error=error)


def get_free_vram_mb(device_index: int = 0) -> VRAMMeasurementResult:
    """
    Get the current free VRAM in MiB for the specified GPU device.
    
    Uses NVML if available, falls back to PyTorch's memory reporting.
    Returns VRAMMeasurementResult with explicit success/failure status.
    
    Args:
        device_index: GPU device index to query (default 0)
        
    Returns:
        VRAMMeasurementResult with free_mb (int) on success, or error message on failure.
    """
    # Try NVML first (most accurate for free memory)
    if NVML_AVAILABLE:
        try:
            import pynvml
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_mb = mem_info.free // (1024 * 1024)
            return VRAMMeasurementResult.success(free_mb)
        except Exception as e:
            logger.warning(f"NVML VRAM measurement failed for device {device_index}: {e}")
            # Fall through to PyTorch fallback
    
    # Fallback to PyTorch (only works if CUDA is available and context exists)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            # Get free memory = total - reserved (reserved includes allocated + cached)
            total_mb = torch.cuda.get_device_properties(device_index).total_memory // (1024 * 1024)
            reserved_mb = torch.cuda.memory_reserved(device_index) // (1024 * 1024)
            free_mb = total_mb - reserved_mb
            if free_mb >= 0:
                return VRAMMeasurementResult.success(free_mb)
    except Exception as e:
        logger.warning(f"PyTorch VRAM measurement failed for device {device_index}: {e}")
        # Fall through to failure
    
    return VRAMMeasurementResult.failure(f"VRAM measurement unavailable for device {device_index}")


class Precision(Enum):
    """Model precision configuration."""
    FP16 = "fp16"
    INT4 = "int4"


@dataclass
class InferenceMetrics:
    """Metrics from a single inference request."""
    latency_ms: float
    input_tokens: int
    output_tokens: int
    tokens_per_second: float
    gpu_memory_mb: int
    peak_gpu_memory_mb: int
    avg_power_w: Optional[float]
    energy_wh: Optional[float]
    response: str
    success: bool
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GPUPowerMonitor:
    """Samples GPU power during inference using NVML."""
    
    def __init__(self, device_index: int = 0, sample_interval_ms: float = 10.0):
        self.device_index = device_index
        self.sample_interval = sample_interval_ms / 1000.0  # seconds
        self.handle = None
        self.sampling = False
        self.samples: List[float] = []
        self._thread = None
        
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
                # Test if power reading works
                pynvml.nvmlDeviceGetPowerUsage(self.handle)
                self.power_supported = True
            except Exception:
                self.power_supported = False
        else:
            self.power_supported = False
    
    def _sample_loop(self):
        import threading
        while self.sampling:
            if self.power_supported and self.handle:
                try:
                    power_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
                    self.samples.append(power_mw / 1000.0)  # Convert to Watts
                except Exception:
                    pass
            time.sleep(self.sample_interval)
    
    def start(self):
        if not self.power_supported:
            return
        self.samples = []
        self.sampling = True
        import threading
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> Dict[str, Optional[float]]:
        if not self.power_supported:
            return {"avg_power_w": None, "energy_wh": None, "samples": 0}
        
        self.sampling = False
        if self._thread:
            self._thread.join(timeout=1.0)
        
        if not self.samples:
            return {"avg_power_w": None, "energy_wh": None, "samples": 0}
        
        avg_power = sum(self.samples) / len(self.samples)
        # Energy = average power * time (in hours)
        duration_s = len(self.samples) * self.sample_interval
        energy_wh = avg_power * (duration_s / 3600.0)
        
        return {
            "avg_power_w": round(avg_power, 3),
            "energy_wh": round(energy_wh, 6),
            "samples": len(self.samples),
            "duration_s": round(duration_s, 3)
        }


class InferenceProvider:
    """
    Manages Qwen2.5-1.5B-Instruct inference at FP16 and INT4 precision.
    
    Handles sequential model loading/unloading to fit within 4GB VRAM.
    Thread-safe with concurrency control and model lifetime protection.
    """
    
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device_index: int = 0,
        power_sample_interval_ms: float = 20.0,
        max_concurrent_inferences: int = 1,
    ):
        self.model_id = model_id
        self.device_index = device_index
        self.power_sample_interval_ms = power_sample_interval_ms
        self.max_concurrent_inferences = max_concurrent_inferences
        
        self._model = None
        self._tokenizer = None
        self._current_precision: Optional[Precision] = None
        self._power_monitor = GPUPowerMonitor(device_index, power_sample_interval_ms)
        
        # Concurrency control
        self._load_lock = threading.Lock()  # Protects model loading/unloading
        self._inference_semaphore = threading.Semaphore(max_concurrent_inferences)
        
        # Model lifetime protection: active inference count + condition variable
        # ensure_loaded() and unload() wait for _active_inferences == 0
        self._active_inferences = 0
        self._model_lifetime_cv = threading.Condition(self._load_lock)
        self._shutdown = False
    
    def _load_fp16(self):
        """Load model in FP16 precision."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=torch.float16,
            device_map=None,  # Explicitly NO device_map
            low_cpu_mem_usage=True
        )
        model = model.to("cuda")
        model.eval()
        return model, tokenizer
    
    def _load_int4(self):
        """Load model in INT4 precision using bitsandbytes."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map="auto",  # bitsandbytes needs device_map for offloading
            low_cpu_mem_usage=True
        )
        model.eval()
        return model, tokenizer
    
    def ensure_loaded(self, precision: Precision) -> Tuple[Any, Any]:
        """
        Ensure the model is loaded at the requested precision.
        
        If a different precision is currently loaded, waits for all active
        inferences to complete, then unloads the current model and loads
        the requested precision. Returns (model, tokenizer).
        Thread-safe.
        """
        with self._model_lifetime_cv:
            # Check if already loaded with correct precision
            if (
                self._current_precision is not None
                and precision is not None
                and self._current_precision.value == precision.value
                and self._model is not None
            ):
                return self._model, self._tokenizer
            
            # Wait for all active inferences to complete before replacing model
            while self._active_inferences > 0:
                self._model_lifetime_cv.wait()
            
            # Unload current model if any
            if self._model is not None:
                self._unload_locked()
            
            print(f"Loading {precision.value.upper()} model...")
            load_start = time.perf_counter()
            
            try:
                if precision == Precision.FP16:
                    model, tokenizer = self._load_fp16()
                else:
                    model, tokenizer = self._load_int4()
                
                load_time = time.perf_counter() - load_start
                
                # Memory after loading
                torch.cuda.synchronize()
                mem_after_load = torch.cuda.memory_allocated() // (1024**2)
                torch.cuda.reset_peak_memory_stats()
                
                print(f"  Load time: {load_time:.2f}s")
                print(f"  GPU Memory after load: {mem_after_load} MB")
                
                self._model = model
                self._tokenizer = tokenizer
                self._current_precision = precision
                
                return model, tokenizer
                
            except Exception as e:
                print(f"  FAILED to load {precision.value}: {e}")
                self._model = None
                self._tokenizer = None
                self._current_precision = None
                raise
    
    def _unload_locked(self):
        """Unload the current model. Must be called with _load_lock held."""
        if self._model is not None:
            print(f"Unloading {self._current_precision.value if self._current_precision else 'model'}...")
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            self._current_precision = None
            torch.cuda.empty_cache()
            gc.collect()
            
            mem_after_unload = torch.cuda.memory_allocated() // (1024**2)
            print(f"  GPU Memory after unload: {mem_after_unload} MB")
    
    def unload(self):
        """Unload the current model and free GPU memory. Thread-safe.
        
        Waits for all active inferences to complete before unloading.
        """
        with self._model_lifetime_cv:
            # Wait for all active inferences to complete
            while self._active_inferences > 0:
                self._model_lifetime_cv.wait()
            self._unload_locked()
    
    def run_inference(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        precision: Precision = Precision.FP16,
    ) -> InferenceMetrics:
        """
        Run inference at the specified precision.
        
        Loads the model if needed, runs inference, returns metrics.
        Thread-safe with concurrency control and model lifetime protection.
        The model is protected for the COMPLETE duration of model.generate().
        """
        # Acquire inference semaphore (non-blocking with timeout)
        acquired = self._inference_semaphore.acquire(timeout=10.0)
        if not acquired:
            return InferenceMetrics(
                latency_ms=0,
                input_tokens=0,
                output_tokens=0,
                tokens_per_second=0,
                gpu_memory_mb=0,
                peak_gpu_memory_mb=0,
                avg_power_w=None,
                energy_wh=None,
                response="",
                success=False,
                error="Inference service busy: max concurrent requests reached"
            )
        
        # Mark inference as active for model lifetime protection
        # This must be done BEFORE ensure_loaded() to prevent model replacement
        with self._model_lifetime_cv:
            self._active_inferences += 1
        
        try:
            # Ensure model is loaded (may wait for other active inferences to complete)
            model, tokenizer = self.ensure_loaded(precision)
            
            self._power_monitor.start()
            
            try:
                # Tokenize input
                inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
                input_tokens = inputs.input_ids.shape[1]
                
                torch.cuda.synchronize()
                start = time.perf_counter()
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        repetition_penalty=1.1,
                        return_dict_in_generate=True
                    )
                
                torch.cuda.synchronize()
                latency_ms = (time.perf_counter() - start) * 1000
                
                power_data = self._power_monitor.stop()
                
                # Handle output format
                if hasattr(outputs, 'sequences'):
                    sequences = outputs.sequences
                else:
                    sequences = outputs[0] if isinstance(outputs, tuple) else outputs
                
                output_tokens = sequences[0].shape[0] - input_tokens
                tokens_per_sec = output_tokens / (latency_ms / 1000) if latency_ms > 0 else 0
                
                # Get only new tokens
                new_tokens = sequences[0][input_tokens:]
                response = tokenizer.decode(new_tokens, skip_special_tokens=True)
                
                # Memory stats
                torch.cuda.synchronize()
                gpu_mem = torch.cuda.memory_allocated() // (1024**2)
                peak_mem = torch.cuda.max_memory_allocated() // (1024**2)
                torch.cuda.reset_peak_memory_stats()
                
                return InferenceMetrics(
                    latency_ms=latency_ms,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    tokens_per_second=tokens_per_sec,
                    gpu_memory_mb=gpu_mem,
                    peak_gpu_memory_mb=peak_mem,
                    avg_power_w=power_data.get("avg_power_w"),
                    energy_wh=power_data.get("energy_wh"),
                    response=response,
                    success=True
                )
                
            except Exception as e:
                return InferenceMetrics(
                    latency_ms=0,
                    input_tokens=0,
                    output_tokens=0,
                    tokens_per_second=0,
                    gpu_memory_mb=0,
                    peak_gpu_memory_mb=0,
                    avg_power_w=None,
                    energy_wh=None,
                    response="",
                    success=False,
                    error=str(e)
                )
        finally:
            # Release model lifetime protection
            # This must be done AFTER model.generate() completes (even on exception)
            with self._model_lifetime_cv:
                self._active_inferences -= 1
                # Notify any waiters (ensure_loaded, unload) that an inference completed
                self._model_lifetime_cv.notify_all()
            self._inference_semaphore.release()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current provider status."""
        return {
            "model_id": self.model_id,
            "current_precision": self._current_precision.value if self._current_precision else None,
            "model_loaded": self._model is not None,
            "nvml_power_supported": self._power_monitor.power_supported,
            "device_index": self.device_index,
        }


def create_inference_provider(
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
    device_index: int = 0,
    max_concurrent_inferences: int = 1,
) -> InferenceProvider:
    """Factory function to create InferenceProvider."""
    return InferenceProvider(
        model_id=model_id,
        device_index=device_index,
        max_concurrent_inferences=max_concurrent_inferences,
    )