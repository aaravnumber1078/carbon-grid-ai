#!/usr/bin/env python3
"""
Phase 0.5: Same-Model FP16 vs INT4 Validation for CarbonGrid-AI
Tests Qwen2.5-1.5B-Instruct at FP16 and INT4 precision on RTX 2050 4GB.
"""

import sys
import time
import json
import torch
import gc
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from contextlib import contextmanager

# Try NVML
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

sys.stdout.reconfigure(encoding='utf-8')


@dataclass
class InferenceResult:
    prompt_id: str
    prompt: str
    configuration: str  # "FP16" or "INT4"
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


@contextmanager
def gpu_memory_tracker():
    """Context manager to track GPU memory."""
    torch.cuda.empty_cache()
    gc.collect()
    mem_before = torch.cuda.memory_allocated() // (1024**2)
    peak_before = torch.cuda.max_memory_allocated() // (1024**2)
    torch.cuda.reset_peak_memory_stats()
    
    try:
        yield
    finally:
        mem_after = torch.cuda.memory_allocated() // (1024**2)
        peak_after = torch.cuda.max_memory_allocated() // (1024**2)
        # Return values via closure or global - we'll handle in calling code


def load_model_fp16(model_id: str):
    """Load model in FP16 without device_map."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map=None,  # Explicitly NO device_map
        low_cpu_mem_usage=True
    )
    model = model.to("cuda")
    model.eval()
    return model, tokenizer


def load_model_int4(model_id: str):
    """Load model in INT4 using bitsandbytes."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",  # bitsandbytes needs device_map for offloading
        low_cpu_mem_usage=True
    )
    model.eval()
    return model, tokenizer


def run_inference(model, tokenizer, prompt: str, max_new_tokens: int = 128) -> Dict[str, Any]:
    """Run inference and return metrics."""
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
    
    # Handle both tensor and GenerateDecoderOnlyOutput
    if hasattr(outputs, 'sequences'):
        sequences = outputs.sequences
    else:
        sequences = outputs[0] if isinstance(outputs, tuple) else outputs
    
    output_tokens = sequences[0].shape[0] - input_tokens
    tokens_per_sec = output_tokens / (latency_ms / 1000) if latency_ms > 0 else 0
    
    # Slice to get only NEW tokens (exclude prompt)
    new_tokens = sequences[0][input_tokens:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    return {
        "latency_ms": latency_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tokens_per_sec": tokens_per_sec,
        "response": response
    }


def benchmark_configuration(
    config_name: str,
    load_fn,
    model_id: str,
    prompts: List[Dict[str, str]],
    max_new_tokens: int = 128
) -> List[InferenceResult]:
    """Benchmark a single configuration (load -> test -> unload)."""
    results = []
    
    print(f"\n{'='*60}")
    print(f"Loading {config_name} model...")
    print(f"{'='*60}")
    
    # Load model
    load_start = time.perf_counter()
    try:
        model, tokenizer = load_fn(model_id)
        load_time = time.perf_counter() - load_start
        
        # Memory after loading
        torch.cuda.synchronize()
        mem_after_load = torch.cuda.memory_allocated() // (1024**2)
        torch.cuda.reset_peak_memory_stats()
        
        print(f"  Load time: {load_time:.2f}s")
        print(f"  GPU Memory after load: {mem_after_load} MB")
        
    except Exception as e:
        print(f"  FAILED to load: {e}")
        for p in prompts:
            results.append(InferenceResult(
                prompt_id=p["id"], prompt=p["text"], configuration=config_name,
                latency_ms=0, input_tokens=0, output_tokens=0, tokens_per_second=0,
                gpu_memory_mb=0, peak_gpu_memory_mb=0, avg_power_w=None,
                energy_wh=None, response="", success=False, error=f"Load failed: {e}"
            ))
        return results
    
    # Power monitor
    power_monitor = GPUPowerMonitor(sample_interval_ms=20.0)
    
    # Run prompts
    for p in prompts:
        print(f"  Testing: {p['id']}...", end=" ", flush=True)
        
        power_monitor.start()
        
        try:
            inf_result = run_inference(model, tokenizer, p["text"], max_new_tokens)
            
            power_data = power_monitor.stop()
            
            # Memory
            torch.cuda.synchronize()
            gpu_mem = torch.cuda.memory_allocated() // (1024**2)
            peak_mem = torch.cuda.max_memory_allocated() // (1024**2)
            torch.cuda.reset_peak_memory_stats()
            
            result = InferenceResult(
                prompt_id=p["id"],
                prompt=p["text"],
                configuration=config_name,
                latency_ms=inf_result["latency_ms"],
                input_tokens=inf_result["input_tokens"],
                output_tokens=inf_result["output_tokens"],
                tokens_per_second=inf_result["tokens_per_sec"],
                gpu_memory_mb=gpu_mem,
                peak_gpu_memory_mb=peak_mem,
                avg_power_w=power_data.get("avg_power_w"),
                energy_wh=power_data.get("energy_wh"),
                response=inf_result["response"][:500],
                success=True
            )
            print(f"OK ({inf_result['latency_ms']:.0f}ms, {inf_result['tokens_per_sec']:.1f} tok/s)")
            
        except Exception as e:
            power_monitor.stop()
            result = InferenceResult(
                prompt_id=p["id"],
                prompt=p["text"],
                configuration=config_name,
                latency_ms=0, input_tokens=0, output_tokens=0, tokens_per_second=0,
                gpu_memory_mb=0, peak_gpu_memory_mb=0, avg_power_w=None,
                energy_wh=None, response="", success=False, error=str(e)
            )
            print(f"FAIL: {e}")
        
        results.append(result)
    
    # Unload
    print(f"  Unloading {config_name}...")
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
    mem_after_unload = torch.cuda.memory_allocated() // (1024**2)
    print(f"  GPU Memory after unload: {mem_after_unload} MB")
    
    return results


def main():
    print("=" * 70)
    print("PHASE 0.5: SAME-MODEL FP16 vs INT4 VALIDATION")
    print("=" * 70)
    
    # Hardware info
    print("\n[0] HARDWARE VERIFICATION")
    print("-" * 40)
    import subprocess
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    gpu_info = result.stdout.strip().split(", ")
    print(f"  GPU: {gpu_info[0]}")
    print(f"  VRAM: {gpu_info[1]} MB ({int(gpu_info[1])/1024:.1f} GB)")
    print(f"  Driver: {gpu_info[2]}")
    print(f"  PyTorch CUDA: {torch.version.cuda}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  NVML Power Monitoring: {'AVAILABLE' if NVML_AVAILABLE else 'NOT AVAILABLE'}")
    
    # Test prompts
    prompts = [
        {"id": "factual", "text": "What is the capital of France?"},
        {"id": "scientific", "text": "Explain how photosynthesis works in plants."},
        {"id": "explanation", "text": "What is the difference between a compiler and an interpreter?"},
        {"id": "summarization", "text": "Summarize the causes of climate change in 3 sentences."},
        {"id": "coding", "text": "Write a Python function to compute factorial recursively."},
        {"id": "reasoning", "text": "If all A are B, and some B are C, can we conclude some A are C? Explain."},
    ]
    
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    
    all_results = []
    
    # --- FP16 Benchmark ---
    fp16_results = benchmark_configuration(
        "FP16", load_model_fp16, model_id, prompts
    )
    all_results.extend(fp16_results)
    
    # --- INT4 Benchmark ---
    int4_results = benchmark_configuration(
        "INT4", load_model_int4, model_id, prompts
    )
    all_results.extend(int4_results)
    
    # --- Analysis ---
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    
    fp16_ok = [r for r in fp16_results if r.success]
    int4_ok = [r for r in int4_results if r.success]
    
    print(f"\nFP16 Successful: {len(fp16_ok)}/{len(fp16_results)}")
    print(f"INT4 Successful: {len(int4_ok)}/{len(int4_results)}")
    
    if fp16_ok and int4_ok:
        # Calculate averages
        def avg(lst, key):
            vals = [getattr(r, key) for r in lst if getattr(r, key) is not None]
            return sum(vals) / len(vals) if vals else None
        
        print(f"\n{'Metric':<30} {'FP16':>15} {'INT4':>15} {'Ratio (INT4/FP16)':>15}")
        print("-" * 75)
        
        # VRAM (peak during inference)
        fp16_vram = avg(fp16_ok, "peak_gpu_memory_mb")
        int4_vram = avg(int4_ok, "peak_gpu_memory_mb")
        print(f"{'Peak VRAM (MB)':<30} {fp16_vram:>15.0f} {int4_vram:>15.0f} {int4_vram/fp16_vram:>15.2f}x")
        
        # Latency
        fp16_lat = avg(fp16_ok, "latency_ms")
        int4_lat = avg(int4_ok, "latency_ms")
        print(f"{'Avg Latency (ms)':<30} {fp16_lat:>15.0f} {int4_lat:>15.0f} {int4_lat/fp16_lat:>15.2f}x")
        
        # Throughput
        fp16_tps = avg(fp16_ok, "tokens_per_second")
        int4_tps = avg(int4_ok, "tokens_per_second")
        print(f"{'Avg Tokens/sec':<30} {fp16_tps:>15.1f} {int4_tps:>15.1f} {int4_tps/fp16_tps:>15.2f}x")
        
        # Power
        fp16_pwr = avg(fp16_ok, "avg_power_w")
        int4_pwr = avg(int4_ok, "avg_power_w")
        if fp16_pwr and int4_pwr:
            print(f"{'Avg Power (W)':<30} {fp16_pwr:>15.2f} {int4_pwr:>15.2f} {int4_pwr/fp16_pwr:>15.2f}x")
        else:
            print(f"{'Avg Power (W)':<30} {'N/A':>15} {'N/A':>15} {'N/A':>15}")
        
        # Energy per request
        fp16_eng = avg(fp16_ok, "energy_wh")
        int4_eng = avg(int4_ok, "energy_wh")
        if fp16_eng and int4_eng:
            print(f"{'Energy/request (Wh)':<30} {fp16_eng:>15.6f} {int4_eng:>15.6f} {int4_eng/fp16_eng:>15.2f}x")
        else:
            print(f"{'Energy/request (Wh)':<30} {'N/A':>15} {'N/A':>15} {'N/A':>15}")
        
        # Output quality comparison (first 100 chars)
        print("\n--- OUTPUT COMPARISON (first 100 chars) ---")
        for p in prompts:
            fp16_r = next((r for r in fp16_ok if r.prompt_id == p["id"]), None)
            int4_r = next((r for r in int4_ok if r.prompt_id == p["id"]), None)
            if fp16_r and int4_r:
                print(f"\n{p['id'].upper()}:")
                print(f"  FP16: {fp16_r.response[:100]}...")
                print(f"  INT4: {int4_r.response[:100]}...")
    
    # Save detailed results
    output = {
        "hardware": {
            "gpu": gpu_info[0],
            "vram_mb": int(gpu_info[1]),
            "driver": gpu_info[2],
            "cuda_version": torch.version.cuda,
            "torch_version": torch.__version__,
            "nvml_power_supported": NVML_AVAILABLE
        },
        "model": model_id,
        "prompts": prompts,
        "results": [asdict(r) for r in all_results],
        "summary": {
            "fp16_success_rate": len(fp16_ok) / len(fp16_results) if fp16_results else 0,
            "int4_success_rate": len(int4_ok) / len(int4_results) if int4_results else 0,
            "both_usable": len(fp16_ok) > 0 and len(int4_ok) > 0
        }
    }
    
    with open("phase0_5_validation_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n\nDetailed results saved to phase0_5_validation_results.json")
    
    # Final assessment
    print("\n" + "=" * 70)
    print("PHASE 0.5 ASSESSMENT")
    print("=" * 70)
    
    fp16_works = len(fp16_ok) == len(prompts)
    int4_works = len(int4_ok) == len(prompts)
    both_usable = fp16_works and int4_works
    energy_measurable = NVML_AVAILABLE and any(r.avg_power_w for r in all_results if r.success)
    
    print(f"\n1. FP16 works reliably: {'YES' if fp16_works else 'PARTIAL/NO'} ({len(fp16_ok)}/{len(prompts)} prompts)")
    print(f"2. INT4 works reliably: {'YES' if int4_works else 'PARTIAL/NO'} ({len(int4_ok)}/{len(prompts)} prompts)")
    print(f"3. Both produce usable outputs: {'YES' if both_usable else 'NO'}")
    print(f"4. Energy/power measurable via NVML: {'YES' if energy_measurable else 'NO'}")
    print(f"5. Same-model comparison defensible: {'YES' if both_usable else 'NO'}")
    print(f"6. Recommend Transformers as primary backend: {'YES' if both_usable else 'NO (use Ollama fallback)'}")
    
    if both_usable:
        vram_savings = (1 - int4_vram/fp16_vram) * 100 if fp16_vram and int4_vram else 0
        print(f"\n  VRAM savings with INT4: ~{vram_savings:.0f}%")
        print(f"  Latency ratio (INT4/FP16): {int4_lat/fp16_lat:.2f}x" if fp16_lat and int4_lat else "")
        print(f"  Throughput ratio (INT4/FP16): {int4_tps/fp16_tps:.2f}x" if fp16_tps and int4_tps else "")
    
    print("\nMEMORY MANAGEMENT NOTES:")
    print("  - FP16: Loads fully on GPU (~3GB), no device_map needed")
    print("  - INT4: Uses bitsandbytes device_map='auto', may offload some layers")
    print("  - Must unload + torch.cuda.empty_cache() between configurations")
    print("  - Sequential loading works; simultaneous loading NOT possible on 4GB")
    
    return all_results


if __name__ == "__main__":
    main()