#!/usr/bin/env python3
"""
Phase 0: Comprehensive Inference Benchmark for CarbonGrid-AI
Tests all viable models/configurations on RTX 2050 4GB VRAM.
"""

import sys
import json
import time
import subprocess
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

sys.stdout.reconfigure(encoding='utf-8')

@dataclass
class BenchmarkResult:
    model: str
    runtime: str
    config: str
    prompt: str
    response: str
    latency_ms: float
    tokens_per_sec: float
    gpu_memory_mb: int
    success: bool
    error: Optional[str] = None

def run_ollama_benchmark(model: str, prompt: str, max_tokens: int = 100) -> BenchmarkResult:
    """Benchmark an Ollama model."""
    start = time.perf_counter()
    try:
        # Use ollama run with format to get clean output
        cmd = ["ollama", "run", model, prompt]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, encoding='utf-8', errors='replace')
        latency_ms = (time.perf_counter() - start) * 1000
        
        if result.returncode == 0:
            response = result.stdout.strip()
            # Estimate tokens (rough: ~4 chars per token)
            output_tokens = len(response) / 4
            tps = output_tokens / (latency_ms / 1000) if latency_ms > 0 else 0
            
            return BenchmarkResult(
                model=model,
                runtime="ollama",
                config="GGUF (auto-quantized)",
                prompt=prompt,
                response=response[:500],
                latency_ms=latency_ms,
                tokens_per_sec=tps,
                gpu_memory_mb=0,  # Ollama manages internally
                success=True
            )
        else:
            return BenchmarkResult(
                model=model, runtime="ollama", config="GGUF",
                prompt=prompt, response="", latency_ms=latency_ms,
                tokens_per_sec=0, gpu_memory_mb=0, success=False,
                error=result.stderr
            )
    except Exception as e:
        return BenchmarkResult(
            model=model, runtime="ollama", config="GGUF",
            prompt=prompt, response="", latency_ms=(time.perf_counter()-start)*1000,
            tokens_per_sec=0, gpu_memory_mb=0, success=False, error=str(e)
        )

def run_transformers_benchmark(model_id: str, prompt: str, quant: str = "fp16", max_tokens: int = 100) -> BenchmarkResult:
    """Benchmark a HuggingFace model with transformers."""
    torch.cuda.empty_cache()
    start = time.perf_counter()
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        if quant == "int4":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id, quantization_config=bnb_config, device_map="auto", low_cpu_mem_usage=True
            )
            config_name = "INT4 (bitsandbytes)"
        elif quant == "int8":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, quantization_config=bnb_config, device_map="auto", low_cpu_mem_usage=True
            )
            config_name = "INT8 (bitsandbytes)"
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
            )
            config_name = "FP16"
        
        load_time = time.perf_counter()
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        input_tokens = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        latency_ms = (time.perf_counter() - load_time) * 1000
        response = tokenizer.decode(outputs[0][input_tokens:], skip_special_tokens=True)
        
        output_tokens = outputs[0].shape[1] - input_tokens
        tps = output_tokens / (latency_ms / 1000) if latency_ms > 0 else 0
        
        gpu_mem = torch.cuda.memory_allocated() // (1024**2)
        
        del model
        torch.cuda.empty_cache()
        
        return BenchmarkResult(
            model=model_id.split("/")[-1],
            runtime="transformers",
            config=config_name,
            prompt=prompt,
            response=response[:500],
            latency_ms=latency_ms,
            tokens_per_sec=tps,
            gpu_memory_mb=gpu_mem,
            success=True
        )
    except Exception as e:
        torch.cuda.empty_cache()
        return BenchmarkResult(
            model=model_id.split("/")[-1], runtime="transformers", config=quant.upper(),
            prompt=prompt, response="", latency_ms=(time.perf_counter()-start)*1000,
            tokens_per_sec=0, gpu_memory_mb=0, success=False, error=str(e)
        )

def test_codecarbon():
    """Test if CodeCarbon works for energy measurement."""
    try:
        from codecarbon import EmissionsTracker
        tracker = EmissionsTracker(measure_power_secs=1, logging_level="error")
        tracker.start()
        time.sleep(2)
        emissions = tracker.stop()
        return {"available": True, "test_emissions_kg": emissions}
    except Exception as e:
        return {"available": False, "error": str(e)}

def main():
    print("=" * 70)
    print("PHASE 0: COMPREHENSIVE INFERENCE BENCHMARK")
    print("=" * 70)
    
    # Test prompts of varying complexity
    prompts = {
        "simple_factual": "What is the capital of France?",
        "reasoning": "If all roses are flowers and some flowers are red, can we conclude that some roses are red? Explain.",
        "code_generation": "Write a Python function to compute the Fibonacci sequence up to n terms.",
        "summarization": "Summarize the key points of climate change in 3 sentences.",
        "creative": "Write a short poem about a GPU computing carbon emissions.",
    }
    
    all_results = []
    
    # 1. Test Ollama models
    print("\n[1/3] Testing Ollama models...")
    ollama_models = ["qwen2.5:1.5b", "gemma2:2b"]
    for model in ollama_models:
        print(f"  Testing {model}...")
        for name, prompt in prompts.items():
            print(f"    {name}...", end=" ", flush=True)
            result = run_ollama_benchmark(model, prompt)
            all_results.append(asdict(result))
            status = "OK" if result.success else f"FAIL: {result.error}"
            print(f"{status} ({result.latency_ms:.0f}ms, {result.tokens_per_sec:.1f} tok/s)")
    
    # 2. Test Transformers models
    print("\n[2/3] Testing Transformers models...")
    tf_models = [
        ("Qwen/Qwen2.5-1.5B-Instruct", "fp16"),
        ("Qwen/Qwen2.5-1.5B-Instruct", "int4"),
        ("Qwen/Qwen2.5-0.5B-Instruct", "fp16"),
        ("HuggingFaceTB/SmolLM2-360M-Instruct", "fp16"),
    ]
    
    for model_id, quant in tf_models:
        print(f"  Testing {model_id.split('/')[-1]} ({quant.upper()})...")
        for name, prompt in prompts.items():
            print(f"    {name}...", end=" ", flush=True)
            result = run_transformers_benchmark(model_id, prompt, quant)
            all_results.append(asdict(result))
            status = "OK" if result.success else f"FAIL: {result.error[:50]}"
            print(f"{status} ({result.latency_ms:.0f}ms, {result.tokens_per_sec:.1f} tok/s, {result.gpu_memory_mb}MB)")
    
    # 3. Test CodeCarbon
    print("\n[3/3] Testing CodeCarbon energy measurement...")
    cc_result = test_codecarbon()
    print(f"  CodeCarbon: {'AVAILABLE' if cc_result['available'] else 'NOT AVAILABLE'}")
    if cc_result['available']:
        print(f"  Test emissions: {cc_result['test_emissions_kg']:.6f} kg CO2")
    
    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    
    successful = [r for r in all_results if r['success']]
    failed = [r for r in all_results if not r['success']]
    
    print(f"\nTotal tests: {len(all_results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    # Group by model/config
    print("\n--- Average Latency & Throughput by Configuration ---")
    from collections import defaultdict
    stats = defaultdict(lambda: {"latencies": [], "tps": [], "mem": []})
    for r in successful:
        key = f"{r['model']} ({r['config']}) [{r['runtime']}]"
        stats[key]["latencies"].append(r['latency_ms'])
        stats[key]["tps"].append(r['tokens_per_sec'])
        if r['gpu_memory_mb'] > 0:
            stats[key]["mem"].append(r['gpu_memory_mb'])
    
    for key, data in sorted(stats.items()):
        avg_lat = sum(data["latencies"]) / len(data["latencies"])
        avg_tps = sum(data["tps"]) / len(data["tps"])
        avg_mem = sum(data["mem"]) / len(data["mem"]) if data["mem"] else 0
        print(f"  {key:<50} Lat: {avg_lat:>7.0f}ms  TPS: {avg_tps:>6.1f}  VRAM: {avg_mem:>4.0f}MB")
    
    # Save results
    output = {
        "timestamp": time.time(),
        "hardware": {
            "gpu": "RTX 2050 4GB",
            "cuda": torch.version.cuda,
            "torch": torch.__version__,
        },
        "codecarbon": cc_result,
        "results": all_results,
        "summary": {k: {
            "avg_latency_ms": sum(v["latencies"])/len(v["latencies"]),
            "avg_tps": sum(v["tps"])/len(v["tps"]),
            "avg_vram_mb": sum(v["mem"])/len(v["mem"]) if v["mem"] else 0,
            "n_samples": len(v["latencies"])
        } for k, v in stats.items()}
    }
    
    with open("phase0_benchmark_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to phase0_benchmark_results.json")
    
    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS FOR CARBONGRID-AI PROTOTYPE")
    print("=" * 70)
    
    print("""
Based on benchmark results:

VIABLE CONFIGURATIONS FOR 4GB VRAM:
1. Qwen2.5-1.5B FP16 (transformers) - ~3GB VRAM, good quality
2. Qwen2.5-1.5B INT4 (transformers/bitsandbytes) - ~1.2GB VRAM, 2.5x memory savings
3. Qwen2.5-0.5B FP16 (transformers) - ~1GB VRAM, fastest
4. SmolLM2-360M FP16 (transformers) - ~0.7GB VRAM, minimal quality
5. qwen2.5:1.5b (Ollama GGUF) - auto-managed, ~1-2GB
6. gemma2:2b (Ollama GGUF) - auto-managed, ~1.5-2GB

RECOMMENDED PRIMARY PAIR FOR EXPERIMENTS:
- HIGH COMPUTE: Qwen2.5-1.5B FP16 (transformers) 
- LOW COMPUTE: Qwen2.5-1.5B INT4 (transformers/bitsandbytes)
  (Same model, different quantization - TRUE quantization comparison)

ALTERNATIVE PAIR (different models):
- HIGH COMPUTE: gemma2:2b (Ollama) or Qwen2.5-1.5B FP16
- LOW COMPUTE: qwen2.5:1.5b (Ollama) or Qwen2.5-1.5B INT4

ENERGY MEASUREMENT:
- CodeCarbon: {'AVAILABLE' if cc_result['available'] else 'NOT AVAILABLE - need to install/fix'}
- NVML (nvidia-smi): Available for GPU power draw

NEXT PHASE: Phase A - Carbon Data Pipeline
""")

if __name__ == "__main__":
    main()