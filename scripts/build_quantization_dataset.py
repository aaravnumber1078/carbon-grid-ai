"""
Phase B1: Build Quantization-Safety Dataset.

Runs inference with FP16 and INT4 for each prompt, evaluates quality,
and creates the experimental dataset.
"""

import sys
import json
import time
import torch
import gc
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))

from carbongrid.evaluation.quality_v2 import evaluate_pair_v2, get_evaluator_v2, calibrate_threshold
from carbongrid.evaluation.features import extract_features, get_feature_names
from carbongrid.carbon.manager import create_carbon_manager, EnergyMeasurement

# NVML for power measurement
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


@dataclass
class DatasetRecord:
    """Single record in the quantization-safety dataset."""
    prompt_id: str
    prompt: str
    task_type: str
    subcategory: str
    complexity_level: int  # 1=easy, 2=moderate, 3=hard, 4=very hard
    expected_difficulty: str
    reference_answer: str
    
    # Input
    input_tokens: int
    
    # FP16 measurements
    fp16_output: str
    fp16_output_tokens: int
    fp16_latency_ms: float
    fp16_energy_wh: Optional[float]
    fp16_power_w: Optional[float]
    fp16_gpu_memory_mb: int
    
    # INT4 measurements
    int4_output: str
    int4_output_tokens: int
    int4_latency_ms: float
    int4_energy_wh: Optional[float]
    int4_power_w: Optional[float]
    int4_gpu_memory_mb: int
    
    # Quality evaluation
    quality_score_fp16: float
    quality_score_int4: float
    quality_difference: float
    evaluation_method: str
    evaluation_confidence: float
    quality_threshold: float
    safe_to_quantize: bool
    
    # Features (pre-inference)
    features: Dict[str, Any]
    
    # Metadata
    model_name: str
    timestamp: str
    measurement_status: str
    co2_fp16_g: Optional[float]
    co2_int4_g: Optional[float]
    carbon_intensity_used: Optional[float]


class NVMLPowerMonitor:
    """GPU power monitoring using NVML."""
    
    def __init__(self, device_index: int = 0, sample_interval_ms: float = 20.0):
        self.device_index = device_index
        self.sample_interval = sample_interval_ms / 1000.0
        self.handle = None
        self.sampling = False
        self.samples: List[float] = []
        self._thread = None
        
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
                # Test
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
                    self.samples.append(power_mw / 1000.0)
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
            return {"avg_power_w": None, "energy_wh": None, "samples": 0, "duration_s": 0}
        
        self.sampling = False
        if self._thread:
            self._thread.join(timeout=1.0)
        
        if not self.samples:
            return {"avg_power_w": None, "energy_wh": None, "samples": 0, "duration_s": 0}
        
        avg_power = sum(self.samples) / len(self.samples)
        duration_s = len(self.samples) * self.sample_interval
        energy_wh = avg_power * (duration_s / 3600.0)
        
        return {
            "avg_power_w": round(avg_power, 3),
            "energy_wh": round(energy_wh, 6),
            "samples": len(self.samples),
            "duration_s": round(duration_s, 3)
        }


def load_model_fp16(model_id: str):
    """Load model in FP16 without device_map."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map=None,
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
        device_map="auto",
        low_cpu_mem_usage=True
    )
    model.eval()
    return model, tokenizer


def run_inference(model, tokenizer, prompt: str, max_new_tokens: int = 256, power_monitor=None) -> Dict[str, Any]:
    """Run inference and return metrics."""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_tokens = inputs.input_ids.shape[1]
    
    if power_monitor:
        power_monitor.start()
    
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
    
    if power_monitor:
        power_data = power_monitor.stop()
    else:
        power_data = {"avg_power_w": None, "energy_wh": None}
    
    # Handle output
    if hasattr(outputs, 'sequences'):
        sequences = outputs.sequences
    else:
        sequences = outputs[0] if isinstance(outputs, tuple) else outputs
    
    output_tokens = sequences[0].shape[0] - input_tokens
    new_tokens = sequences[0][input_tokens:]
    
    from transformers import AutoTokenizer
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    # GPU memory
    gpu_mem = torch.cuda.memory_allocated() // (1024**2)
    peak_mem = torch.cuda.max_memory_allocated() // (1024**2)
    torch.cuda.reset_peak_memory_stats()
    
    return {
        "latency_ms": latency_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tokens_per_sec": output_tokens / (latency_ms / 1000) if latency_ms > 0 else 0,
        "response": response,
        "avg_power_w": power_data.get("avg_power_w"),
        "energy_wh": power_data.get("energy_wh"),
        "gpu_memory_mb": gpu_mem,
        "peak_gpu_memory_mb": peak_mem
    }


def run_config_inference(model_id: str, config: str, prompts: List[Dict], max_new_tokens: int = 256) -> List[Dict]:
    """Run inference for all prompts with a specific configuration."""
    print(f"\n{'='*60}")
    print(f"Loading {config} model...")
    print(f"{'='*60}")
    
    if config == "FP16":
        model, tokenizer = load_model_fp16(model_id)
    elif config == "INT4":
        model, tokenizer = load_model_int4(model_id)
    else:
        raise ValueError(f"Unknown config: {config}")
    
    print(f"  GPU Memory after load: {torch.cuda.memory_allocated() // (1024**2)} MB")
    
    power_monitor = NVMLPowerMonitor(sample_interval_ms=20.0)
    
    results = []
    for i, p in enumerate(prompts):
        print(f"  [{i+1}/{len(prompts)}] {p['prompt_id']} ({p['task_type']})...", end=" ", flush=True)
        
        try:
            result = run_inference(model, tokenizer, p["prompt"], max_new_tokens, power_monitor)
            result["prompt_id"] = p["prompt_id"]
            results.append(result)
            print(f"OK ({result['latency_ms']:.0f}ms, {result['tokens_per_sec']:.1f} tok/s)")
        except Exception as e:
            print(f"FAIL: {e}")
            results.append({
                "prompt_id": p["prompt_id"],
                "latency_ms": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "tokens_per_sec": 0,
                "response": "",
                "avg_power_w": None,
                "energy_wh": None,
                "gpu_memory_mb": 0,
                "peak_gpu_memory_mb": 0,
                "error": str(e)
            })
    
    # Unload
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    print(f"  GPU Memory after unload: {torch.cuda.memory_allocated() // (1024**2)} MB")
    
    return results


def main():
    print("=" * 70)
    print("PHASE B1: BUILD QUANTIZATION-SAFETY DATASET (50 PROMPTS)")
    print("=" * 70)
    
    # Load prompts
    with open("data/raw/b2_prompts.json", 'r') as f:
        prompts_data = json.load(f)
    
    prompts = [{"prompt_id": p["prompt_id"], "prompt": p["prompt"], "task_type": p["task_type"], 
                "subcategory": p["subcategory"], "complexity_level": p.get("complexity_level", 0),
                "expected_difficulty": p.get("expected_difficulty", ""), 
                "reference_answer": p.get("reference_answer", "")} for p in prompts_data]
    
    print(f"Loaded {len(prompts)} prompts")
    
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    quality_threshold = 0.7
    
    # Run FP16
    fp16_results = run_config_inference(model_id, "FP16", prompts, max_new_tokens=256)
    
    # Run INT4
    int4_results = run_config_inference(model_id, "INT4", prompts, max_new_tokens=256)
    
    # Create carbon manager for CO2 estimation
    carbon_manager = create_carbon_manager("live")
    current_carbon = carbon_manager.get_current_carbon("national")
    carbon_intensity = current_carbon.carbon_intensity_gco2_per_kwh if current_carbon else 200.0
    print(f"\nUsing carbon intensity: {carbon_intensity:.1f} gCO2/kWh")
    
    # Build dataset records
    print("\n" + "=" * 60)
    print("EVALUATING QUALITY")
    print("=" * 60)
    
    records = []
    safe_count = 0
    unsafe_count = 0
    failed_count = 0
    
    for p in prompts:
        pid = p["prompt_id"]
        fp16 = next((r for r in fp16_results if r["prompt_id"] == pid), None)
        int4 = next((r for r in int4_results if r["prompt_id"] == pid), None)
        
        if not fp16 or not int4 or "error" in fp16 or "error" in int4:
            failed_count += 1
            continue
        
        # Quality evaluation
        eval_result = evaluate_pair_v2(
            fp16["response"],
            int4["response"],
            p["task_type"],
            p["reference_answer"],
            threshold=quality_threshold
        )
        
        safe = eval_result.passed
        if safe:
            safe_count += 1
        else:
            unsafe_count += 1
        
        # CO2 estimation
        co2_fp16 = None
        co2_int4 = None
        if fp16.get("energy_wh") and carbon_intensity:
            co2_fp16 = (fp16["energy_wh"] / 1000) * carbon_intensity
        if int4.get("energy_wh") and carbon_intensity:
            co2_int4 = (int4["energy_wh"] / 1000) * carbon_intensity
        
        # Extract features
        feat = extract_features(p["prompt"], p["task_type"])
        
        record = DatasetRecord(
            prompt_id=pid,
            prompt=p["prompt"],
            task_type=p["task_type"],
            subcategory=p["subcategory"],
            complexity_level=p.get("complexity_level", 0),
            expected_difficulty=p["expected_difficulty"],
            reference_answer=p["reference_answer"],
            input_tokens=fp16["input_tokens"],
            fp16_output=fp16["response"],
            fp16_output_tokens=fp16["output_tokens"],
            fp16_latency_ms=fp16["latency_ms"],
            fp16_energy_wh=fp16.get("energy_wh"),
            fp16_power_w=fp16.get("avg_power_w"),
            fp16_gpu_memory_mb=fp16["gpu_memory_mb"],
            int4_output=int4["response"],
            int4_output_tokens=int4["output_tokens"],
            int4_latency_ms=int4["latency_ms"],
            int4_energy_wh=int4.get("energy_wh"),
            int4_power_w=int4.get("avg_power_w"),
            int4_gpu_memory_mb=int4["gpu_memory_mb"],
            quality_score_fp16=1.0,  # FP16 is reference
            quality_score_int4=eval_result.score,
            quality_difference=eval_result.score - 1.0,
            evaluation_method=eval_result.method,
            evaluation_confidence=eval_result.confidence,
            quality_threshold=quality_threshold,
            safe_to_quantize=safe,
            features=feat.to_dict(),
            model_name="Qwen2.5-1.5B-Instruct",
            timestamp=datetime.now(timezone.utc).isoformat(),
            measurement_status="complete" if "error" not in fp16 and "error" not in int4 else "partial",
            co2_fp16_g=co2_fp16,
            co2_int4_g=co2_int4,
            carbon_intensity_used=carbon_intensity
        )
        
        records.append(record)
        status = "SAFE" if safe else "UNSAFE"
        print(f"  {pid}: {status} (score={eval_result.score:.3f}, method={eval_result.method})")
    
    # Save dataset
    output_path = Path("data/processed/b2_dataset.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump([asdict(r) for r in records], f, indent=2, default=str)
    
    print(f"\nDataset saved to {output_path}")
    print(f"Records: {len(records)}")
    print(f"  Safe for INT4: {safe_count}")
    print(f"  Unsafe for INT4: {unsafe_count}")
    print(f"  Failed: {failed_count}")
    
    # Calibration
    print("\n" + "=" * 60)
    print("THRESHOLD CALIBRATION")
    print("=" * 60)
    
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    calibration = calibrate_threshold([asdict(r) for r in records], thresholds)
    print(f"\n{'Threshold':>12} {'Safe':>6} {'Unsafe':>8} {'Total':>6} {'Safe%':>8}")
    for t, v in calibration.items():
        print(f"{t:>12.2f} {v['safe']:>6} {v['unsafe']:>8} {v['total']:>6} {v['safe_pct']:>7.1f}%")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    
    # By task type
    from collections import Counter
    task_safe = Counter()
    task_total = Counter()
    
    for r in records:
        task_total[r.task_type] += 1
        if r.safe_to_quantize:
            task_safe[r.task_type] += 1
    
    print("\nBy Task Type:")
    for task in sorted(task_total.keys()):
        total = task_total[task]
        safe = task_safe[task]
        print(f"  {task:20} Total: {total:2}  Safe: {safe:2}  Unsafe: {total-safe:2}  Safe%: {safe/total*100:.0f}%")
    
    # By complexity
    print("\nBy Complexity Level:")
    for level in range(1, 5):
        level_records = [r for r in records if r.complexity_level == level]
        if level_records:
            total = len(level_records)
            safe = sum(1 for r in level_records if r.safe_to_quantize)
            print(f"  Level {level}: Total: {total:2}  Safe: {safe:2}  Unsafe: {total-safe:2}  Safe%: {safe/total*100:.0f}%")
    
    # Task x Complexity
    print("\nTask x Complexity Safe%:")
    for task in sorted(task_total.keys()):
        for level in range(1, 5):
            subset = [r for r in records if r.task_type == task and r.complexity_level == level]
            if subset:
                total = len(subset)
                safe = sum(1 for r in subset if r.safe_to_quantize)
                print(f"  {task:20} Level {level}: {safe}/{total} ({safe/total*100:.0f}%)")
    
    # Latency/Energy comparison
    if records:
        fp16_lat = sum(r.fp16_latency_ms for r in records) / len(records)
        int4_lat = sum(r.int4_latency_ms for r in records) / len(records)
        fp16_eng = sum(r.fp16_energy_wh for r in records if r.fp16_energy_wh) / max(1, sum(1 for r in records if r.fp16_energy_wh))
        int4_eng = sum(r.int4_energy_wh for r in records if r.int4_energy_wh) / max(1, sum(1 for r in records if r.int4_energy_wh))
        fp16_pwr = sum(r.fp16_power_w for r in records if r.fp16_power_w) / max(1, sum(1 for r in records if r.fp16_power_w))
        int4_pwr = sum(r.int4_power_w for r in records if r.int4_power_w) / max(1, sum(1 for r in records if r.int4_power_w))
        
        print(f"\nAverage Latency: FP16={fp16_lat:.0f}ms  INT4={int4_lat:.0f}ms  Ratio={int4_lat/fp16_lat:.2f}x")
        print(f"Average Energy:  FP16={fp16_eng:.6f}Wh  INT4={int4_eng:.6f}Wh  Ratio={int4_eng/fp16_eng:.2f}x")
        print(f"Average Power:   FP16={fp16_pwr:.1f}W  INT4={int4_pwr:.1f}W  Ratio={int4_pwr/fp16_pwr:.2f}x")
        
        # CO2
        co2_fp16 = sum(r.co2_fp16_g for r in records if r.co2_fp16_g) / max(1, sum(1 for r in records if r.co2_fp16_g))
        co2_int4 = sum(r.co2_int4_g for r in records if r.co2_int4_g) / max(1, sum(1 for r in records if r.co2_int4_g))
        print(f"Est. CO2:        FP16={co2_fp16:.4f}g  INT4={co2_int4:.4f}g  Diff={co2_int4-co2_fp16:+.4f}g")
    
    # Examples
    print("\nExample SAFE cases:")
    for r in records:
        if r.safe_to_quantize:
            print(f"  {r.prompt_id} ({r.task_type}, L{r.complexity_level}): score={r.quality_score_int4:.3f}, conf={r.evaluation_confidence:.2f}")
            break
    
    print("\nExample UNSAFE cases:")
    for r in records:
        if not r.safe_to_quantize:
            print(f"  {r.prompt_id} ({r.task_type}, L{r.complexity_level}): score={r.quality_score_int4:.3f}, conf={r.evaluation_confidence:.2f}")
            break
    
    # Limitations
    print("\n" + "=" * 60)
    print("EVALUATION LIMITATIONS")
    print("=" * 60)
    print("""
1. Factual: Entity matching + semantic similarity (may miss synonyms)
2. Extraction: Exact field/value comparison (high confidence)
3. Summarization: Semantic + length + concept coverage (proxy)
4. Code: Syntax + AST + execution (where testable)
5. Reasoning: Semantic + answer key + logical structure
6. Creative: Semantic similarity only (explicit proxy, low confidence)
7. Scientific: Semantic + concept coverage
8. Explanation: Semantic + structure + concepts

FP16 is reference quality - NOT ground truth.
Threshold is configurable (default varies by task).
Energy includes idle power, 20ms sampling.
CO2 is ESTIMATED from MEASURED energy × PROVIDER intensity.
""")


if __name__ == "__main__":
    main()