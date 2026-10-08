#!/usr/bin/env python3
"""
Phase G.1.2 Pilot Measurement Script

Measures FP16 and INT4 energy/latency for 25 pilot prompts with 3 repetitions each.
Uses the existing carbongrid.inference.provider.InferenceProvider which handles:
- Sequential model loading/unloading
- NVML power monitoring
- GPU synchronization
"""

import sys
import time
import json
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from enum import Enum

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# Use the existing provider
from carbongrid.inference.provider import InferenceProvider, Precision, InferenceMetrics


class MeasurementSource(Enum):
    MEASURED = "measured"
    ESTIMATED = "estimated"
    FAILED = "failed"


@dataclass
class PilotRun:
    """Single inference run result for pilot."""
    run_id: str
    prompt_id: str
    configuration: str
    repetition: int
    config_order: int  # 0 = FP16 first, 1 = INT4 first
    success: bool
    latency_ms: float
    input_tokens: int
    output_tokens: int
    tokens_per_second: float
    gpu_memory_mb: int
    peak_gpu_memory_mb: int
    avg_power_w: Optional[float]
    energy_wh: Optional[float]
    measurement_source: str
    power_samples: int
    power_duration_s: float
    response: str
    error: Optional[str]
    timestamp: str


def load_pilot_prompts() -> List[Dict[str, Any]]:
    """Load the 25 pilot prompts."""
    with open(Path("C:/CARBON GRID AI/data/evaluation/phase_g/pilot_prompts.json")) as f:
        data = json.load(f)
    return data["pilot_prompts"]


def run_pilot():
    """Run the full pilot measurement."""
    print("=" * 70)
    print("PHASE G.1.2 PILOT MEASUREMENT")
    print("=" * 70)

    prompts = load_pilot_prompts()
    print(f"\nLoaded {len(prompts)} pilot prompts")

    provider = InferenceProvider()
    all_runs: List[PilotRun] = []
    run_counter = 0

    # Warm-up prompt (not measured)
    warmup_prompt = "Hello, how are you?"

    for prompt_idx, prompt_data in enumerate(prompts):
        prompt_id = prompt_data["prompt_id"]
        prompt_text = prompt_data["prompt_text"]
        max_tokens = prompt_data.get("max_tokens_requested", 200)
        category = prompt_data["category"]

        print(f"\n[{prompt_idx+1}/{len(prompts)}] {prompt_id} ({category})")
        print(f"  Prompt: {prompt_text[:100]}...")
        print(f"  Max tokens: {max_tokens}")

        # Randomize configuration order for this prompt
        configs = [Precision.FP16, Precision.INT4]
        random.shuffle(configs)
        config_order = 0 if configs[0] == Precision.FP16 else 1

        for config_idx, precision in enumerate(configs):
            print(f"\n  Config {config_idx+1}/2: {precision.value.upper()}")

            # Warm-up (3 runs, discarded)
            print(f"  Warm-up ({precision.value})...")
            for i in range(3):
                _ = provider.run_inference(warmup_prompt, 20, precision)
            print(f"    Warm-up complete")

            # Thermal stabilization
            if config_idx > 0:
                print(f"  Thermal stabilization (30s)...")
                time.sleep(30)

            # 3 measured repetitions
            for rep in range(3):
                run_counter += 1
                run_id = f"pilot_{run_counter:04d}"

                print(f"    Rep {rep+1}/3...", end=" ", flush=True)
                metrics: InferenceMetrics = provider.run_inference(prompt_text, max_tokens, precision)

                run = PilotRun(
                    run_id=run_id,
                    prompt_id=prompt_id,
                    configuration=precision.value,
                    repetition=rep + 1,
                    config_order=config_order,
                    success=metrics.success,
                    latency_ms=metrics.latency_ms,
                    input_tokens=metrics.input_tokens,
                    output_tokens=metrics.output_tokens,
                    tokens_per_second=metrics.tokens_per_second,
                    gpu_memory_mb=metrics.gpu_memory_mb,
                    peak_gpu_memory_mb=metrics.peak_gpu_memory_mb,
                    avg_power_w=metrics.avg_power_w,
                    energy_wh=metrics.energy_wh,
                    measurement_source=MeasurementSource.MEASURED.value if metrics.energy_wh else MeasurementSource.ESTIMATED.value,
                    power_samples=0,  # Provider doesn't expose sample count directly
                    power_duration_s=0.0,
                    response=metrics.response,
                    error=metrics.error,
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                )

                all_runs.append(run)

                if run.success:
                    energy_str = f"{run.energy_wh:.6f} Wh" if run.energy_wh else "N/A"
                    power_str = f"{run.avg_power_w:.1f} W" if run.avg_power_w else "N/A"
                    print(f"lat={run.latency_ms:.0f}ms tok/s={run.tokens_per_second:.1f} energy={energy_str} power={power_str}")
                else:
                    print(f"FAILED: {run.error}")

        # Unload after both configs for this prompt
        provider.unload()
        time.sleep(5)  # Extra cooling

    # Save results
    output = {
        "metadata": {
            "phase": "G.1.2_pilot",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_prompts": len(prompts),
            "repetitions_per_config": 3,
            "total_runs": len(all_runs),
            "model": "Qwen/Qwen2.5-1.5B-Instruct",
            "device": "RTX 2050 (4GB)",
            "power_sample_interval_ms": 20.0,
            "warmup_runs_per_config": 3,
            "thermal_stabilization_s": 30,
            "gpu_sync": True,
            "nvml_power_monitoring": True,
            "config_order_randomized": True,
            "model_unloaded_between_configs": True,
            "model_loading_energy_excluded": True,
        },
        "runs": [asdict(r) for r in all_runs],
    }

    output_path = Path("C:/CARBON GRID AI/data/evaluation/phase_g/pilot_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n\n{'=' * 70}")
    print(f"PILOT COMPLETE - Results saved to {output_path}")
    print(f"{'=' * 70}")

    # Quick summary
    successful = [r for r in all_runs if r.success]
    failed = [r for r in all_runs if not r.success]
    fp16_runs = [r for r in successful if r.configuration == "fp16"]
    int4_runs = [r for r in successful if r.configuration == "int4"]

    print(f"\nTotal runs: {len(all_runs)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"FP16 runs: {len(fp16_runs)}")
    print(f"INT4 runs: {len(int4_runs)}")

    if fp16_runs and int4_runs:
        fp16_energy = [r.energy_wh for r in fp16_runs if r.energy_wh]
        int4_energy = [r.energy_wh for r in int4_runs if r.energy_wh]
        if fp16_energy and int4_energy:
            print(f"\nFP16 energy: mean={sum(fp16_energy)/len(fp16_energy):.6f} Wh, n={len(fp16_energy)}")
            print(f"INT4 energy: mean={sum(int4_energy)/len(int4_energy):.6f} Wh, n={len(int4_energy)}")

    return all_runs


if __name__ == "__main__":
    run_pilot()