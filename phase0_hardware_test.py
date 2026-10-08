#!/usr/bin/env python3
"""
Phase 0: Hardware/Runtime/Model Compatibility Test for CarbonGrid-AI
Tests RTX 2050 6GB compatibility with various inference runtimes and models.
"""

import sys
import subprocess
import json
import platform
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

def run_cmd(cmd: List[str], timeout: int = 30) -> Dict[str, Any]:
    """Run command and return structured result."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "stdout": "", "stderr": "Timeout", "returncode": -1}
    except Exception as e:
        return {"success": False, "stdout": "", "stderr": str(e), "returncode": -1}

def check_python() -> Dict[str, Any]:
    """Check Python version and key packages."""
    info = {
        "version": sys.version,
        "executable": sys.executable,
        "platform": platform.platform()
    }
    
    packages = ["torch", "transformers", "accelerate", "bitsandbytes", "llama_cpp", "ollama"]
    for pkg in packages:
        try:
            mod = __import__(pkg)
            info[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            info[pkg] = "NOT_INSTALLED"
    
    return info

def check_nvidia() -> Dict[str, Any]:
    """Check NVIDIA GPU, driver, CUDA."""
    info = {}
    
    # nvidia-smi
    result = run_cmd(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"])
    if result["success"]:
        lines = result["stdout"].split("\n")
        if lines:
            parts = [p.strip() for p in lines[0].split(",")]
            info["gpu_name"] = parts[0]
            info["vram_mb"] = int(parts[1]) if parts[1].isdigit() else 0
            info["driver_version"] = parts[2] if len(parts) > 2 else "unknown"
    
    # nvidia-smi for CUDA version
    result = run_cmd(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader,nounits"])
    if result["success"]:
        info["compute_capability"] = result["stdout"].strip()
    
    # CUDA version from nvcc
    result = run_cmd(["nvcc", "--version"])
    if result["success"]:
        for line in result["stdout"].split("\n"):
            if "release" in line.lower():
                info["cuda_version"] = line.strip()
                break
    
    return info

def check_pytorch_cuda() -> Dict[str, Any]:
    """Check PyTorch CUDA support."""
    info = {}
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["torch_cuda_version"] = torch.version.cuda
            info["device_count"] = torch.cuda.device_count()
            info["device_name"] = torch.cuda.get_device_name(0)
            info["device_capability"] = torch.cuda.get_device_capability(0)
            
            # Memory info
            props = torch.cuda.get_device_properties(0)
            info["total_memory_mb"] = props.total_memory // (1024**2)
            info["multi_processor_count"] = props.multi_processor_count
    except Exception as e:
        info["error"] = str(e)
    return info

def check_ollama() -> Dict[str, Any]:
    """Check Ollama availability and models."""
    info = {"available": False}
    result = run_cmd(["ollama", "list"])
    if result["success"]:
        info["available"] = True
        info["models"] = result["stdout"]
    else:
        info["error"] = result["stderr"]
    
    # Check ollama serve status
    result = run_cmd(["curl", "-s", "http://localhost:11434/api/tags"], timeout=5)
    if result["success"]:
        try:
            info["api_models"] = json.loads(result["stdout"])
        except:
            info["api_models"] = "parse_error"
    return info

def check_llama_cpp() -> Dict[str, Any]:
    """Check llama.cpp availability."""
    info = {"available": False}
    # Check if llama-cpp-python is installed
    try:
        import llama_cpp
        info["available"] = True
        info["version"] = getattr(llama_cpp, "__version__", "unknown")
    except ImportError:
        info["error"] = "llama_cpp not installed"
    
    # Check for llama.cpp binary
    for cmd in [["llama-cli", "--version"], ["./llama.cpp/llama-cli", "--version"]]:
        result = run_cmd(cmd)
        if result["success"]:
            info["binary_version"] = result["stdout"]
            break
    return info

def check_disk_space() -> Dict[str, Any]:
    """Check available disk space."""
    info = {}
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        info["total_gb"] = total // (1024**3)
        info["used_gb"] = used // (1024**3)
        info["free_gb"] = free // (1024**3)
    except Exception as e:
        info["error"] = str(e)
    return info

def test_model_inference(model_id: str, runtime: str, prompt: str = "What is 2+2?") -> Dict[str, Any]:
    """Test actual inference with a model."""
    result = {"model": model_id, "runtime": runtime, "success": False}
    
    if runtime == "ollama":
        cmd = ["ollama", "run", model_id, prompt]
        r = run_cmd(cmd, timeout=120)
        result["success"] = r["success"]
        result["output"] = r["stdout"][:500] if r["success"] else r["stderr"]
        result["error"] = r["stderr"] if not r["success"] else None
    
    elif runtime == "transformers":
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            print(f"  Loading {model_id}...")
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            result["success"] = True
            result["output"] = response[:500]
            
            # Memory usage
            result["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() // (1024**2)
            result["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() // (1024**2)
            
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            result["error"] = str(e)
    
    elif runtime == "llama_cpp":
        try:
            from llama_cpp import Llama
            # This would need a GGUF model path
            result["error"] = "Requires local GGUF model path"
        except Exception as e:
            result["error"] = str(e)
    
    return result

def recommend_models(vram_gb: float) -> List[Dict[str, Any]]:
    """Recommend models based on VRAM."""
    recommendations = []
    
    if vram_gb >= 24:
        recommendations.extend([
            {"name": "Llama-3.1-8B", "size_gb": 16, "quant": "FP16", "runtime": "transformers"},
            {"name": "Llama-3.1-8B", "size_gb": 8, "quant": "INT8", "runtime": "transformers"},
            {"name": "Llama-3.1-8B", "size_gb": 5, "quant": "INT4", "runtime": "transformers"},
            {"name": "Mistral-7B", "size_gb": 14, "quant": "FP16", "runtime": "transformers"},
            {"name": "Mistral-7B", "size_gb": 7, "quant": "INT8", "runtime": "transformers"},
            {"name": "Mistral-7B", "size_gb": 4, "quant": "INT4", "runtime": "transformers"},
        ])
    elif vram_gb >= 12:
        recommendations.extend([
            {"name": "Llama-3.1-8B", "size_gb": 8, "quant": "INT8", "runtime": "transformers"},
            {"name": "Llama-3.1-8B", "size_gb": 5, "quant": "INT4", "runtime": "transformers"},
            {"name": "Mistral-7B", "size_gb": 7, "quant": "INT8", "runtime": "transformers"},
            {"name": "Mistral-7B", "size_gb": 4, "quant": "INT4", "runtime": "transformers"},
            {"name": "Phi-3-mini-4k", "size_gb": 7, "quant": "FP16", "runtime": "transformers"},
            {"name": "Phi-3-mini-4k", "size_gb": 4, "quant": "INT4", "runtime": "transformers"},
            {"name": "Gemma-2-2B", "size_gb": 4, "quant": "FP16", "runtime": "transformers"},
        ])
    elif vram_gb >= 8:
        recommendations.extend([
            {"name": "Llama-3.1-8B", "size_gb": 5, "quant": "INT4", "runtime": "transformers"},
            {"name": "Mistral-7B", "size_gb": 4, "quant": "INT4", "runtime": "transformers"},
            {"name": "Phi-3-mini-4k", "size_gb": 4, "quant": "INT4", "runtime": "transformers"},
            {"name": "Phi-3-mini-4k", "size_gb": 7, "quant": "FP16", "runtime": "transformers"},
            {"name": "Gemma-2-2B", "size_gb": 4, "quant": "FP16", "runtime": "transformers"},
            {"name": "Gemma-2-2B", "size_gb": 2, "quant": "INT4", "runtime": "transformers"},
            {"name": "Qwen2.5-1.5B", "size_gb": 3, "quant": "FP16", "runtime": "transformers"},
            {"name": "Qwen2.5-1.5B", "size_gb": 1.5, "quant": "INT4", "runtime": "transformers"},
            {"name": "SmolLM2-1.7B", "size_gb": 3.5, "quant": "FP16", "runtime": "transformers"},
        ])
    elif vram_gb >= 6:
        recommendations.extend([
            {"name": "Phi-3-mini-4k", "size_gb": 4, "quant": "INT4", "runtime": "transformers"},
            {"name": "Gemma-2-2B", "size_gb": 2, "quant": "INT4", "runtime": "transformers"},
            {"name": "Gemma-2-2B", "size_gb": 4, "quant": "FP16", "runtime": "transformers (tight)"},
            {"name": "Qwen2.5-1.5B", "size_gb": 3, "quant": "FP16", "runtime": "transformers"},
            {"name": "Qwen2.5-1.5B", "size_gb": 1.5, "quant": "INT4", "runtime": "transformers"},
            {"name": "Qwen2.5-0.5B", "size_gb": 1, "quant": "FP16", "runtime": "transformers"},
            {"name": "SmolLM2-1.7B", "size_gb": 3.5, "quant": "FP16", "runtime": "transformers (tight)"},
            {"name": "SmolLM2-360M", "size_gb": 0.8, "quant": "FP16", "runtime": "transformers"},
        ])
    else:
        recommendations.extend([
            {"name": "Qwen2.5-0.5B", "size_gb": 1, "quant": "FP16", "runtime": "transformers"},
            {"name": "SmolLM2-360M", "size_gb": 0.8, "quant": "FP16", "runtime": "transformers"},
        ])
    
    # Add Ollama variants (uses system RAM + VRAM)
    ollama_models = [
        "phi3:mini", "phi3:mini-4k", "gemma2:2b", "qwen2.5:1.5b", 
        "qwen2.5:0.5b", "smollm2:1.7b", "smollm2:360m", "llama3.2:1b"
    ]
    for m in ollama_models:
        recommendations.append({"name": m, "runtime": "ollama", "note": "Ollama manages memory"})
    
    return recommendations

def main():
    print("=" * 70)
    print("PHASE 0: HARDWARE/RUNTIME/MODEL COMPATIBILITY TEST")
    print("=" * 70)
    
    results = {}
    
    # 1. Python & packages
    print("\n[1/7] Checking Python environment...")
    results["python"] = check_python()
    print(f"  Python: {results['python']['version'].split()[0]}")
    for pkg in ["torch", "transformers", "accelerate", "bitsandbytes", "llama_cpp", "ollama"]:
        status = results["python"].get(pkg, "NOT_INSTALLED")
        print(f"  {pkg}: {status}")
    
    # 2. NVIDIA GPU
    print("\n[2/7] Checking NVIDIA GPU...")
    results["nvidia"] = check_nvidia()
    if "gpu_name" in results["nvidia"]:
        print(f"  GPU: {results['nvidia']['gpu_name']}")
        print(f"  VRAM: {results['nvidia']['vram_mb']} MB ({results['nvidia']['vram_mb']/1024:.1f} GB)")
        print(f"  Driver: {results['nvidia']['driver_version']}")
        print(f"  Compute Capability: {results['nvidia'].get('compute_capability', 'unknown')}")
    else:
        print(f"  ERROR: {results['nvidia'].get('error', 'nvidia-smi failed')}")
    
    if "cuda_version" in results["nvidia"]:
        print(f"  CUDA (nvcc): {results['nvidia']['cuda_version']}")
    
    # 3. PyTorch CUDA
    print("\n[3/7] Checking PyTorch CUDA...")
    results["pytorch_cuda"] = check_pytorch_cuda()
    if "error" not in results["pytorch_cuda"]:
        print(f"  Torch: {results['pytorch_cuda']['torch_version']}")
        print(f"  CUDA Available: {results['pytorch_cuda']['cuda_available']}")
        if results["pytorch_cuda"]["cuda_available"]:
            print(f"  Torch CUDA: {results['pytorch_cuda']['torch_cuda_version']}")
            print(f"  Device: {results['pytorch_cuda']['device_name']}")
            print(f"  Capability: {results['pytorch_cuda']['device_capability']}")
            print(f"  Total VRAM: {results['pytorch_cuda']['total_memory_mb']} MB")
    else:
        print(f"  ERROR: {results['pytorch_cuda']['error']}")
    
    # 4. Ollama
    print("\n[4/7] Checking Ollama...")
    results["ollama"] = check_ollama()
    if results["ollama"]["available"]:
        print(f"  Ollama: Available")
        print(f"  Models:\n{results['ollama']['models']}")
    else:
        print(f"  Ollama: Not available ({results['ollama'].get('error', 'unknown')})")
    
    # 5. llama.cpp
    print("\n[5/7] Checking llama.cpp...")
    results["llama_cpp"] = check_llama_cpp()
    if results["llama_cpp"]["available"]:
        print(f"  llama-cpp-python: {results['llama_cpp'].get('version', 'unknown')}")
    else:
        print(f"  llama-cpp-python: Not installed")
    if "binary_version" in results["llama_cpp"]:
        print(f"  llama.cpp binary: {results['llama_cpp']['binary_version']}")
    
    # 6. Disk space
    print("\n[6/7] Checking disk space...")
    results["disk"] = check_disk_space()
    if "free_gb" in results["disk"]:
        print(f"  Free: {results['disk']['free_gb']} GB / {results['disk']['total_gb']} GB")
    else:
        print(f"  ERROR: {results['disk'].get('error', 'unknown')}")
    
    # 7. Model recommendations
    print("\n[7/7] Model Recommendations for RTX 2050 6GB VRAM...")
    vram_gb = results["nvidia"].get("vram_mb", 6144) / 1024
    if "total_memory_mb" in results["pytorch_cuda"]:
        vram_gb = results["pytorch_cuda"]["total_memory_mb"] / 1024
    
    print(f"  Detected VRAM: {vram_gb:.1f} GB")
    recommendations = recommend_models(vram_gb)
    
    print(f"\n  RECOMMENDED MODELS (prioritized for 6GB VRAM):")
    print(f"  {'Model':<25} {'Size':<8} {'Quant':<8} {'Runtime':<15} {'Note'}")
    print(f"  {'-'*70}")
    for r in recommendations[:15]:
        note = r.get("note", "")
        print(f"  {r['name']:<25} {r.get('size_gb', '?'):<8} {r.get('quant', '?'):<8} {r['runtime']:<15} {note}")
    
    # Save results
    output_file = Path("phase0_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_file}")
    
    # Summary
    print("\n" + "=" * 70)
    print("PHASE 0 SUMMARY")
    print("=" * 70)
    
    gpu_name = results["nvidia"].get("gpu_name", "Unknown")
    vram = results["nvidia"].get("vram_mb", 0) / 1024
    cuda_ok = results["pytorch_cuda"].get("cuda_available", False)
    torch_ver = results["pytorch_cuda"].get("torch_version", "Not installed")
    ollama_ok = results["ollama"].get("available", False)
    llama_cpp_ok = results["llama_cpp"].get("available", False)
    
    print(f"GPU: {gpu_name} ({vram:.1f} GB VRAM)")
    print(f"PyTorch CUDA: {'WORKING' if cuda_ok else 'NOT WORKING'} (torch {torch_ver})")
    print(f"Ollama: {'AVAILABLE' if ollama_ok else 'NOT AVAILABLE'}")
    print(f"llama-cpp-python: {'AVAILABLE' if llama_cpp_ok else 'NOT AVAILABLE'}")
    print(f"Disk: {results['disk'].get('free_gb', '?')} GB free")
    
    print("\nRECOMMENDED APPROACH FOR 6GB RTX 2050:")
    print("  1. PRIMARY: Qwen2.5-1.5B (FP16, ~3GB) or Qwen2.5-0.5B (FP16, ~1GB)")
    print("  2. SECONDARY: Gemma-2-2B (INT4/GGUF Q4, ~2GB) via Ollama or llama.cpp")
    print("  3. FALLBACK: SmolLM2-360M (FP16, ~0.8GB) - guaranteed to fit")
    print("  4. OLLAMA MODELS: phi3:mini, gemma2:2b, qwen2.5:1.5b (auto-managed)")
    print("\n  QUANTIZATION OPTIONS ACTUALLY TESTABLE:")
    print("    - FP16 (full precision) vs INT4 (GGUF Q4_K_M) via Ollama/llama.cpp")
    print("    - Different models as 'configurations' (not same-model quantization)")
    print("    - bitsandbytes INT8/INT4 via transformers (if CUDA works)")
    
    print("\nNEXT STEPS:")
    print("  1. Install missing packages if needed")
    print("  2. Pull recommended Ollama models: ollama pull qwen2.5:1.5b")
    print("  3. Test actual inference with test_model_inference()")
    print("  4. Confirm Phase 0 results before proceeding to Phase A")
    
    return results

if __name__ == "__main__":
    main()