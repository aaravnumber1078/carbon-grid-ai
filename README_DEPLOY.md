# CarbonGrid-AI Deployment Guide

## Overview

This guide covers deploying CarbonGrid-AI as a single Docker container on Hugging Face Spaces (GPU) or any Docker-compatible platform.

## Architecture

- **Single container** serving both FastAPI backend and frontend
- **FastAPI** on port 7860 (HF Spaces default)
- **Frontend** served at `/` via `FileResponse`
- **GPU required** for Qwen2.5-1.5B inference (FP16/INT4 via bitsandbytes)
- **Model weights** downloaded at runtime from Hugging Face Hub
- **Classifier artifacts** baked into image (`models/quantization_safety/`)
- **Carbon replay data** baked into image (`data/replay/`)

## Quick Start: Local Docker Test

### Prerequisites

- Docker with NVIDIA Container Toolkit (`nvidia-docker`)
- NVIDIA GPU with 8GB+ VRAM (T4, A10G, L4, H100, RTX 3080/4080/4090)
- Hugging Face token (optional, for gated models)

### Build and Run

```bash
# Clone and enter project
cd "AI model final"

# Build image (takes 5-10 min first time)
docker build -t carbongrid-ai .

# Run locally with GPU
docker run --rm --gpus all \
  -p 7860:7860 \
  -e HF_TOKEN=$HF_TOKEN \
  carbongrid-ai

# Or with custom config
docker run --rm --gpus all \
  -p 7860:7860 \
  -e HF_TOKEN=$HF_TOKEN \
  -e MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct \
  -e CARBON_MODE=live \
  -e DECISION_PROFILE=phase05 \
  carbongrid-ai
```

### Test Endpoints

```bash
# Health check
curl http://localhost:7860/health

# Classifier info
curl http://localhost:7860/classifier/info

# Carbon status
curl http://localhost:7860/carbon/status

# Generate (first request loads model ~15-30s)
curl -X POST http://localhost:7860/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain quantum computing simply.", "max_new_tokens": 128}'

# Frontend
open http://localhost:7860
```

## Hugging Face Spaces Deployment

### 1. Create New Space

1. Go to https://huggingface.co/new-space
2. **Owner**: your username/org
3. **Space name**: `carbongrid-ai` (or your choice)
4. **Visibility**: Public or Private
5. **SDK**: **Docker**
6. **Hardware**: **NVIDIA T4 Small** (free) or **NVIDIA A10G** (paid)
7. Click **Create Space**

### 2. Configure Secrets

In Space Settings → **Repository secrets**, add:

| Secret | Value | Required |
|--------|-------|----------|
| `HF_TOKEN` | Your HF token (Settings → Access Tokens) | If model is gated |

### 3. Push Code

```bash
# In your local project directory
git init
git add .
git commit -m "Initial deployment"

# Add HF Space as remote
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/carbongrid-ai

# Push
git push space main
```

### 4. Monitor Build

- Go to your Space page → **Logs** tab
- Build takes 5-15 minutes (installs PyTorch CUDA, transforms, bitsandbytes)
- Container starts automatically after build

### 5. Verify Deployment

- **Health**: `https://YOUR_USERNAME-carbongrid-ai.hf.space/health`
- **Frontend**: `https://YOUR_USERNAME-carbongrid-ai.hf.space/`
- **API Docs**: `https://YOUR_USERNAME-carbongrid-ai.hf.space/docs`

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `7860` | Port (HF Spaces expects 7860) |
| `MODEL_ID` | `Qwen/Qwen2.5-1.5B-Instruct` | HF model repo |
| `DEVICE_INDEX` | `0` | GPU device index |
| `MAX_CONCURRENT` | `1` | Max concurrent inferences |
| `CARBON_MODE` | `live` | `live`, `replay`, or `offline` |
| `CARBON_CONNECT_TIMEOUT` | `3.0` | Carbon API connect timeout (s) |
| `CARBON_READ_TIMEOUT` | `7.0` | Carbon API read timeout (s) |
| `CARBON_MAX_TOTAL_TIME` | `15.0` | Total carbon provider budget (s) |
| `FP16_VRAM_MB` | `3200` | Min free VRAM for FP16 |
| `INT4_VRAM_MB` | `1400` | Min free VRAM for INT4 |
| `DECISION_PROFILE` | `phase05` | `phase05` or `b2_workload` |
| `HF_TOKEN` | (none) | HF token for gated models |
| `KMP_DUPLICATE_LIB_OK` | `TRUE` | Workaround for OpenMP conflict |

## GPU Requirements

| GPU | VRAM | FP16 | INT4 (NF4) | Notes |
|-----|------|------|------------|-------|
| T4 (free tier) | 16 GB | ✅ | ✅ | Recommended minimum |
| A10G | 24 GB | ✅ | ✅ | Faster inference |
| L4 | 24 GB | ✅ | ✅ | Good price/performance |
| RTX 3080/4080 | 10-16 GB | ✅ | ✅ | Local testing |
| RTX 2050 (4 GB) | 4 GB | ⚠️ | ✅ | Tight for FP16 |

**Minimum**: 6 GB VRAM for INT4 only. 8 GB+ recommended for both precisions.

## Carbon Provider Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `live` | UK Carbon Intensity API (real-time) | Production with internet |
| `replay` | Pre-recorded data from `data/replay/` | Offline demo, testing |
| `offline` | Fixed 200 gCO₂/kWh fallback | No internet, no replay data |

**For HF Spaces**: Use `live` (has internet) or `replay` (guaranteed work).

## Decision Profiles

| Profile | FP16 Energy | INT4 Energy | FP16 Latency | INT4 Latency | Best For |
|---------|-------------|-------------|--------------|--------------|----------|
| `phase05` | 0.0488 Wh | 0.0493 Wh | 5.9s | 10.6s | Short answers, balanced |
| `b2_workload` | 0.1061 Wh | 0.0945 Wh | 11.6s | 22.1s | Long answers, INT4 saves energy |

**Default**: `phase05` (matches validation benchmarks)

## Testing Checklist

After deployment, verify:

- [ ] `/health` returns `{"status": "healthy", "classifier_loaded": true, ...}`
- [ ] `/classifier/info` shows C.1.1 threshold (0.50), test accuracy (~0.75)
- [ ] `/carbon/status` shows mode, intensity, source
- [ ] `/generate` returns response + `carbon_grid` telemetry
- [ ] Frontend loads at `/` with console working
- [ ] Simulator tab works (client-side decision mirror)
- [ ] Grid carbon tab shows live/replay data
- [ ] System status tab shows GPU info

## Troubleshooting

### Build Fails: CUDA Version Mismatch

Ensure base image matches PyTorch CUDA version:
- `torch==2.7.1+cu118` → `nvidia/cuda:11.8.0-runtime-ubuntu22.04`

### Runtime: Out of Memory

- Reduce `MAX_CONCURRENT` to 1 (default)
- Use `DECISION_PROFILE=phase05` (lower VRAM)
- Ensure no other processes using GPU

### Runtime: Model Download Fails

- Add `HF_TOKEN` secret for gated models
- Check network connectivity from Space
- Model downloads to `/root/.cache/huggingface` (ephemeral)

### Runtime: NVML Power Monitoring Returns Null

- Some cloud GPUs (T4) don't expose power via NVML
- UI will show "estimated" not "measured" — this is expected
- Energy/CO2 still calculated from profile baselines

### Runtime: First Request Times Out

- Model loads lazily on first request (~15-30s)
- Increase client timeout to 60s+
- Consider adding a `/warmup` endpoint for production

## File Structure in Container

```
/app/
├── carbongrid/           # Python package
├── frontend/
│   └── index.html        # Served at /
├── models/
│   └── quantization_safety/  # Classifier artifacts (baked)
│       ├── classifier_c1_1.pkl
│       ├── feature_extractor_c1_1.pkl
│       └── metadata_c1_1.json
└── data/
    └── replay/
        └── carbon_replay.json  # Replay carbon data (baked)
```

## Security Notes

- **Never commit** `.env`, `HF_TOKEN`, or secrets
- `.dockerignore` excludes caches, eval outputs, scripts
- Container runs as non-root user (uid 1000)
- CORS allows all origins (required for HF Spaces proxy)

## Updating

```bash
# Make changes locally
git add -A
git commit -m "Update: description"
git push space main
```

HF Spaces auto-rebuilds on push to main.

## Support

- **Issues**: GitHub Issues on source repo
- **HF Spaces Docs**: https://huggingface.co/docs/hub/spaces
- **Docker GPU**: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/