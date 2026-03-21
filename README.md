# Qwen3-TTS RunPod Worker

[![Runpod](https://api.runpod.io/badge/earetaurus/qwen3-tts-runpod)](https://console.runpod.io/hub/earetaurus/qwen3-tts-runpod)

> **This project is ~80% vibe coded.** Built rapidly with heavy AI assistance. Use at your own discretion.

> If this project is useful to you, please consider using [my RunPod referral link](https://runpod.io?ref=akghcny7) to support the infrastructure costs.

## Acknowledgements

This project stands on the shoulders of giants:

- **[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)** by Qwen Team (Alibaba) — the underlying TTS model
- **[faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts)** by [@andimarafioti](https://github.com/andimarafioti) — CUDA graph acceleration for fast inference
- **Icons by [Flaticon](https://www.flaticon.com/uicons)**

## Features

- Works on **RunPod Serverless** and **Dedicated Pods**
- Voice cloning via reference audio (~5 GB VRAM needed)
- Voicemap support — pre-download voices from a JSON URL
- OpenAI-compatible `/v1/audio/speech` endpoint (SillyTavern compatible)
- Streaming audio output
- Fast cold starts with RunPod cached models (serverless)
- Lazy voice loading

## Quick Start

### 1. Build and Push

```bash
docker build -t your-docker-hub/runpod-qwen3-tts:latest ./public
docker push your-docker-hub/runpod-qwen3-tts:latest
```

### 2. Deploy on RunPod

1. Create a new **Serverless** endpoint on [runpod.io](https://runpod.io)
2. Set the **Docker Image** to your pushed image
3. Optionally set **Model** to `Qwen/Qwen3-TTS-12Hz-1.7B-Base` for cached model support
4. Add environment variables (see below)
5. Deploy

### 3. Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | `5000` | HTTP server port |
| `VOICE_MAP_URL` | — | URL to a JSON voicemap (see format below) |
| `USE_CACHED_MODEL` | `0` | Set to `1` to load model from `/runpod-volume/huggingface-cache/` |
| `USE_NETWORK_VOLUME` | `0` | Set to `1` to store HF cache on network volume |
| `POD_MODE` | `0` | Set to `1` for dedicated Pod — stores model/voices in `/workspace`, pre-downloads voicemap |
| `WARMUP_RUNS` | `3` | Number of warmup runs for CUDA graph capture |

#### Voicemap Format

```json
{
  "voicename": {
    "url": "https://example.com/path/to/voice.wav",
    "ref_text": "The text spoken in the reference audio."
  }
}
```

Example:
```json
{
  "voice_one": {
    "url": "https://example.com/pub/voice_one.wav",
    "ref_text": "Hello, this is my reference audio for voice cloning."
  }
}
```

## API Reference

### `POST /tts`
Custom endpoint for TTS generation.

**Form fields:**
- `text` (required) — Text to synthesize
- `voicemap` — Name of a voicemap entry (from `VOICE_MAP_URL`)
- `language` — Language (default: `English`)
- `ref_audio` — Upload reference WAV file
- `ref_url` — URL to reference WAV file
- `ref_text` — Text spoken in reference audio
- `mode` — `voice_clone`, `custom`, or `voice_design`
- `speaker` — Speaker name (for `custom` mode)
- `instruct` — Instruction text

**Response:**
```json
{
  "audio_b64": "<base64-encoded WAV>",
  "sample_rate": 24000,
  "metrics": {
    "total_ms": 1234,
    "audio_duration_s": 5.2,
    "rtf": 0.042
  }
}
```

**Curl example:**
```bash
curl -X POST "https://your-endpoint.api.runpod.ai/tts" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "text=Hello world!" \
  -F "voicemap=voice_one" \
  | jq -r '.audio_b64' | base64 -d > output.wav
```

### `POST /v1/audio/speech`
OpenAI-compatible speech endpoint.

**JSON body:**
```json
{
  "model": "qwen3-tts",
  "input": "Hello world!",
  "voice": "voice_one",
  "response_format": "mp3",
  "speed": 1.0
}
```

**Response:** Binary audio (`audio/mp3` or `audio/wav`).

### SillyTavern Setup

Install the **OpenAI Compatible TTS** extension, then configure:

- **Provider Endpoint**: `https://your-endpoint.api.runpod.ai/v1/audio/speech`
- **API Key**: Your RunPod API key
- **Model**: `qwen3-tts`
- **Available Voices**: Comma-separated voicemap names

### `GET /v1/models`
Returns available models.

### `GET /voices`
Lists available voicemaps and supported speakers/languages.

### `GET /ping`
Health check — returns `200` when ready, `204` during initialization.

### `GET /`
Root endpoint — returns status message.

## Docker Build (from scratch)

```bash
docker build -t your-docker-hub/runpod-qwen3-tts:latest ./public
docker push your-docker-hub/runpod-qwen3-tts:latest
```

## Dedicated Pod Deployment

The container also runs on RunPod **Dedicated Pods** (GPU persistent, no cold starts).

**Requirements:**
- **~5 GB VRAM** (tested on RTX 4090, should work on any 8GB+ GPU)
- Ubuntu 22.04 or similar Linux
- CUDA 12.6+ compatible drivers

**Setup:**
1. Create a Pod with a CUDA-capable template (e.g. `Ubuntu 22.04 CUDA 12.6`)
2. Expose port `5000` (or set `PORT` env var)
3. Pull and run the container:
   ```bash
   docker run -d --gpus all \
     -p 5000:5000 \
     -e PORT=5000 \
     -e VOICE_MAP_URL=https://your-voicemap.json \
     -e POD_MODE=1 \
     -e WARMUP_RUNS=3 \
     -v qwen3-workspace:/workspace \
     --name qwen3-tts \
     your-docker-hub/runpod-qwen3-tts:latest
   ```
4. Access at `http://<pod-ip>:5000`

**Notes:**
- Set `POD_MODE=1` to store model cache and voices in `/workspace` (use a volume mount for persistence)
- `WARMUP_RUNS` controls CUDA graph warmup iterations (serverless only — ignored on Pods after first run)
- The container downloads the ~4GB model on first start

## License

Apache License 2.0 — see [LICENSE](LICENSE) file.
