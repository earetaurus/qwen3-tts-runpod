#!/usr/bin/env python3
"""
Qwen 3 TTS Load Balancing Worker for RunPod Serverless.
Apache License 2.0 — see LICENSE file.

Voicemap URL provided via VOICE_MAP_URL environment variable.
"""

import base64
import io
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
import torch
import uvicorn
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse

from faster_qwen3_tts import FasterQwen3TTS

app = FastAPI(title="Qwen 3 TTS Worker")


_model = None
_model_loaded = False
_voice_map = {}
_voice_cache = {}

MAX_TEXT_CHARS = 5000


def _load_voice_map():
    global _voice_map
    voice_map_url = os.getenv("VOICE_MAP_URL", "")
    if not voice_map_url:
        print("Warning: VOICE_MAP_URL env not set")
        return
    try:
        print(f"Fetching voice map from {voice_map_url}...")
        resp = requests.get(voice_map_url, timeout=30)
        resp.raise_for_status()
        _voice_map = resp.json()
        print(f"Loaded {len(_voice_map)} voices from VOICE_MAP_URL")
    except Exception as e:
        print(f"Error loading VOICE_MAP_URL: {e}")


def _download_and_cache_voice(name: str, url: str, ref_text: str) -> bool:
    try:
        print(f"Downloading voice '{name}' from {url}...")
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        content = resp.content
        import hashlib

        digest = hashlib.sha1(content).hexdigest()
        path = VOICE_CACHE_DIR / f"qwen3_voice_{digest}.wav"
        if not path.exists():
            path.write_bytes(content)
        _voice_cache[name] = {"path": str(path), "ref_text": ref_text}
        print(f"  -> cached at {path}")
        return True
    except Exception as e:
        print(f"  -> failed: {e}")
        return False


def _ensure_voice_cached(name: str) -> dict:
    if name in _voice_cache:
        return _voice_cache[name]
    data = _voice_map.get(name)
    if not data:
        raise KeyError(name)
    voice_data = data if isinstance(data, dict) else {"url": data, "ref_text": ""}
    url = voice_data.get("url", "")
    ref_text = voice_data.get("ref_text", "")
    if not url:
        raise KeyError(name)
    _download_and_cache_voice(name, url, ref_text)
    return _voice_cache[name]


_pod_mode = os.getenv("POD_MODE", "0") == "1"
_workspace = Path("/workspace")
_use_network_volume = os.getenv("USE_NETWORK_VOLUME", "0") == "1"
HF_HOME = (
    _workspace / ".hf_home"
    if _pod_mode
    else (
        Path("/runpod-volume/.hf_home")
        if _use_network_volume
        else Path("/tmp/.hf_home")
    )
)
HF_HOME.mkdir(parents=True, exist_ok=True)
MODEL_LOCK = HF_HOME / ".download.lock"
VOICE_CACHE_DIR = _workspace / "voices"
VOICE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_use_cached_model = os.getenv("USE_CACHED_MODEL", "0") == "1"
HF_CACHE_ROOT = "/runpod-volume/huggingface-cache/hub"
MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

if _use_cached_model:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _resolve_snapshot_path(model_id: str) -> str:
    if not os.path.isdir(HF_CACHE_ROOT):
        raise RuntimeError(
            f"Cached model directory not found: {HF_CACHE_ROOT}\n"
            f"Did you set the 'Model' field on your RunPod endpoint to '{model_id}'?\n"
            f"See: https://docs.runpod.io/docs/cached-models"
        )
    if "/" not in model_id:
        raise ValueError(f"model_id '{model_id}' must be in 'org/name' format")
    org, name = model_id.split("/", 1)
    model_root = os.path.join(HF_CACHE_ROOT, f"models--{org.lower()}--{name.lower()}")
    refs_main = os.path.join(model_root, "refs", "main")
    snapshots_dir = os.path.join(model_root, "snapshots")
    if os.path.isfile(refs_main):
        with open(refs_main) as f:
            snapshot_hash = f.read().strip()
        candidate = os.path.join(snapshots_dir, snapshot_hash)
        if os.path.isdir(candidate):
            return candidate
    if os.path.isdir(snapshots_dir):
        versions = [
            d
            for d in os.listdir(snapshots_dir)
            if os.path.isdir(os.path.join(snapshots_dir, d))
        ]
        if versions:
            versions.sort()
            return os.path.join(snapshots_dir, versions[-1])
    raise RuntimeError(f"Cached model not found: {model_id}")


def _acquire_model_lock():
    lock_path = str(MODEL_LOCK)
    if MODEL_LOCK.exists():
        waited = 0
        while waited < 300:
            if not MODEL_LOCK.exists():
                break
            import time

            time.sleep(1)
            waited += 1
            print(f"Waiting for model download lock... ({waited}s)")
    Path(lock_path).touch()


def _release_model_lock():
    try:
        MODEL_LOCK.unlink()
    except:
        pass


@app.on_event("startup")
async def startup_event():
    global _model, _model_loaded
    _load_voice_map()

    if _pod_mode:
        print(f"POD_MODE enabled — pre-downloading voices to {VOICE_CACHE_DIR}...")
        for name, data in _voice_map.items():
            try:
                _download_and_cache_voice(
                    name, data.get("url", data), data.get("ref_text", "")
                )
            except Exception:
                pass
        print(f"  -> {len(_voice_cache)} voices cached")

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading FasterQwen3TTS model on {device}...")

        _acquire_model_lock()
        try:
            if _use_cached_model:
                model_path = _resolve_snapshot_path(MODEL_NAME)
                print(f"Loading from cached snapshot: {model_path}")
                _model = FasterQwen3TTS.from_pretrained(
                    model_path,
                    device=device,
                    dtype=torch.bfloat16,
                )
            else:
                _model = FasterQwen3TTS.from_pretrained(
                    MODEL_NAME,
                    device=device,
                    dtype=torch.bfloat16,
                )
        finally:
            _release_model_lock()

        print("Capturing CUDA graphs...")
        _model._warmup(prefill_len=8)
        _model_loaded = True
        print("Model ready!")
    except Exception as e:
        import traceback

        print(f"Error loading model: {e}")
        traceback.print_exc()
        _model_loaded = False


@app.get("/ping")
def health_check():
    if not _model_loaded:
        return JSONResponse(status_code=204, content={"status": "initializing"})
    return JSONResponse(status_code=200, content={"status": "healthy"})


@app.post("/tts")
async def generate_tts(
    text: str = Form(...),
    language: str = Form("English"),
    mode: str = Form("voice_clone"),
    voicemap: str = Form(""),
    ref_text: str = Form(""),
    speaker: str = Form(""),
    instruct: str = Form(""),
    ref_audio: UploadFile = File(None),
    ref_url: str = Form(""),
):
    global _model
    if not _model or not _model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Text too long ({len(text)} chars). Maximum is {MAX_TEXT_CHARS} characters.",
        )

    ref_path = None
    resolved_ref_text = ref_text

    if voicemap:
        try:
            voice = _ensure_voice_cached(voicemap)
            ref_path = voice["path"]
            resolved_ref_text = voice["ref_text"]
        except KeyError:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown voicemap '{voicemap}'. Available: {list(_voice_map.keys())}",
            )
    elif ref_url:
        response = requests.get(ref_url, timeout=30)
        response.raise_for_status()
        content = response.content
        import hashlib

        digest = hashlib.sha1(content).hexdigest()
        tmp_dir = Path(tempfile.gettempdir())
        path = tmp_dir / f"qwen3_tts_ref_{digest}.wav"
        if not path.exists():
            path.write_bytes(content)
        ref_path = str(path)
    elif ref_audio and ref_audio.filename:
        content = await ref_audio.read()
        import hashlib

        digest = hashlib.sha1(content).hexdigest()
        tmp_dir = Path(tempfile.gettempdir())
        path = tmp_dir / f"qwen3_tts_ref_{digest}.wav"
        if not path.exists():
            path.write_bytes(content)
        ref_path = str(path)

    t0 = time.perf_counter()

    if mode == "voice_clone":
        if not ref_path or not resolved_ref_text:
            raise HTTPException(
                status_code=400,
                detail="voice_clone mode requires ref_audio (or voicemap) and ref_text",
            )
        audio_list, sr = _model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=ref_path,
            ref_text=resolved_ref_text,
        )
    elif mode == "custom":
        if not speaker:
            raise HTTPException(
                status_code=400, detail="speaker is required for custom mode"
            )
        audio_list, sr = _model.generate_custom_voice(
            text=text,
            speaker=speaker,
            language=language,
            instruct=instruct,
        )
    else:
        audio_list, sr = _model.generate_voice_design(
            text=text,
            instruct=instruct,
            language=language,
        )

    audio = (
        np.concatenate([a for a in audio_list if len(a) > 0])
        if isinstance(audio_list, list)
        else audio_list
    )
    elapsed = time.perf_counter() - t0
    dur = len(audio) / sr
    rtf = dur / elapsed if elapsed > 0 else 0.0

    return JSONResponse(
        {
            "audio_b64": base64.b64encode(_to_wav_bytes(audio, sr)).decode(),
            "sample_rate": sr,
            "metrics": {
                "total_ms": round(elapsed * 1000),
                "audio_duration_s": round(dur, 3),
                "rtf": round(rtf, 3),
            },
        }
    )


def _to_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.squeeze()
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _chunk_to_wav_stream(audio: np.ndarray, sr: int) -> bytes:
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.squeeze()
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _stream_wav_chunks(audio_list, sr: int, stream: bool):
    if not stream:
        audio = (
            np.concatenate([a for a in audio_list if len(a) > 0])
            if isinstance(audio_list, list)
            else audio_list
        )
        yield _chunk_to_wav_stream(audio, sr)
        return

    chunks_out = 0
    for chunk in audio_list if isinstance(audio_list, list) else [audio_list]:
        if len(chunk) == 0:
            continue
        yield _chunk_to_wav_stream(chunk, sr)
        chunks_out += 1
    if chunks_out == 0:
        yield _chunk_to_wav_stream(np.zeros(sr, dtype=np.float32), sr)


def _wav_to_mp3(wav_bytes: bytes, speed: float = 1.0) -> bytes:
    ffmpeg_path = shutil.which("ffmpeg")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_in:
        wav_in.write(wav_bytes)
        wav_path = wav_in.name
    mp3_path = wav_path + ".mp3"
    try:
        cmd = [ffmpeg_path, "-y", "-i", wav_path]
        if speed != 1.0:
            cmd += ["-filter:a", f"atempo={speed}"]
        cmd += ["-q:a", "2", mp3_path]
        subprocess.run(cmd, check=True, capture_output=True)
        return Path(mp3_path).read_bytes()
    finally:
        try:
            os.unlink(wav_path)
        except:
            pass
        try:
            os.unlink(mp3_path)
        except:
            pass


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "qwen3-tts",
                "object": "model",
                "created": 1700000000,
                "owned_by": "qwen",
            }
        ],
    }


@app.post("/v1/audio/speech")
async def openai_speech(body: dict = Body(...)):
    if not _model or not _model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    input_text = body.get("input", "")
    voice = body.get("voice", "")
    response_format = body.get("response_format", "mp3")
    speed = float(body.get("speed", 1.0))
    stream = body.get("stream", True)

    if len(input_text) > MAX_TEXT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Text too long ({len(input_text)} chars). Maximum is {MAX_TEXT_CHARS}.",
        )

    try:
        v = _ensure_voice_cached(voice)
        ref_path = v["path"]
        ref_text = v["ref_text"]
    except KeyError:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice '{voice}'. Available: {list(_voice_map.keys())}",
        )

    audio_list, sr = _model.generate_voice_clone(
        text=input_text,
        language="English",
        ref_audio=ref_path,
        ref_text=ref_text,
    )

    if response_format == "mp3":
        audio = (
            np.concatenate([a for a in audio_list if len(a) > 0])
            if isinstance(audio_list, list)
            else audio_list
        )
        wav_bytes = _to_wav_bytes(audio, sr)
        content = _wav_to_mp3(wav_bytes, speed)
        return Response(content=content, media_type="audio/mp3")

    if stream:
        return StreamingResponse(
            _stream_wav_chunks(audio_list, sr, True),
            media_type="audio/wav",
        )

    audio = (
        np.concatenate([a for a in audio_list if len(a) > 0])
        if isinstance(audio_list, list)
        else audio_list
    )
    wav_bytes = _to_wav_bytes(audio, sr)
    return Response(content=wav_bytes, media_type="audio/wav")


@app.get("/voices")
async def get_voices():
    voicemaps = list(_voice_map.keys())
    try:
        speakers = _model.get_supported_speakers() if _model else []
        languages = _model.get_supported_languages() if _model else []
    except Exception:
        speakers, languages = [], []
    return {"voicemaps": voicemaps, "speakers": speakers, "languages": languages}


@app.get("/stats")
async def stats():
    return {
        "model_loaded": _model_loaded,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count()
        if torch.cuda.is_available()
        else 0,
    }


@app.get("/")
async def root():
    datacenter_id = os.getenv("RUNPOD_DC_ID")
    return {
        "message": f"Qwen 3 TTS Worker - {datacenter_id if datacenter_id else 'Ready'}".strip()
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
