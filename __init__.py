
"""
ComfyUI custom node: comfyui-asset-to-url-0x0
Uploads IMAGE/AUDIO/FILE/BYTES to 0x0.st and returns a direct URL (STRING).
"""

import io
import os
import wave
import requests
import numpy as np

try:
    import torch
except Exception:
    torch = None

try:
    from PIL import Image
except Exception:
    Image = None

CATEGORY = "I/O → URL"

def _upload_bytes(filename: str, data: bytes) -> str:
    r = requests.post("https://0x0.st", files={"file": (filename, data)}, timeout=120)
    r.raise_for_status()
    return r.text.strip()

def _is_audio_obj(obj):
    return isinstance(obj, dict) and "samples" in obj and "sample_rate" in obj

def _to_numpy(x):
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _image_tensor_to_bytes(img, fmt="png", quality=95):
    if Image is None:
        raise RuntimeError("Pillow (PIL) is required for image encoding")
    # Comfy IMAGE is torch.Tensor [B,H,W,C] or [H,W,C] in [0,1]
    if torch is not None and isinstance(img, torch.Tensor):
        t = img
        if t.dim() == 4:
            t = t[0]
        arr = (t.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    else:
        arr = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    from PIL import Image as PILImage
    pil = PILImage.fromarray(arr)
    bio = io.BytesIO()
    if fmt.lower() == "png":
        pil.save(bio, format="PNG")
        filename = "image.png"
    else:
        pil.save(bio, format="JPEG", quality=int(quality))
        filename = "image.jpg"
    return filename, bio.getvalue()

def _audio_to_wav_bytes(samples, sample_rate, filename="audio.wav"):
    arr = _to_numpy(samples)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    channels = arr.shape[0]
    arr = np.clip(arr, -1.0, 1.0)
    pcm = (arr.T * 32767.0).astype(np.int16)
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes(order="C"))
    return (filename if str(filename).strip() else "audio.wav"), bio.getvalue()

class ImageToURL_0x0:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "format": (["png", "jpg"], {"default": "png"}),
                "quality": ("INT", {"default": 95, "min": 1, "max": 100})
            }
        }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    CATEGORY = CATEGORY
    def run(self, image, format="png", quality=95):
        fname, data = _image_tensor_to_bytes(image, fmt=format, quality=quality)
        return (_upload_bytes(fname, data),)

class AudioToURL_0x0:
    """
    Accepts various AUDIO shapes produced in Comfy / Comfy Deploy:
    - dict {'samples','sample_rate'}
    - tuple/list (samples, sample_rate)
    - STRING local path to audio file
    - bytes/bytearray
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "filename": ("STRING", {"default": ""}),  # used if we need to encode or for raw bytes
            }
        }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    CATEGORY = CATEGORY
    def run(self, audio, filename=""):
        # case 1: classic AUDIO dict
        if _is_audio_obj(audio):
            fname, data = _audio_to_wav_bytes(audio["samples"], audio["sample_rate"], filename or "audio.wav")
            return (_upload_bytes(fname, data),)
        # case 2: tuple/list (samples, sample_rate)
        if isinstance(audio, (tuple, list)) and len(audio) == 2:
            samples, sr = audio
            fname, data = _audio_to_wav_bytes(samples, sr, filename or "audio.wav")
            return (_upload_bytes(fname, data),)
        # case 3: a local file path (string)
        if isinstance(audio, str) and os.path.isfile(audio):
            with open(audio, "rb") as f:
                payload = f.read()
            base = os.path.basename(audio)  # keep original extension (mp3/m4a/wav)
            return (_upload_bytes(base, payload),)
        # case 4: raw bytes
        if isinstance(audio, (bytes, bytearray, memoryview)):
            fname = filename if str(filename).strip() else "audio.bin"
            return (_upload_bytes(fname, bytes(audio)),)
        raise RuntimeError("AudioToURL_0x0: unsupported AUDIO object. Pass dict{'samples','sample_rate'}, (samples,sr), local path STRING, or bytes.")

class PathToURL_0x0:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"path": ("STRING", {"forceInput": True})}}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    CATEGORY = CATEGORY
    def run(self, path):
        p = str(path)
        if not os.path.isfile(p):
            raise RuntimeError(f"Path does not exist or is not a file: {p}")
        with open(p, "rb") as f:
            data = f.read()
        return (_upload_bytes(os.path.basename(p), data),)

NODE_CLASS_MAPPINGS = {
    "ImageToURL_0x0": ImageToURL_0x0,
    "AudioToURL_0x0": AudioToURL_0x0,
    "PathToURL_0x0": PathToURL_0x0,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageToURL_0x0": "Image → URL (0x0.st)",
    "AudioToURL_0x0": "Audio → URL (0x0.st)",
    "PathToURL_0x0": "Path → URL (0x0.st)",
}
