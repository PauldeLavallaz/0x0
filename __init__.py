
"""
ComfyUI custom node: comfyui-asset-to-url-0x0
Uploads IMAGE/AUDIO/FILE/BYTES to 0x0.st and returns a direct URL (STRING).
Designed to work with Comfy Deploy's External* nodes.
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
    pil = Image.fromarray(arr)
    bio = io.BytesIO()
    if fmt.lower() == "png":
        pil.save(bio, format="PNG")
        filename = "image.png"
    else:
        pil.save(bio, format="JPEG", quality=int(quality))
        filename = "image.jpg"
    return filename, bio.getvalue()


def _audio_obj_to_wav_bytes(audio, filename="audio.wav"):
    samples = audio["samples"]
    sr = int(audio["sample_rate"])
    if torch is not None and isinstance(samples, torch.Tensor):
        arr = samples.detach().cpu().numpy()
    else:
        arr = np.asarray(samples)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    channels = arr.shape[0]
    arr = np.clip(arr, -1.0, 1.0)
    pcm = (arr.T * 32767.0).astype(np.int16)
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes(order="C"))
    return (filename if filename.strip() else "audio.wav"), bio.getvalue()


class AnyToURL_0x0:
    """
    Accepts ANY input (IMAGE / AUDIO / BYTES / STRING path) and uploads to 0x0.st.
    - IMAGE (tensor): encoded as PNG/JPG
    - AUDIO ({samples, sample_rate}): encoded as WAV
    - BYTES/bytearray: uploaded as-is
    - STRING: if it's a valid local filepath, uploaded directly
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data": ("ANY", {"forceInput": True}),
                "filename_hint": ("STRING", {"default": "file.bin"}),
                "image_format": (["png", "jpg"], {"default": "png"}),
                "jpeg_quality": ("INT", {"default": 95, "min": 1, "max": 100}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, data, filename_hint="file.bin", image_format="png", jpeg_quality=95):
        # IMAGE tensor?
        if (torch is not None and isinstance(data, torch.Tensor)) or (hasattr(data, "shape") and hasattr(data, "dtype")):
            fname, payload = _image_tensor_to_bytes(data, fmt=image_format, quality=jpeg_quality)
            return (_upload_bytes(fname, payload),)

        # AUDIO dict?
        if _is_audio_obj(data):
            fname, payload = _audio_obj_to_wav_bytes(data, filename="audio.wav")
            return (_upload_bytes(fname, payload),)

        # raw bytes?
        if isinstance(data, (bytes, bytearray, memoryview)):
            fname = filename_hint if filename_hint.strip() else "file.bin"
            return (_upload_bytes(fname, bytes(data)),)

        # local path?
        if isinstance(data, str) and os.path.isfile(data):
            with open(data, "rb") as f:
                payload = f.read()
            base = os.path.basename(data)
            return (_upload_bytes(base, payload),)

        raise RuntimeError("AnyToURL_0x0: unsupported input type. Pass IMAGE, AUDIO, BYTES or local file path STRING.")


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
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "filename": ("STRING", {"default": "audio.wav"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, audio, filename="audio.wav"):
        if not _is_audio_obj(audio):
            raise RuntimeError("AudioToURL_0x0 expects an AUDIO object with 'samples' and 'sample_rate'")
        fname, data = _audio_obj_to_wav_bytes(audio, filename=filename)
        return (_upload_bytes(fname, data),)


class PathToURL_0x0:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"forceInput": True}),
            }
        }

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
    "AnyToURL_0x0": AnyToURL_0x0,
    "ImageToURL_0x0": ImageToURL_0x0,
    "AudioToURL_0x0": AudioToURL_0x0,
    "PathToURL_0x0": PathToURL_0x0,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnyToURL_0x0": "ANY → URL (0x0.st)",
    "ImageToURL_0x0": "Image → URL (0x0.st)",
    "AudioToURL_0x0": "Audio → URL (0x0.st)",
    "PathToURL_0x0": "Path → URL (0x0.st)",
}
