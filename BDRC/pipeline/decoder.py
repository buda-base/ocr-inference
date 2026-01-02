import asyncio
import concurrent.futures as futures
from .types import ImageTask, DecodedFrame
from .config import PipelineConfig
import io
import os
from typing import Tuple, Optional
import numpy as np
import cv2

class ImageDecodeError(RuntimeError):
    """Raised when an image cannot be decoded or processed safely."""

class Decoder:
    """Decode stage (thread pool).

    Input: (ImageTask, bytes)
    Output: DecodedFrame (binarized uint8, resized) pushed to q_frames; sends sentinel at end.
    """
    def __init__(self, cfg: PipelineConfig, q_bytes: asyncio.Queue, q_frames: asyncio.Queue):
        self.cfg = cfg
        self.q_bytes = q_bytes
        self.q_frames = q_frames
        self.pool = futures.ThreadPoolExecutor(max_workers=cfg.decode_threads, thread_name_prefix="decode")

    def _decode_one(self, task: ImageTask, body: bytes) -> DecodedFrame:
        # TODO: exception handling
        frame = bytes_to_binary_frame(task.s3_key, body, max_width=self.cfg.max_width, max_height=self.cfg.max_height)
        return DecodedFrame(task=task, frame=None, width=frame.shape[1], height=frame.shape[0])

    async def run(self):
        loop = asyncio.get_running_loop()
        while True:
            task, body = await self.q_bytes.get()
            if task is None:
                break
            fut = loop.run_in_executor(self.pool, self._decode_one, task, body)
            decoded = await fut
            await self.q_frames.put(decoded)
        await self.q_frames.put(None)


def _ext_lower(filename: str) -> str:
    return os.path.splitext(filename)[1].lower().lstrip(".")

def _compute_downscale(w: int, h: int, max_w: int, max_h: int) -> float:
    if w <= 0 or h <= 0:
        raise ImageDecodeError(f"Invalid image dimensions: {w}x{h}")
    s = min(max_w / float(w), max_h / float(h), 1.0)
    return s

def _downscale_gray(gray: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    h, w = gray.shape[:2]
    s = _compute_downscale(w, h, max_w, max_h)
    if s >= 1.0:
        return gray
    new_w = max(1, int(w * s))
    new_h = max(1, int(h * s))
    # INTER_AREA is best for downscaling continuous-tone grayscale
    return cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

def _downscale_binary(binary: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    h, w = binary.shape[:2]
    s = _compute_downscale(w, h, max_w, max_h)
    if s >= 1.0:
        return binary
    new_w = max(1, int(w * s))
    new_h = max(1, int(h * s))
    # Preserve 0/255 exactly for already-binary inputs
    return cv2.resize(binary, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

def _to_gray_cv(img: np.ndarray) -> np.ndarray:
    """
    Convert cv2-decoded image to single-channel uint8 grayscale with minimal overhead.
    """
    if img is None:
        raise ImageDecodeError("cv2.imdecode returned None")

    if img.dtype != np.uint8:
        # cv2.imdecode should yield uint8; if not, normalize safely.
        img = np.clip(img, 0, 255).astype(np.uint8, copy=False)

    if img.ndim == 2:
        return img  # already gray

    if img.ndim != 3:
        raise ImageDecodeError(f"Unexpected cv2 image ndim={img.ndim}")

    ch = img.shape[2]
    if ch == 3:
        # OpenCV uses BGR
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if ch == 4:
        # BGRA
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    raise ImageDecodeError(f"Unsupported channel count: {ch}")

def _adaptive_binarize(gray: np.ndarray, block_size: int, c: int) -> np.ndarray:
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ImageDecodeError("Adaptive binarization requires grayscale uint8")
    # block_size must be odd and >= 3
    if block_size < 3:
        block_size = 3
    if (block_size & 1) == 0:
        block_size += 1

    # adaptiveThreshold returns 0/255 uint8
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c,
    )

def _decode_via_cv2(image_bytes: bytes, likely_jpeg: bool) -> np.ndarray:
    """
    Decodes bytes as a uint8 grayscale frame with cv2 (mostly used for jpegs)

    Uses IMREAD_GRAYSCALE in imdecode when relevant for performance gains

    Returns the frame
    """
    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    ext = os.path.splitext(filename)[1].lower()

    if likely_jpeg:
        # Fast path: grayscale at decode time
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img

        # Fallback
        img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    else:
        img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ImageDecodeError("OpenCV failed to decode image")

    return img

def _decode_via_pil(filename: str, image_bytes: bytes) -> tuple[np.ndarray, bool]:
    """
    Decodes bytes as a uint8 grayscale frame with PIL (mostly used for tiffs)

    Returns the frame and a boolean True if the image is already binary (no need to re-binarize it)
    """
    try:
        from PIL import Image
    except Exception as e:
        raise ImageDecodeError(f"PIL not available: {e}") from e

    try:
        with Image.open(io.BytesIO(image_bytes)) as im:
            n_frames = getattr(im, "n_frames", 1)
            if n_frames != 1:
                raise ImageDecodeError(f"Multi-page image not supported ({n_frames} frames)")

            mode = im.mode

            # ---- binary-fast-path modes ----
            if mode == "1":
                arr = np.array(im, dtype=np.uint8) * 255
                return arr, True

            if mode in ("P", "L"):
                # Most Group4 TIFFs land here
                im = im.convert("L")
                arr = np.array(im, dtype=np.uint8)
                return arr, False

            # ---- definitely non-binary ----
            if mode in ("RGB", "RGBA", "CMYK"):
                im = im.convert("RGB")
                rgb = np.array(im, dtype=np.uint8)
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                return gray, False

            # ---- fallback ----
            im = im.convert("L")
            arr = np.array(im, dtype=np.uint8)
            return arr, False

    except ImageDecodeError:
        raise
    except Exception as e:
        raise ImageDecodeError(f"PIL decode failed for '{filename}': {e}") from e

def bytes_to_binary_frame(
    filename: str,
    image_bytes: bytes,
    *,
    max_width: int = 4096,
    max_height: int = 2048,
    block_size: int = 31,
    c: int = 15,
) -> np.ndarray:
    """
    Decode image bytes into a binary (0/255) uint8 OpenCV frame (2D array),
    downscaled to fit within (max_width, max_height).

    Optimized fast paths (99% of the data):
      - JPEG: OpenCV decode first; PIL fallback for CMYK/odd cases.
      - TIFF (Group4, already binary): PIL first (more reliable), OpenCV fallback.

    Raises ImageDecodeError on unrecoverable failures.
    """
    if not isinstance(filename, str) or not filename:
        raise ImageDecodeError("filename must be a non-empty string")
    if not isinstance(image_bytes, (bytes, bytearray, memoryview)):
        raise ImageDecodeError("image_bytes must be bytes-like")

    ext = _ext_lower(filename)

    # Decide preferred decoder lane based on extension and typical reliability
    likely_tiff = ext in ("tif", "tiff")
    likely_jpeg = ext in ("jpg", "jpeg")

    gray: Optional[np.ndarray] = None
    likely_binary = False

    # Decoder lane 1
    try:
        if likely_tiff:
            gray, likely_binary = _decode_via_pil(filename, image_bytes)
        else:
            img = _decode_via_cv2(image_bytes, likely_jpeg)
            gray = _to_gray_cv(img)
    except Exception as e1:
        # Decoder lane 2 (fallback)
        try:
            if gray is None:
                if likely_tiff:
                    # If PIL-first failed, try OpenCV
                    img = _decode_via_cv2(image_bytes, likely_jpeg)
                    gray = _to_gray_cv(img)
                else:
                    # If OpenCV-first failed, try PIL
                    gray, likely_binary = _decode_via_pil(filename, image_bytes)
        except Exception as e2:
            raise ImageDecodeError(
                f"Failed to decode '{filename}' via both OpenCV and PIL: "
                f"primary_error={type(e1).__name__}: {e1}; "
                f"fallback_error={type(e2).__name__}: {e2}"
            ) from e2

    if gray is None:
        raise ImageDecodeError("Decoded image is None after decoding attempts")

    # Ensure 2D uint8 (may be a bit too defensive)
    if gray.ndim != 2:
        # Extremely defensive: force to gray if something slipped through
        gray = _to_gray_cv(gray)

    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8, copy=False)

    # If already binary, preserve it and just downscale with nearest-neighbor
    if likely_binary:
        binary = _downscale_binary(gray, max_width, max_height)
        # Enforce exactly {0,255}
        if binary.max() == 1:
            binary = (binary * 255).astype(np.uint8, copy=False)
        return np.ascontiguousarray(binary)

    # Downscale before adaptive threshold (big speed win)
    gray = _downscale_gray(gray, max_width, max_height)

    # Adaptive threshold to 0/255
    binary = _adaptive_binarize(gray, block_size=block_size, c=c)

    # Enforce contiguous uint8 frame
    return np.ascontiguousarray(binary, dtype=np.uint8)
