
import asyncio
import traceback
from .types_common import DecodedFrame, Record, InferredFrame, PipelineError, EndOfStream
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from .img_helpers import apply_transform_1
from .debug_helpers import should_debug_image, save_debug_image, save_debug_contours

import cv2
import numpy as np
import numpy.typing as npt

# -----------------------------
# Packed-bits mask transport (GPU->CPU bandwidth optimization)
# -----------------------------
def _decode_line_mask(line_mask):
    """Accepts either:
    - a regular uint8 HxW numpy mask in {0,255}
    - or a tagged tuple ("packedbits", packed_u8[H,W8], H, W, pad)
      where packed bits are little-endian along width.

    Returns:
        uint8 HxW mask in {0,255}
    """
    if isinstance(line_mask, tuple) and len(line_mask) == 5 and line_mask[0] == "packedbits":
        _tag, packed, h, w, pad = line_mask
        if not isinstance(packed, np.ndarray):
            packed = np.asarray(packed, dtype=np.uint8)
        # Unpack little-endian to match packing in GPU stage.
        unpacked01 = np.unpackbits(packed, axis=1, bitorder="little")
        if pad:
            unpacked01 = unpacked01[:, :w]
        # Convert {0,1} -> {0,255} for existing invariants.
        return (unpacked01.astype(np.uint8) * 255)
    return line_mask

class LDPostProcessor:
    """
    Consumes InferredFrame from GPU pass 1 and pass 2, decides rotation/TPS, routes.

    Inputs:
      - q_gpu_pass_1_to_post_processor:  InferredFrameMsg (gpu_pass_1)
      - q_gpu_pass_2_to_post_processor: InferredFrameMsg (gpu_pass_2)
    Outputs:
      - q_post_processor_to_gpu_pass_2: DecodedFrameMsg (decoded stream)
      - q_post_processor_to_writer: RecordMsg (record stream)

    Records are produced ONLY here (per your note).
    """

    def __init__(
        self,
        cfg,
        q_gpu_pass_1_to_post_processor: asyncio.Queue,
        q_gpu_pass_2_to_post_processor: asyncio.Queue,
        q_post_processor_to_gpu_pass_2: asyncio.Queue,
        q_post_processor_to_writer: asyncio.Queue,
    ):
        self.cfg = cfg
        self.q_first = q_gpu_pass_1_to_post_processor
        self.q_second = q_gpu_pass_2_to_post_processor
        self.q_post_processor_to_gpu_pass_2 = q_post_processor_to_gpu_pass_2
        self.q_post_processor_to_writer = q_post_processor_to_writer

        self._p1_done = False
        self._p2_done = False

    async def _emit_pipeline_error(
        self,
        *,
        internal_stage: str,
        exc: BaseException,
        task: Any,
        source_etag: Optional[str],
        retryable: bool = False,
        attempt: int = 1,
    ) -> None:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        err = PipelineError(
            stage="LDPostProcessor",
            task=task,
            source_etag=source_etag,
            error_type=type(exc).__name__,
            message=f"[{internal_stage}] {exc}",
            traceback=tb,
            retryable=bool(retryable),
            attempt=int(attempt),
        )
        await self.q_post_processor_to_writer.put(err)

    async def _pop_one(self, q: asyncio.Queue, timeout_s: float):
        try:
            return await asyncio.wait_for(q.get(), timeout=timeout_s)
        except asyncio.TimeoutError:
            return None  # no item right now

    async def run(self):
        # Pass2 is cheap; prioritize it, but don't starve pass1 entirely.
        p2_budget = getattr(self.cfg, "reprocess_budget", 3)
        p1_budget = 1
        timeout_s = getattr(self.cfg, "controller_poll_ms", 5) / 1000.0

        while True:
            # Terminate only after both GPU streams have ended.
            if self._p1_done and self._p2_done:
                await self.q_post_processor_to_writer.put(EndOfStream(stream="record", producer="LDPostProcessor"))
                return

            took = False

            # --- Prefer gpu_pass_2 results (much faster to process) ---
            if not self._p2_done:
                for _ in range(p2_budget):
                    msg = await self._pop_one(self.q_second, timeout_s)
                    if msg is None:
                        break
                    took = True

                    if isinstance(msg, EndOfStream) and msg.stream == "gpu_pass_2":
                        self._p2_done = True
                        break

                    if isinstance(msg, PipelineError):
                        await self.q_post_processor_to_writer.put(msg)
                        continue

                    frame: InferredFrame = msg
                    try:
                        await self._finalize_record(frame)  # cheap path
                    except Exception as e:
                        await self._emit_pipeline_error(
                            internal_stage="run.finalize_pass2",
                            exc=e,
                            task=frame.task,
                            source_etag=frame.source_etag,
                            retryable=False,
                            attempt=1,
                        )

            if took:
                continue

            # --- Then gpu_pass_1 results ---
            if not self._p1_done:
                for _ in range(p1_budget):
                    msg = await self._pop_one(self.q_first, timeout_s)
                    if msg is None:
                        break
                    took = True

                    if isinstance(msg, EndOfStream) and msg.stream == "gpu_pass_1":
                        self._p1_done = True
                        # No more pass1 frames => no more reprocess frames will be generated.
                        await self.q_post_processor_to_gpu_pass_2.put(
                            EndOfStream(stream="transformed_pass_1", producer="LDPostProcessor")
                        )
                        break

                    if isinstance(msg, PipelineError):
                        await self.q_post_processor_to_writer.put(msg)
                        continue

                    frame: InferredFrame = msg
                    try:
                        await self._handle_pass1(frame)
                    except Exception as e:
                        await self._emit_pipeline_error(
                            internal_stage="run.handle_pass1",
                            exc=e,
                            task=frame.task,
                            source_etag=frame.source_etag,
                            retryable=False,
                            attempt=1,
                        )

            if not took:
                await asyncio.sleep(0)

    async def _handle_pass1(self, inf_frame: InferredFrame) -> None:
        # Decode packed-bit masks if GPU stage sent them
        line_mask = _decode_line_mask(inf_frame.line_mask)
        # Check debug mode once and reuse
        debug_this_image = should_debug_image(self.cfg, inf_frame.task.img_filename)
        
        # Debug helper (fire-and-forget, errors won't affect pipeline)
        async def _save_debug_safe(func, *args):
            try:
                await func(*args)
            except Exception:
                pass  # Silently ignore debug save errors
        
        if debug_this_image:
            asyncio.create_task(_save_debug_safe(save_debug_image, self.cfg, inf_frame.task.img_filename, "10_pass1_line_mask", line_mask))
        
        # 1. detect contours
        contours = get_filtered_contours(line_mask)

        # Debug: save pass1 contours
        if debug_this_image:
            asyncio.create_task(_save_debug_safe(save_debug_contours, self.cfg, inf_frame.task.img_filename, "11_pass1_contours", inf_frame.frame, contours))

        # 2. rotation angle
        h, w = line_mask.shape
        rotation_angle = get_rotation_angle(
            contours, h, w,
            max_angle_deg=self.cfg.max_angle_deg,
            min_angle_deg=self.cfg.min_angle_deg,
        )
        if rotation_angle != 0.0:
            contours = rotate_contours(contours, rotation_angle, h, w)
            
            # Debug: save rotation image
            if debug_this_image:
                # Import private helper for rotation (same package, so it's fine)
                from .img_helpers import _apply_rotation_1
                rotated_frame = _apply_rotation_1(inf_frame.frame, rotation_angle)
                asyncio.create_task(_save_debug_safe(save_debug_image, self.cfg, inf_frame.task.img_filename, "12_rotation", rotated_frame))

        # 3. TPS points (may be None)
        tps_points = get_tps_points(
            contours, h, w,
            legacy_tps_detect=self.cfg.legacy_tps_detect,
            alpha=self.cfg.tps_alpha,
            add_corners=self.cfg.add_corners,
        )

        # 4. either finalize or enqueue decoded frame to reprocess
        if tps_points is None and rotation_angle == 0.0:
            # scale contours to original image dimensions (inf_frame.orig_h, inf_frame.orig_w)
            contours = scale_contours(contours, line_mask.shape[0], line_mask.shape[1], inf_frame.orig_h, inf_frame.orig_w)
            contours_bboxes = get_contour_bboxes(contours)
            rec = Record(
                task=inf_frame.task,
                source_etag=inf_frame.source_etag,
                rotation_angle=None,
                tps_data=None,
                contours=contours,
                nb_contours=len(contours),
                contours_bboxes=contours_bboxes,
            )
            await self.q_post_processor_to_writer.put(rec)
            return

        input_pts = output_pts = None
        alpha = None
        tps_data = None
        if tps_points is not None:
            input_pts, output_pts = tps_points
            alpha = self.cfg.tps_alpha
            # Only construct tps_data when we have valid TPS points
            if input_pts is not None and output_pts is not None:
                tps_data = (input_pts, output_pts, alpha)

        transformed_frame = apply_transform_1(inf_frame.frame, rotation_angle, input_pts, output_pts, alpha)
        
        # Debug: save TPS image (if TPS was applied)
        if debug_this_image and tps_points is not None:
            # We want to show TPS on the rotated image (if rotation was applied)
            # Import private helpers (same package, so it's fine)
            from .img_helpers import _apply_rotation_1, _apply_tps_1
            if rotation_angle != 0.0:
                rotated_frame = _apply_rotation_1(inf_frame.frame, rotation_angle)
                tps_frame = _apply_tps_1(rotated_frame, input_pts, output_pts, alpha)
            else:
                tps_frame = _apply_tps_1(inf_frame.frame, input_pts, output_pts, alpha)
            asyncio.create_task(_save_debug_safe(save_debug_image, self.cfg, inf_frame.task.img_filename, "13_tps", tps_frame))
        
        await self.q_post_processor_to_gpu_pass_2.put(
            DecodedFrame(
                task=inf_frame.task,
                source_etag=inf_frame.source_etag,
                frame=transformed_frame,
                orig_w=inf_frame.orig_w,
                orig_h=inf_frame.orig_h,
                is_binary=False, # we don't map to binary after processing binary images
                first_pass=False,
                rotation_angle=rotation_angle,
                tps_data=tps_data,
            )
        )

    async def _finalize_record(self, inf_frame: InferredFrame) -> None:
        # Decode packed-bit masks if GPU stage sent them
        line_mask = _decode_line_mask(inf_frame.line_mask)
        # Debug: save pass2 line mask (fire-and-forget, errors won't affect pipeline)
        async def _save_debug_safe(func, *args):
            try:
                await func(*args)
            except Exception:
                pass  # Silently ignore debug save errors
        
        if should_debug_image(self.cfg, inf_frame.task.img_filename):
            asyncio.create_task(_save_debug_safe(save_debug_image, self.cfg, inf_frame.task.img_filename, "20_pass2_line_mask", line_mask))
        
        # Cheap pass2 finalization: just contours
        contours = get_filtered_contours(line_mask)
        
        # Debug: save pass2 contours (on the transformed frame if available)
        if should_debug_image(self.cfg, inf_frame.task.img_filename):
            # Use the frame from inf_frame (which is the transformed frame after rotation/TPS)
            asyncio.create_task(_save_debug_safe(save_debug_contours, self.cfg, inf_frame.task.img_filename, "21_pass2_contours", inf_frame.frame, contours))
        
        # scale contours to original image dimensions (frame.orig_h, frame.orig_w)
        contours = scale_contours(contours, line_mask.shape[0], line_mask.shape[1], inf_frame.orig_h, inf_frame.orig_w)
        # scale tps_data to original image dimensions
        tps_data = inf_frame.tps_data
        if tps_data is not None:
            # tps_data should only be a tuple if we have valid TPS points
            input_pts, output_pts, alpha = tps_data
            scaled_tps_points = scale_tps_points(input_pts, output_pts, line_mask.shape[0], line_mask.shape[1], inf_frame.orig_h, inf_frame.orig_w)
            tps_data = (scaled_tps_points[0], scaled_tps_points[1], alpha)
        contours_bboxes = get_contour_bboxes(contours)
        h, w = line_mask.shape[:2]
        rec = Record(
            task=inf_frame.task,
            source_etag=inf_frame.source_etag,
            rotation_angle=inf_frame.rotation_angle,
            tps_data=tps_data,
            contours=contours,
            nb_contours=len(contours),
            contours_bboxes=contours_bboxes,
        )
        await self.q_post_processor_to_writer.put(rec)


# -----------------------------
# Defaults (tweakable)
# -----------------------------
MAX_ANGLE_DEG_DEFAULT = 5.0
MIN_ANGLE_DEG_DEFAULT = 0.2

# Contour filtering defaults (for "line-ish" contours in a binary mask)
MIN_AREA_FRAC_DEFAULT = 0.001   # fraction of image area
MIN_W_FRAC_DEFAULT = 0.01       # fraction of image width
MIN_H_PX_DEFAULT = 10           # minimum bbox height in px

# TPS defaults
TPS_SLICE_WIDTH_DEFAULT = 40
TPS_MIN_POINTS_PER_WINDOW_DEFAULT = 25
TPS_Y_LO_PCT_DEFAULT = 10.0
TPS_Y_HI_PCT_DEFAULT = 90.0
TPS_MAX_MISSING_WINDOWS_DEFAULT = 2
TPS_ALPHA_DEFAULT = 0.5
TPS_ADD_CORNERS_DEFAULT = True

def scale_contours(
    contours: Iterable[np.ndarray],
    src_h: int,
    src_w: int,
    dst_h: int,
    dst_w: int,
) -> List[np.ndarray]:
    """
    Scales contours produced by cv2.findContours from a (src_h, src_w) frame
    into a (dst_h, dst_w) frame.

    Expected contour formats:
      - (N, 1, 2)  [OpenCV standard]
      - (N, 2)

    Coordinates are assumed to be (x, y).

    Returns:
        A list of scaled contours with the same shape as the input contours,
        dtype float32.
    """
    if src_h == dst_h and src_w == dst_w:
        return contours

    if src_h <= 0 or src_w <= 0 or dst_h <= 0 or dst_w <= 0:
        raise ValueError(
            f"Invalid dimensions: src=({src_h},{src_w}) dst=({dst_h},{dst_w})"
        )

    sx = dst_w / float(src_w)
    sy = dst_h / float(src_h)

    scaled_contours: List[np.ndarray] = []

    for contour in contours:
        if not isinstance(contour, np.ndarray):
            raise TypeError("Each contour must be a numpy array")

        # Work on a float copy to avoid integer truncation during scaling
        pts = contour.astype(np.float64, copy=True)

        if pts.ndim == 3 and pts.shape[1:] == (1, 2):
            # (N, 1, 2)
            pts[:, 0, 0] *= sx  # x
            pts[:, 0, 1] *= sy  # y

        elif pts.ndim == 2 and pts.shape[1] == 2:
            # (N, 2)
            pts[:, 0] *= sx  # x
            pts[:, 1] *= sy  # y

        else:
            raise ValueError(
                f"Unsupported contour shape {contour.shape}; "
                "expected (N,1,2) or (N,2)"
            )

        # Round and cast back to int32
        pts = np.round(pts).astype(np.int32)

        scaled_contours.append(pts)

    return scaled_contours

def scale_tps_points(
    tps_input_points: np.ndarray,
    tps_output_points: np.ndarray,
    src_h: int,
    src_w: int,
    dst_h: int,
    dst_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Same scaling as scale_contours, but for TPS point arrays.

    Assumptions:
      - tps_input_points and tps_output_points are float64 numpy arrays.
      - Shape is (N, 2) where each row is (x, y).
      - Returns new arrays (does not modify inputs).

    Returns:
        (scaled_input_points, scaled_output_points) as float64 arrays.
    """
    if src_h == dst_h and src_w == dst_w:
        return tps_input_points, tps_output_points

    if src_h <= 0 or src_w <= 0 or dst_h <= 0 or dst_w <= 0:
        raise ValueError(
            f"Invalid dimensions: src=({src_h},{src_w}) dst=({dst_h},{dst_w})"
        )

    if not isinstance(tps_input_points, np.ndarray) or not isinstance(
        tps_output_points, np.ndarray
    ):
        raise TypeError("tps_input_points and tps_output_points must be numpy arrays")

    sx = dst_w / float(src_w)
    sy = dst_h / float(src_h)

    scaled_in = tps_input_points.copy()
    scaled_out = tps_output_points.copy()

    scaled_in[:, 0] *= sx
    scaled_in[:, 1] *= sy

    scaled_out[:, 0] *= sx
    scaled_out[:, 1] *= sy

    return scaled_in, scaled_out



# -----------------------------
# Validation helpers
# -----------------------------
def _assert_mask_uint8_binary_0_255(line_mask: npt.NDArray[np.uint8]) -> None:
    if line_mask.dtype != np.uint8:
        raise TypeError(f"line_mask must be uint8, got {line_mask.dtype}")
    if line_mask.ndim != 2:
        raise ValueError(f"line_mask must be 2D (H,W), got shape={line_mask.shape}")

    # strict binary check (your invariant for line masks)
    u = np.unique(line_mask)
    if not (u.size <= 2 and set(map(int, u)).issubset({0, 255})):
        raise ValueError(f"line_mask must be binarized in {{0,255}}, got unique={u[:10]}")

def get_contour_bboxes(contours):
    """
    Returns axis-aligned (x, y, w, h) bboxes,
    computed from minAreaRect for robustness.
    """
    bboxes = []

    for cnt in contours:
        if cnt is None or len(cnt) < 3:
            continue

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)      # 4x2
        box = np.int32(np.round(box))

        x, y, w, h = cv2.boundingRect(box)
        bboxes.append((x, y, w, h))

    return bboxes

# -----------------------------
# Public API 1: contours
# -----------------------------
def get_filtered_contours(
    line_mask: npt.NDArray[np.uint8],
    *,
    min_area_frac: float = MIN_AREA_FRAC_DEFAULT,
    min_w_frac: float = MIN_W_FRAC_DEFAULT,
    min_h_px: int = MIN_H_PX_DEFAULT,
    retrieval_mode: int = cv2.RETR_LIST,
    approx_mode: int = cv2.CHAIN_APPROX_SIMPLE,
) -> List[npt.NDArray[np.int32]]:
    """
    Find contours in a uint8 binary mask in {0,255}, and filter to keep line-like shapes.
    """
    _assert_mask_uint8_binary_0_255(line_mask)

    contours, _ = cv2.findContours(line_mask, retrieval_mode, approx_mode)
    if not contours:
        return []

    h, w = line_mask.shape
    img_area = float(h * w)
    min_area = img_area * float(min_area_frac)
    min_w = float(w) * float(min_w_frac)

    out: List[npt.NDArray[np.int32]] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area <= min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if bw < min_w or bh < int(min_h_px):
            continue
        out.append(c)

    return out

# -----------------------------
# Angle estimation helpers
# -----------------------------
def _rect_angle_to_skew_deg(angle_deg: float) -> float:
    """
    Normalize cv2.minAreaRect angle to a skew angle near 0 deg.
    Output roughly in (-45, 45].
    """
    a = float(angle_deg)

    # Handle [0, 90) convention
    if 0.0 <= a <= 90.0:
        if a > 45.0:
            a -= 90.0
        return a

    # Handle (-90, 0] convention
    if a < -45.0:
        a += 90.0
    return a


def _robust_centered_angle_deg(
    angles_deg: npt.NDArray[np.float32],
    weights: Optional[npt.NDArray[np.float32]] = None,
) -> float:
    """
    Robust aggregation: MAD-trim then weighted mean.
    """
    if angles_deg.size == 0:
        return 0.0

    med = float(np.median(angles_deg))
    mad = float(np.median(np.abs(angles_deg - med)))
    if mad == 0.0:
        if weights is None:
            return float(np.mean(angles_deg))
        wsum = float(np.sum(weights))
        return float(np.sum(angles_deg * weights) / wsum) if wsum > 0 else float(np.mean(angles_deg))

    keep = np.abs(angles_deg - med) <= (3.0 * mad)
    kept = angles_deg[keep]
    if kept.size == 0:
        return med

    if weights is None:
        return float(np.mean(kept))

    kept_w = weights[keep]
    wsum = float(np.sum(kept_w))
    return float(np.sum(kept * kept_w) / wsum) if wsum > 0 else float(np.mean(kept))


def _rotation_matrix(angle_deg: float, h: int, w: int) -> npt.NDArray[np.float32]:
    center = (w / 2.0, h / 2.0)
    return cv2.getRotationMatrix2D(center, float(angle_deg), 1.0).astype(np.float32)


# -----------------------------
# Public API 2: angle
# -----------------------------
def get_rotation_angle(
    contours: Sequence[npt.NDArray[np.int32]],
    h: int,
    w: int,
    *,
    max_angle_deg: float = MAX_ANGLE_DEG_DEFAULT,
    min_angle_deg: float = MIN_ANGLE_DEG_DEFAULT,
    use_area_weights: bool = True,
) -> float:
    """
    Returns the deskew rotation angle in degrees (positive = CCW).
    Rules:
      - If |angle| > max_angle_deg => raises ValueError
      - If |angle| < min_angle_deg => returns 0.0
    """
    if not contours:
        return 0.0

    angles: List[float] = []
    weights: List[float] = []

    for c in contours:
        rect = cv2.minAreaRect(c)
        (rw, rh) = rect[1]
        raw_angle = rect[2]
        skew = _rect_angle_to_skew_deg(raw_angle)

        if abs(skew) < 1e-6:
            continue
        if abs(skew) > float(max_angle_deg):
            continue

        angles.append(skew)
        if use_area_weights:
            weights.append(max(float(rw * rh), 1.0))

    if not angles:
        return 0.0

    a = np.asarray(angles, dtype=np.float32)
    wts = np.asarray(weights, dtype=np.float32) if (use_area_weights and len(weights) == len(angles)) else None
    angle = float(_robust_centered_angle_deg(a, wts))

    if abs(angle) > float(max_angle_deg):
        raise ValueError(f"Detected angle {angle:.3f}° exceeds max {max_angle_deg:.3f}°")

    if abs(angle) < float(min_angle_deg):
        return 0.0

    return angle


# -----------------------------
# Public API 3: rotate contours
# -----------------------------
def rotate_contours(
    contours: Sequence[npt.NDArray[np.int32]],
    angle_deg: float,
    h: int,
    w: int,
) -> List[npt.NDArray[np.int32]]:
    """
    Rotate contours using the same affine transform you should use for the image/mask.
    Output contours are int32 with rounding-to-nearest.
    """
    if not contours or abs(float(angle_deg)) < 1e-12:
        return [c.copy() for c in contours]

    M = _rotation_matrix(angle_deg, h, w)

    out: List[npt.NDArray[np.int32]] = []
    for c in contours:
        rc = cv2.transform(c, M)
        out.append(np.rint(rc).astype(np.int32))
    return out


# -----------------------------
# TPS: robust contour-only detection (default)
# -----------------------------
def _robust_window_centerline_from_points(
    xs: npt.NDArray[np.int32],
    ys: npt.NDArray[np.int32],
    *,
    x0: int,
    x1: int,
    fallback_x: int,
    fallback_y: int,
    min_points: int,
    y_lo_pct: float,
    y_hi_pct: float,
) -> Tuple[int, int, int, int]:
    m = (xs >= x0) & (xs < x1)
    n = int(np.count_nonzero(m))
    if n < min_points:
        return int(fallback_x), int(fallback_y), 0, n

    wx = xs[m].astype(np.float32)
    wy = ys[m].astype(np.float32)

    cx = int(np.median(wx))
    y_lo = float(np.percentile(wy, y_lo_pct))
    y_hi = float(np.percentile(wy, y_hi_pct))

    thickness = max(0, int(round(y_hi - y_lo)))
    cy = int(round((y_lo + y_hi) * 0.5))
    return cx, cy, thickness, n


def _check_line_tps_geom_robust(
    contour: npt.NDArray[np.int32],
    h: int,
    w: int,
    *,
    slice_width: int,
    min_points_per_window: int,
    y_lo_pct: float,
    y_hi_pct: float,
    max_missing_windows: int,
) -> Tuple[bool, Optional[List[List[int]]], Optional[List[List[int]]], float]:
    x, y, bw, bh = cv2.boundingRect(contour)

    pts = contour.reshape(-1, 2)  # (N,2) (x,y)
    xs = pts[:, 0]
    ys = pts[:, 1]

    win = [
        (x, x + slice_width),
        (x + bw // 4 - slice_width, x + bw // 4),
        (x + bw // 2, x + bw // 2 + slice_width),
        (x + bw // 2 + bw // 4, x + bw // 2 + (bw // 4) + slice_width),
        (x + bw - slice_width, x + bw),
    ]

    def clamp(a: int, lo: int, hi: int) -> int:
        return max(lo, min(hi, a))

    win = [(clamp(a, 0, w), clamp(b, 0, w)) for (a, b) in win]

    fallback_x = x + bw // 2
    fallback_y = y + bh // 2
    fallback_th = max(1, bh)

    centers_xy: List[Tuple[int, int]] = []
    thicknesses: List[int] = []
    missing = 0

    for (x0, x1) in win:
        cx, cy, th, _n = _robust_window_centerline_from_points(
            xs, ys,
            x0=x0, x1=x1,
            fallback_x=fallback_x,
            fallback_y=fallback_y,
            min_points=min_points_per_window,
            y_lo_pct=y_lo_pct,
            y_hi_pct=y_hi_pct,
        )

        if th <= 0:
            missing += 1
            th = fallback_th

        centers_xy.append((cx, cy))
        thicknesses.append(th)

    if missing > max_missing_windows:
        return False, None, None, 0.0

    all_centers_y = [cy for (_, cy) in centers_xy]
    max_ydelta = float(max(all_centers_y) - min(all_centers_y))
    mean_th = float(np.mean(thicknesses))
    mean_center_y = float(np.mean(all_centers_y))

    if max_ydelta > mean_th:
        target_y = int(round(mean_center_y))
        input_pts = [[cy, cx] for (cx, cy) in centers_xy]          # [y,x]
        output_pts = [[target_y, cx] for (cx, cy) in centers_xy]   # [y,x]
        return True, input_pts, output_pts, max_ydelta

    return False, None, None, 0.0


# -----------------------------
# TPS: legacy mask-writing detection (fallback)
# -----------------------------
def _get_global_center_from_slice(slice_image_2d: npt.NDArray[np.uint8], start_x: int, bbox_y: int) -> Tuple[int, int, int]:
    contours, _ = cv2.findContours(slice_image_2d, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        cx = slice_image_2d.shape[1] // 2
        cy = slice_image_2d.shape[0] // 2
        bh = slice_image_2d.shape[0]
        return start_x + cx, bbox_y + cy, bh

    areas = [cv2.contourArea(c) for c in contours]
    biggest = contours[int(np.argmax(areas))]
    _, _, _, bh = cv2.boundingRect(biggest)
    (cx_f, cy_f), _, _ = cv2.minAreaRect(biggest)
    return start_x + int(cx_f), bbox_y + int(cy_f), int(bh)


def _check_line_tps_legacy(
    contour: npt.NDArray[np.int32],
    h: int,
    w: int,
    *,
    slice_width: int,
) -> Tuple[bool, Optional[List[List[int]]], Optional[List[List[int]]], float]:
    mask = np.zeros((h, w), dtype=np.uint8)
    x, y, bw, bh = cv2.boundingRect(contour)
    cv2.drawContours(mask, [contour], contourIdx=0, color=255, thickness=-1)

    def clamp(a: int, lo: int, hi: int) -> int:
        return max(lo, min(hi, a))

    # 5 slices
    s1x0, s1x1 = x, x + slice_width
    s2x0, s2x1 = x + bw // 4 - slice_width, x + bw // 4
    s3x0, s3x1 = x + bw // 2, x + bw // 2 + slice_width
    s4x0, s4x1 = x + bw // 2 + bw // 4, x + bw // 2 + (bw // 4) + slice_width
    s5x0, s5x1 = x + bw - slice_width, x + bw

    s1x0, s1x1 = clamp(s1x0, 0, w), clamp(s1x1, 0, w)
    s2x0, s2x1 = clamp(s2x0, 0, w), clamp(s2x1, 0, w)
    s3x0, s3x1 = clamp(s3x0, 0, w), clamp(s3x1, 0, w)
    s4x0, s4x1 = clamp(s4x0, 0, w), clamp(s4x1, 0, w)
    s5x0, s5x1 = clamp(s5x0, 0, w), clamp(s5x1, 0, w)

    sl1 = mask[y:y + bh, s1x0:s1x1]
    sl2 = mask[y:y + bh, s2x0:s2x1]
    sl3 = mask[y:y + bh, s3x0:s3x1]
    sl4 = mask[y:y + bh, s4x0:s4x1]
    sl5 = mask[y:y + bh, s5x0:s5x1]

    p1x, p1y, b1h = _get_global_center_from_slice(sl1, s1x0, y)
    p2x, p2y, b2h = _get_global_center_from_slice(sl2, s2x0, y)
    p3x, p3y, b3h = _get_global_center_from_slice(sl3, s3x0, y)
    p4x, p4y, b4h = _get_global_center_from_slice(sl4, s4x0, y)
    p5x, p5y, b5h = _get_global_center_from_slice(sl5, s5x0, y)

    all_bh = [b1h, b2h, b3h, b4h, b5h]
    all_cy = [p1y, p2y, p3y, p4y, p5y]

    max_ydelta = float(max(all_cy) - min(all_cy))
    mean_bh = float(np.mean(all_bh))
    mean_cy = float(np.mean(all_cy))

    if max_ydelta > mean_bh:
        target_y = int(round(mean_cy))
        input_pts = [[p1y, p1x], [p2y, p2x], [p3y, p3x], [p4y, p4x], [p5y, p5x]]      # [y,x]
        output_pts = [[target_y, p1x], [target_y, p2x], [target_y, p3x], [target_y, p4x], [target_y, p5x]]
        return True, input_pts, output_pts, max_ydelta

    return False, None, None, 0.0


def _get_global_tps_line_idx(line_data: List[Dict[str, Any]]) -> int:
    all_y_deltas = [float(ld["max_yd"]) if ld["tps"] else 0.0 for ld in line_data]
    mean_delta = float(np.mean(all_y_deltas))

    best_diff = float(max(all_y_deltas))
    best_y = None
    for yd in all_y_deltas:
        if yd > 0:
            diff = abs(mean_delta - yd)
            if diff < best_diff:
                best_diff = diff
                best_y = yd

    if best_y is None:
        return -1
    return int(all_y_deltas.index(best_y))


# -----------------------------
# Public API 4: TPS points (robust default, legacy optional)
# -----------------------------
def get_tps_points(
    contours: Sequence[npt.NDArray[np.int32]],
    h: int,
    w: int,
    *,
    legacy_tps_detect: bool = False, # same (costly) behavior as v1 app
    slice_width: int = TPS_SLICE_WIDTH_DEFAULT,
    min_points_per_window: int = TPS_MIN_POINTS_PER_WINDOW_DEFAULT,
    y_lo_pct: float = TPS_Y_LO_PCT_DEFAULT,
    y_hi_pct: float = TPS_Y_HI_PCT_DEFAULT,
    max_missing_windows: int = TPS_MAX_MISSING_WINDOWS_DEFAULT,
    alpha: float = TPS_ALPHA_DEFAULT,
    add_corners: bool = True,
) -> Optional[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
    """
    Returns TPS control points (input_pts, output_pts) in float64 or None if no TPS correction needed.

    IMPORTANT:
      - Points are in [y, x] order
      - If add_corners=True (default), image corners are INCLUDED
        in input_pts / output_pts
    """
    if not contours:
        return None

    line_data: List[Dict[str, Any]] = []

    for cnt in contours:
        if legacy_tps_detect:
            tps_status, input_pts, output_pts, max_yd = _check_line_tps_legacy(
                cnt, h, w, slice_width=slice_width
            )
        else:
            tps_status, input_pts, output_pts, max_yd = _check_line_tps_geom_robust(
                cnt, h, w,
                slice_width=slice_width,
                min_points_per_window=min_points_per_window,
                y_lo_pct=y_lo_pct,
                y_hi_pct=y_hi_pct,
                max_missing_windows=max_missing_windows,
            )

        line_data.append(
            {
                "tps": tps_status,
                "input_pts": input_pts,
                "output_pts": output_pts,
                "max_yd": max_yd,
            }
        )

    if not any(ld["tps"] for ld in line_data):
        return None

    best_idx = _get_global_tps_line_idx(line_data)
    if best_idx < 0:
        return None

    input_pts = line_data[best_idx]["input_pts"]
    output_pts = line_data[best_idx]["output_pts"]
    if input_pts is None or output_pts is None:
        return None

    input_pts_np = np.asarray(input_pts, dtype=np.float64)
    output_pts_np = np.asarray(output_pts, dtype=np.float64)

    if add_corners:
        corners = np.array(
            [
                [0.0, 0.0],
                [0.0, float(w - 1)],
                [float(h - 1), 0.0],
                [float(h - 1), float(w - 1)],
            ],
            dtype=np.float64,
        )
        input_pts_np = np.ascontiguousarray(np.concatenate([input_pts_np, corners], axis=0))
        output_pts_np = np.ascontiguousarray(np.concatenate([output_pts_np, corners], axis=0))

    # points are [y,x], corners already included
    return (input_pts_np, output_pts_np)