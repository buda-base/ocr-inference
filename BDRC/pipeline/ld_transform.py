
import asyncio
from .types import DecodedFrame, Record, InferredFrame
from typing import Any, Dict, List, Optional, Sequence, Tuple
from .img_helpers import apply_transform_1

import cv2
import numpy as np
import numpy.typing as npt

class TransformController:
        """Consumes InferredFrame, decides and apply rotation/TPS, and routes.

        Input: InferredFrame from GPU first pass.
        Output: Either
           - a final Record to the writer, or 
           - a transformed (rotated + TPS) DecodedFrame enqueued to the reprocess lane.
        """
    def __init__(self, cfg, q_first_results: asyncio.Queue, q_reprocess_frames: asyncio.Queue, q_records: asyncio.Queue):
        self.cfg = cfg
        self.q_first_results = q_first_results
        self.q_reprocess_frames = q_reprocess_frames
        self.q_records = q_records

    async def run(self):
        """Main loop: read summaries, decide, write or re-enqueue; propagate sentinel."""
        while True:
            frame: InferredFrame = await self.q_first_results.get()
            if item is None:
                await self.q_reprocess_frames.put(None)
                break
            # 1. detect contours
            contours = get_filtered_contours(frame.line_mask)
            # 2. get rotation angle, if non-0, rotate contours
            h, w = frame.line_mask.shape
            rotation_angle = get_rotation_angle(contours, h, w, max_angle_deg=self.cfg.max_angle_deg, min_angle_deg=self.cfg.min_angle_deg)
            if rotation_angle != 0.0:
                contours = rotate_contours(contours, rotation_angle, h, w)
            # 3. get tps data
            tps_data = get_tps_points(contours, h, w, legacy_tps_detect=self.cfg.legacy_tps_detect, alpha=self.cfg.tps_alpha, tps_add_corners=self.cfg.add_corners)
            input_pts, output_pts, alpha = None
            if tps_data is not None:
                input_pts, output_pts = tps_data
                alpha = self.cfg.tps_alpha
            # 4. if no second pass required, send record
            if tps_data is None and rotation_angle == 0.0:
                # save contour bboxes (?)
                contours_bboxes = get_contour_bboxes(contours)
                rec = Record(
                    img_filename=frame.img_filename,
                    s3_etag=frame.s3_etag,
                    resized_w=w,
                    resized_h=h,
                    rotation_angle=None,
                    tps_data=None,
                    contours=contours,
                    nb_contours=len(contours),
                    contours_bboxes=contours_bboxes
                )
                frame = None # gc
                await self.q_records.put(rec)
            else:
                # or if a second pass is required, create a new decodedframe by transforming the initial frame
                transformed_frame = apply_transform_1(frame.frame, rotation_angle, input_pts, output_pts, alpha)
                new_decoded_frame = DecodedFrame(
                    img_filename=frame.img_filename,
                    s3_etag=frame.s3_etag,
                    frame=transformed_frame,
                    first_pass=False,
                    is_binary=False,
                    rotation_angle=rotation_angle,
                    tps_data=(input_pts, output_pts, alpha),
                )
                frame = None # gc
                await self.q_reprocess_frames.put(new_decoded_frame)

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
    (cx, cy), _, _ = cv2.minAreaRect(biggest)
    return start_x + int(cx), bbox_y + int(cy), int(bh)


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
) -> Optional[Dict[str, Any]]:
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

    input_pts = np.asarray(input_pts, dtype=np.float64)
    output_pts = np.asarray(output_pts, dtype=np.float64)

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
        input_pts = np.concatenate([input_pts, corners], axis=0)
        output_pts = np.concatenate([output_pts, corners], axis=0)

    # points are [y,x], corners already included
    return (input_pts, output_pts)
