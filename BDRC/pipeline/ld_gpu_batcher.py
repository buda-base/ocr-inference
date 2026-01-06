import asyncio
from dataclasses import dataclass
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

import torch
import torch.nn.functional as F

from .types_common import DecodedFrame, PipelineError, EndOfStream, InferredFrame
from .img_helpers import adaptive_binarize

# -----------------------------
# Internal bookkeeping structs
# -----------------------------
@dataclass
class _PendingFrame:
    """
    Represents one input image being processed (possibly split into many tiles).

    We store enough metadata to stitch tile predictions back into a full-size mask.
    """
    frame_id: int
    lane_second_pass: bool

    # Original input (kept as-is for packaging in InferredFrame)
    orig_frame: Any
    orig_is_binary: bool

    # Shapes/padding
    orig_h: int
    orig_w: int
    pad_y: int
    pad_x: int
    h_pad: int
    w_pad: int

    # Tiling geometry
    patch_size: int
    x_starts: List[int]
    y_starts: List[int]

    # Accumulator on GPU for stitching (max in overlaps)
    accum_soft: torch.Tensor  # [1, H_pad, W_pad], float32 on device

    # Progress tracking
    expected_tiles: int
    received_tiles: int = 0


@dataclass
class _TileWorkItem:
    """
    One tile that belongs to a pending frame.
    """
    frame_id: int
    tile_index: int
    x0: int
    y0: int
    tile_1ch: torch.Tensor  # [1, 512, 512], float32 on device


class LDGpuBatcher:
    """
    Two-lane GPU micro-batcher and inference runner.

    Inputs:
      - q_init: DecodedFrame (first pass)
      - q_re:   DecodedFrame (second pass; higher priority)
    Outputs:
      - q_gpu_pass_1_to_post_processor and q_gpu_pass_2_to_post_processor: InferredFrame
    """

    def __init__(
        self,
        cfg,
        q_decoder_to_gpu_pass_1: asyncio.Queue,
        q_post_processor_to_gpu_pass_2: asyncio.Queue,
        q_gpu_pass_1_to_post_processor: asyncio.Queue,
        q_gpu_pass_2_to_post_processor: asyncio.Queue,
    ):
        self.cfg = cfg
        self.q_init = q_decoder_to_gpu_pass_1
        self.q_re = q_post_processor_to_gpu_pass_2
        self.q_gpu_pass_1_to_post_processor = q_gpu_pass_1_to_post_processor
        self.q_gpu_pass_2_to_post_processor = q_gpu_pass_2_to_post_processor

        self._init_done = False
        self._re_done = False

        # Device/model configuration
        self.device = "cuda"
        # TODO: import model
        self.model = getattr(cfg, "model", None)
        if self.model is None:
            raise ValueError("cfg.model must be set (torch.nn.Module)")
        self.model.to(self.device)
        self.model.eval()

        self.line_class_index: int = 0

        self.batch_timeout_s: float = cfg.batch_timeout_ms / 1000.0

        # Internal buffers
        self._next_frame_id: int = 1
        self._pending_frames: Dict[int, _PendingFrame] = {}
        self._tile_pool: Deque[_TileWorkItem] = deque()

        # For "images" mode: keep pending frames in arrival order so we can group them
        self._image_queue: Deque[int] = deque()

    async def _pop_one(self, q: asyncio.Queue, timeout_s: float):
        try:
            return await asyncio.wait_for(q.get(), timeout=timeout_s)
        except asyncio.TimeoutError:
            return None  # means "no item right now", NOT end-of-stream

    # -----------------------------
    # Main scheduler loop
    # -----------------------------
    async def run(self):
        # Prefer reprocess lane, but don't starve init
        reprocess_budget = self.cfg.reprocess_budget
        init_budget = 1

        while True:
            # termination condition: both lanes ended (and any internal buffers flushed)
            if self._init_done and self._re_done:
                await self._flush()
                await self.q_gpu_pass_1_to_post_processor.put(EndOfStream(stream="gpu_pass_1", producer="LDGpuBatcher"))
                await self.q_gpu_pass_2_to_post_processor.put(EndOfStream(stream="gpu_pass_2", producer="LDGpuBatcher"))
                return

            took_any = False

            # TODO: re-do with less blocking

            # --- prefer reprocess lane (from LDPostProcessor) ---
            if not self._re_done:
                for _ in range(reprocess_budget):
                    msg = await self._pop_one(self.q_re, self.batch_timeout_s)
                    if msg is None:
                        break
                    took_any = True

                    if isinstance(msg, EndOfStream) and msg.stream == "transformed_pass_1":
                        self._re_done = True
                        break
                    if isinstance(msg, PipelineError):
                        await self.q_gpu_pass_2_to_post_processor.put(msg)
                        continue

                    # enqueue + maybe process a batch
                    await self._enqueue_decoded_frame(msg, second_pass=True)
                    await self._maybe_process_batches()

                if took_any:
                    continue

            # --- then init lane (from Decoder) ---
            if not self._init_done:
                for _ in range(init_budget):
                    msg = await self._pop_one(self.q_init, self.batch_timeout_s)
                    if msg is None:
                        break
                    took_any = True

                    if isinstance(msg, EndOfStream) and msg.stream == "decoded":
                        self._init_done = True
                        break
                    if isinstance(msg, PipelineError):
                        await self.q_gpu_pass_1_to_post_processor.put(msg)
                        continue

                    await self._enqueue_decoded_frame(msg, second_pass=False)
                    await self._maybe_process_batches()

            # If neither lane had work, still give the batcher a chance to flush partial batches.
            # This is important to keep latency bounded when traffic is low.
            if not took_any:
                await self._maybe_process_batches(force_on_timeout=True)
                await asyncio.sleep(0)

    # -----------------------------
    # Frame ingestion
    # -----------------------------
    async def _enqueue_decoded_frame(self, dec_frame: DecodedFrame, second_pass: bool) -> None:
        """
        Convert frame -> (optional binarize on CPU) -> torch on GPU -> padded -> tiled -> register work items.
        """
        # Validate expected input type (keep errors explicit and early)
        gray = dec_frame.frame
        if not isinstance(gray, np.ndarray) or gray.ndim != 2 or gray.dtype != np.uint8:
            raise ValueError("DecodedFrame.frame must be a 2D numpy uint8 array (H, W)")

        # Optional binarization on CPU (OpenCV). This is typically fast and avoids
        # re-implementing Gaussian adaptive thresholding on GPU.
        is_binary = bool(dec_frame.is_binary)
        if not is_binary:
            gray = adaptive_binarize(
                gray,
                block_size=self.cfg.binarize_block_size,
                c=self.cfg.binarize_c,
            )
            is_binary = True  # after adaptiveThreshold, values are {0, 255}

        h, w = int(gray.shape[0]), int(gray.shape[1])

        # Convert to torch on CPU first, then to GPU.
        # Keep it uint8 until we are on GPU to reduce PCIe bandwidth a bit.
        t_u8 = torch.from_numpy(gray)  # [H, W], uint8, CPU
        t_u8 = t_u8.unsqueeze(0)       # [1, H, W]

        # Move to GPU and normalize to [0, 1] float32
        t = t_u8.to(self.device, non_blocking=True).float().div_(255.0)  # [1, H, W] float32

        # Compute padding + tiling starts
        x_starts, y_starts, pad_x, pad_y, h_pad, w_pad = self._compute_tiling_geometry(h, w)

        # Pad with "white" background = 1.0 (original uses pad value 255 for uint8)
        # F.pad order for 3D [C,H,W] is (pad_left, pad_right, pad_top, pad_bottom)
        t_pad = F.pad(t, (0, pad_x, 0, pad_y), value=1.0)  # [1, H_pad, W_pad]

        # Allocate accumulator for stitching predictions on GPU.
        # We'll use max() in overlaps (very robust for segmentation probabilities).
        accum = torch.zeros((1, h_pad, w_pad), device=self.device, dtype=torch.float32)

        frame_id = self._next_frame_id
        self._next_frame_id += 1

        expected_tiles = len(x_starts) * len(y_starts)
        pending = _PendingFrame(
            frame_id=frame_id,
            lane_second_pass=second_pass,
            orig_frame=dec_frame.frame,      # keep original (not necessarily binarized) for output packaging
            orig_is_binary=bool(dec_frame.is_binary),
            orig_h=h,
            orig_w=w,
            pad_y=pad_y,
            pad_x=pad_x,
            h_pad=h_pad,
            w_pad=w_pad,
            patch_size=self.cfg.patch_size,
            x_starts=x_starts,
            y_starts=y_starts,
            accum_soft=accum,
            expected_tiles=expected_tiles,
            received_tiles=0,
        )
        self._pending_frames[frame_id] = pending

        if self.cfg.batch_type == "images":
            # Record frame order for grouping
            self._image_queue.append(frame_id)

        # Create tile work items
        tile_index = 0
        for y0 in y_starts:
            for x0 in x_starts:
                tile_1ch = t_pad[:, y0 : y0 + self.cfg.patch_size, x0 : x0 + self.cfg.patch_size]  # [1,512,512]
                self._tile_pool.append(
                    _TileWorkItem(
                        frame_id=frame_id,
                        tile_index=tile_index,
                        x0=x0,
                        y0=y0,
                        tile_1ch=tile_1ch,
                    )
                )
                tile_index += 1

    def _compute_tiling_geometry(self, h: int, w: int) -> Tuple[List[int], List[int], int, int, int, int]:
        """
        Horizontal overlap:
          step_x = ps - oh
          x starts at 0, step_x, 2*step_x, ... and we "snap" the last tile to x = w - ps if needed
          (so the last overlap can be larger than oh). No pad_x needed for coverage when w > ps.

        Vertical overlap:
          step_y = ps - ov
          y starts at 0, step_y, 2*step_y, ... and we "snap" the last tile to y = h - ps if needed
          (so the last overlap can be larger than ov). No pad_y needed for coverage when h > ps.

        Padding is only used when the image is smaller than a single patch in that dimension.
        """
        ps = self.cfg.patch_size
        ov = self.cfg.patch_vertical_overlap_px
        oh = self.cfg.patch_horizontal_overlap_px

        # --- Validate strides ---
        step_x = ps - oh
        if step_x <= 0:
            raise ValueError("Invalid horizontal overlap; step_x must be > 0 (require oh < ps)")

        step_y = ps - ov
        if step_y <= 0:
            raise ValueError("Invalid vertical overlap; step_y must be > 0 (require ov < ps)")

        # --- X axis: overlap + snap last tile to the right edge ---
        if w <= ps:
            # Need at least one tile; pad to reach ps width
            x_starts = [0]
            pad_x = ps - w
            w_pad = ps
        else:
            # Regular starts that fit, then snap one tile to end if needed
            x_starts = list(range(0, w - ps + 1, step_x))
            last_start = x_starts[-1]
            last_end = last_start + ps
            if last_end < w:
                x_starts.append(w - ps)

            # No padding needed for coverage
            pad_x = 0
            w_pad = w

            # Safety clamp (should be redundant)
            x_starts[-1] = min(x_starts[-1], w - ps)

        # --- Y axis: overlap + snap last tile to the bottom edge ---
        if h <= ps:
            y_starts = [0]
            pad_y = ps - h
            h_pad = ps
        else:
            y_starts = list(range(0, h - ps + 1, step_y))
            last_start = y_starts[-1]
            last_end = last_start + ps
            if last_end < h:
                y_starts.append(h - ps)

            pad_y = 0
            h_pad = h

            # Safety clamp (should be redundant)
            y_starts[-1] = min(y_starts[-1], h - ps)

        return x_starts, y_starts, pad_x, pad_y, h_pad, w_pad


    # -----------------------------
    # Batch processing
    # -----------------------------
    async def _maybe_process_batches(self, force_on_timeout: bool = False) -> None:
        """
        Decide whether to run inference now.

        - In "tiles" mode: run when we have tiles_batch_n tiles (or if force_on_timeout and any tiles exist).
        - In "images" mode: run when we have image_batch_n images (or if force_on_timeout and any images exist).
        """
        if self.cfg.batch_type == "tiles":
            if len(self._tile_pool) >= self.cfg.tiles_batch_n or (force_on_timeout and len(self._tile_pool) > 0):
                await self._process_one_tiles_batch(min(self.cfg.tiles_batch_n, len(self._tile_pool)))
        elif self.cfg.batch_type == "images":
            if len(self._image_queue) >= self.cfg.image_batch_n or (force_on_timeout and len(self._image_queue) > 0):
                await self._process_one_images_batch(min(self.cfg.image_batch_n, len(self._image_queue)))
        else:
            raise ValueError("cfg.batch_type must be either 'tiles' or 'images'")

    async def _process_one_tiles_batch(self, batch_size: int) -> None:
        """
        Pop up to batch_size tiles across any frames, run model, scatter + stitch.
        """
        items: List[_TileWorkItem] = []
        for _ in range(batch_size):
            items.append(self._tile_pool.popleft())

        # Stack tiles into [B, 1, 512, 512], then expand to 3 channels without copy.
        tiles_1 = torch.stack([it.tile_1ch for it in items], dim=0)  # [B,1,512,512]
        tiles_3 = tiles_1.expand(-1, 3, -1, -1)                     # [B,3,512,512] view

        soft = self._infer_tiles_to_soft(tiles_3)  # [B,1,512,512] float32 on device

        # Scatter each tile prediction into its pending frame accumulator
        for i, it in enumerate(items):
            pending = self._pending_frames.get(it.frame_id)
            if pending is None:
                # Shouldn't happen, but don't crash the pipeline for one corrupted state.
                continue

            # Max-stitch into accumulator (robust in overlap regions)
            y0, x0 = it.y0, it.x0
            region = pending.accum_soft[:, y0 : y0 + self.cfg.patch_size, x0 : x0 + self.cfg.patch_size]
            pending.accum_soft[:, y0 : y0 + self.cfg.patch_size, x0 : x0 + self.cfg.patch_size] = torch.maximum(
                region, soft[i]
            )

            pending.received_tiles += 1

        # Finalize any frames that are now complete
        await self._finalize_completed_frames()

    async def _process_one_images_batch(self, image_batch_size: int) -> None:
        """
        Pop up to image_batch_size frames, gather ALL their tiles, run ONE big model call,
        scatter + stitch. This minimizes model launches and is closest to your reference pipeline.
        """
        frame_ids: List[int] = []
        for _ in range(image_batch_size):
            frame_ids.append(self._image_queue.popleft())

        # Gather all tiles for these frames.
        # IMPORTANT: we keep "other frames" tiles in the pool untouched.
        # To do this efficiently without complex indexing, we do a simple pass that
        # extracts relevant tiles and keeps the rest.
        selected: List[_TileWorkItem] = []
        remaining: Deque[_TileWorkItem] = deque()

        selected_set = set(frame_ids)
        while self._tile_pool:
            it = self._tile_pool.popleft()
            if it.frame_id in selected_set:
                selected.append(it)
            else:
                remaining.append(it)
        self._tile_pool = remaining

        if not selected:
            return

        # Stack + infer
        tiles_1 = torch.stack([it.tile_1ch for it in selected], dim=0)  # [T,1,512,512]
        tiles_3 = tiles_1.expand(-1, 3, -1, -1)                         # [T,3,512,512]

        soft = self._infer_tiles_to_soft(tiles_3)  # [T,1,512,512]

        # Scatter back
        for i, it in enumerate(selected):
            pending = self._pending_frames.get(it.frame_id)
            if pending is None:
                continue
            y0, x0 = it.y0, it.x0
            region = pending.accum_soft[:, y0 : y0 + self.cfg.patch_size, x0 : x0 + self.cfg.patch_size]
            pending.accum_soft[:, y0 : y0 + self.cfg.patch_size, x0 : x0 + self.cfg.patch_size] = torch.maximum(
                region, soft[i]
            )
            pending.received_tiles += 1

        await self._finalize_completed_frames()

    def _infer_tiles_to_soft(self, tiles_3: torch.Tensor) -> torch.Tensor:
        """
        Runs the model on tiles and returns sigmoid probabilities for the selected class.

        tiles_3: [B, 3, 512, 512] float32 on device
        returns: [B, 1, 512, 512] float32 on device
        """
        with torch.inference_mode():
            logits = self.model(tiles_3)

            # Handle either [B,1,H,W] or [B,C,H,W]
            if logits.ndim != 4:
                raise ValueError(f"Model output must be 4D [B,C,H,W], got shape={tuple(logits.shape)}")

            # TODO: review by Eric
            if logits.shape[1] == 1:
                sel = logits  # [B,1,H,W]
            else:
                idx = self.line_class_index
                if idx < 0 or idx >= logits.shape[1]:
                    idx = 0
                sel = logits[:, idx : idx + 1, :, :]  # keep channel dim

            soft = torch.sigmoid(sel).to(torch.float32)
            return soft

    async def _finalize_completed_frames(self) -> None:
        """
        Emit InferredFrame for any pending frames whose tiles are all received.
        """
        done_ids: List[int] = []
        for frame_id, pending in self._pending_frames.items():
            if pending.received_tiles >= pending.expected_tiles:
                done_ids.append(frame_id)

        for frame_id in done_ids:
            pending = self._pending_frames.pop(frame_id, None)
            if pending is None:
                continue

            # Crop padding if any (in this implementation pad_y is currently 0 except small images,
            # but pad_x may exist; keep it general and correct).
            mask_soft = pending.accum_soft  # [1, H_pad, W_pad]
            if pending.pad_y > 0:
                mask_soft = mask_soft[:, : pending.orig_h, :]
            if pending.pad_x > 0:
                mask_soft = mask_soft[:, :, : pending.orig_w]

            # Threshold -> uint8 {0,255} on GPU, then move to CPU
            binary = (mask_soft > self.cfg.class_threshold).to(torch.uint8) * 255  # [1,H,W]
            line_mask_np = binary.squeeze(0).cpu().numpy()  # [H,W], uint8

            out = InferredFrame(
                frame=pending.orig_frame,
                is_binary=pending.orig_is_binary,
                line_mask=line_mask_np,
            )

            if pending.lane_second_pass:
                await self.q_gpu_pass_2_to_post_processor.put(out)
            else:
                await self.q_gpu_pass_1_to_post_processor.put(out)

    # -----------------------------
    # Flush
    # -----------------------------
    async def _flush(self) -> None:
        """
        Ensure we process everything remaining in internal buffers.

        This must:
          - run remaining partial batches
          - finalize any pending frames
        """
        # Keep processing until no work remains.
        # In practice, this loops only a few times.
        while self._tile_pool or self._image_queue:
            if self.cfg.batch_type == "tiles":
                # Process all remaining tiles in chunks
                n = min(self.cfg.tiles_batch_n, len(self._tile_pool))
                if n == 0:
                    break
                await self._process_one_tiles_batch(n)
            else:
                # Process all remaining images in chunks
                n = min(self.cfg.image_batch_n, len(self._image_queue))
                if n == 0:
                    break
                await self._process_one_images_batch(n)

        # Safety net: if any frames are somehow left (shouldn't happen), try to finalize.
        await self._finalize_completed_frames()
