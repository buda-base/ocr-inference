import asyncio
import traceback
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

    We keep enough metadata from DecodedFrame to emit correct outputs and errors.
    """
    frame_id: int
    lane_second_pass: bool

    # Propagated metadata from DecodedFrame
    task: Any  # ImageTask
    source_etag: Optional[str]
    first_pass: bool
    rotation_angle: Optional[float]
    tps_data: Optional[Any]

    # The (possibly binarized) frame used for inference (H,W uint8)
    frame: Any
    is_binary: bool

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
        self._p1_eos_sent = False

        # Device/model configuration
        self.device = "cuda"
        self.model = getattr(cfg, "model", None)
        if self.model is None:
            raise ValueError("cfg.model must be set (torch.nn.Module)")
        if not isinstance(self.model, torch.nn.Module):
            raise TypeError(f"cfg.model must be torch.nn.Module, got {type(self.model)}")
        self.model.to(self.device)
        self.model.eval()

        self.line_class_index: int = 0
        self._accum_dtype = torch.float16

        self.batch_timeout_s: float = cfg.batch_timeout_ms / 1000.0


        # CUDA streams for overlap
        # - compute happens on the default stream
        # - D2H copies for finalized masks happen on a dedicated stream so we can overlap
        #   the next tile batch compute with output transfer.
        self._d2h_stream = torch.cuda.Stream(device=self.device) if torch.cuda.is_available() else None
        self._d2h_inflight_limit = int(getattr(cfg, "d2h_inflight_limit", 2))
        self._d2h_inflight = 0
        # Internal buffers
        self._next_frame_id: int = 1
        self._pending_frames: Dict[int, _PendingFrame] = {}
        # Tile pool: naturally bounded by queue backpressure (see _enqueue_decoded_frame docstring)
        self._tile_pool: Deque[_TileWorkItem] = deque()
        
        # PyTorch profiler (optional)
        # NOTE: The most interpretable workflow is:
        #   - keep ONE profiler context open during the run loop,
        #   - advance it with prof.step() once per model invocation,
        #   - use a schedule to capture a short representative window,
        #   - write traces for TensorBoard (much easier than raw Chrome/Perfetto).
        self._profiler: Optional[Any] = None
        self._profiler_started: bool = False
        self._profiler_step: int = 0

        self._profiler_enabled = getattr(cfg, "enable_pytorch_profiler", False)
        if self._profiler_enabled:
            try:
                import os
                prof_dir = getattr(self.cfg, "profiler_dir", "tb_profiler_logs")
                os.makedirs(prof_dir, exist_ok=True)

                # Keep stacks/shapes off by default: they add a LOT of events and make traces hard to read.
                prof_with_stack = getattr(self.cfg, "profiler_with_stack", False)
                prof_record_shapes = getattr(self.cfg, "profiler_record_shapes", False)
                prof_profile_memory = getattr(self.cfg, "profiler_profile_memory", True)

                # Capture only a short window (wait/warmup/active) unless you explicitly change these.
                wait = int(getattr(self.cfg, "profiler_wait", 1))
                warmup = int(getattr(self.cfg, "profiler_warmup", 1))
                active = int(getattr(self.cfg, "profiler_active", 3))
                repeat = int(getattr(self.cfg, "profiler_repeat", 1))

                sched = torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)

                self._profiler = torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    schedule=sched,
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(prof_dir),
                    record_shapes=prof_record_shapes,
                    profile_memory=prof_profile_memory,
                    with_stack=prof_with_stack,
                )
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to initialize PyTorch profiler: {e}")
                self._profiler_enabled = False
                self._profiler = None

    async def _throttle_tile_pool(self, incoming_tiles: int) -> None:
        """
        Backpressure safety: if cfg.max_tiles_in_pool is set, avoid unbounded growth
        of the internal tile pool under load (which can hold onto large GPU tensors).
        """
        max_tiles = self.cfg.max_tiles_in_pool
        if max_tiles <= 0:
            return
        if incoming_tiles <= 0:
            return
        if incoming_tiles > max_tiles:
            raise ValueError(
                f"incoming_tiles ({incoming_tiles}) exceeds max_tiles_in_pool ({max_tiles}); "
                f"increase max_tiles_in_pool or reduce tiling (patch_size/overlaps)."
            )

        # Keep draining until there is room for the incoming tiles.
        # We force processing even for partial batches to guarantee progress under low traffic.
        while (len(self._tile_pool) + incoming_tiles) > max_tiles:
            await self._maybe_process_batches(force_on_timeout=True)
            await asyncio.sleep(0)


    # -----------------------------
    # Error handling helpers
    # -----------------------------
    @staticmethod
    def _is_cuda_oom(exc: BaseException) -> bool:
        msg = str(exc).lower()
        return "out of memory" in msg or "cuda out of memory" in msg

    def _maybe_reinit_gpu(self, reason: str) -> None:
        """Best-effort GPU/model reinit hook (opt-in via cfg)."""
        try:
            do_on_oom = bool(getattr(self.cfg, "gpu_reinit_on_oom", False))
            do_on_err = bool(getattr(self.cfg, "gpu_reinit_on_error", False))
            if not (do_on_oom or do_on_err):
                return
            if (not do_on_err) and (reason != "oom"):
                return

            try:
                self.model.to("cpu")
            except Exception:
                pass

            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
            except Exception:
                pass

            try:
                self.model.to(self.device)
                self.model.eval()
            except Exception:
                pass
        except Exception:
            return

    async def _emit_pipeline_error(
        self,
        *,
        internal_stage: str,
        exc: BaseException,
        lane_second_pass: bool,
        task: Any,
        source_etag: Optional[str],
        retryable: bool = False,
        attempt: int = 1,
        timeout_s: float = 5.0,
    ) -> None:
        import logging
        logger = logging.getLogger(__name__)
        
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        err = PipelineError(
            stage="LDGpuBatcher",
            task=task,
            source_etag=source_etag,
            error_type=type(exc).__name__,
            message=f"[{internal_stage}] {exc}",
            traceback=tb,
            retryable=bool(retryable),
            attempt=int(attempt),
        )
        target_q = self.q_gpu_pass_2_to_post_processor if lane_second_pass else self.q_gpu_pass_1_to_post_processor
        
        try:
            await asyncio.wait_for(target_q.put(err), timeout=timeout_s)
        except asyncio.TimeoutError:
            # Log critical - queue is stuck, but don't block the pipeline
            logger.critical(
                f"Failed to emit error after {timeout_s}s: queue full. "
                f"Dropping error for {task.img_filename if task else 'unknown'}"
            )

    def _drop_frame(self, frame_id: int) -> None:
        """Remove any partial state for a frame to avoid deadlocks/leaks after an error."""
        if frame_id in self._pending_frames:
            pending = self._pending_frames[frame_id]
            # Explicitly free GPU memory
            if hasattr(pending, 'accum_soft') and pending.accum_soft is not None:
                del pending.accum_soft
            del self._pending_frames[frame_id]

        if self._tile_pool:
            self._tile_pool = deque([t for t in self._tile_pool if t.frame_id != frame_id])


    async def _emit_inferred_after_d2h(
        self,
        *,
        pending: _PendingFrame,
        packed_host: torch.Tensor,
        done_evt: Optional[torch.cuda.Event],
        h: int,
        w: int,
        pad: int,
    ) -> None:
        """Wait for async D2H copy to complete, then emit InferredFrame.

        This runs as a background asyncio task so the main batching loop can keep feeding the GPU.
        """
        try:
            if done_evt is not None:
                # Avoid blocking the event loop thread.
                await asyncio.to_thread(done_evt.synchronize)

            packed_np = packed_host.numpy()
            line_mask_packed = ("packedbits", packed_np, int(h), int(w), int(pad))

            out = InferredFrame(
                task=pending.task,
                source_etag=pending.source_etag,
                frame=pending.frame,
                orig_h=pending.orig_h,
                orig_w=pending.orig_w,
                is_binary=pending.is_binary,
                first_pass=pending.first_pass,
                rotation_angle=pending.rotation_angle,
                tps_data=pending.tps_data,
                line_mask=line_mask_packed,
            )

            if pending.lane_second_pass:
                await self.q_gpu_pass_2_to_post_processor.put(out)
            else:
                await self.q_gpu_pass_1_to_post_processor.put(out)

        except Exception as e:
            retryable = self._is_cuda_oom(e)
            if retryable:
                self._maybe_reinit_gpu("oom")
            await self._emit_pipeline_error(
                internal_stage="finalize.emit_after_d2h",
                exc=e,
                lane_second_pass=bool(pending.lane_second_pass),
                task=pending.task,
                source_etag=pending.source_etag,
                retryable=bool(retryable),
                attempt=1,
            )
        finally:
            self._d2h_inflight = max(0, self._d2h_inflight - 1)

    async def _pop_one(self, q: asyncio.Queue, timeout_s: float):
        try:
            return await asyncio.wait_for(q.get(), timeout=timeout_s)
        except asyncio.TimeoutError:
            return None  # means "no item right now", NOT end-of-stream

    # -----------------------------
    # Main scheduler loop
    # -----------------------------
    async def run(self) -> None:
        """Main scheduler loop for GPU batching."""
        # Prefer reprocess lane, but don't starve init
        reprocess_budget = self.cfg.reprocess_budget
        init_budget = 1


        # Start profiler once for the whole run loop (much more readable than re-entering per inference).
        if self._profiler_enabled and self._profiler is not None and not self._profiler_started:
            try:
                self._profiler.__enter__()
                self._profiler_started = True
            except Exception:
                # If profiler fails mid-run, disable it rather than crashing the scheduler.
                self._profiler_enabled = False
                self._profiler = None
                self._profiler_started = False


        try:
            while True:
                # termination condition: both lanes ended (and any internal buffers flushed)
                if self._init_done and self._re_done:
                    await self._flush()
                    if not self._p1_eos_sent:
                        await self.q_gpu_pass_1_to_post_processor.put(
                            EndOfStream(stream="gpu_pass_1", producer="LDGpuBatcher")
                        )
                        self._p1_eos_sent = True
                    await self.q_gpu_pass_2_to_post_processor.put(EndOfStream(stream="gpu_pass_2", producer="LDGpuBatcher"))
                    break

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
                        try:
                            await self._enqueue_decoded_frame(msg, second_pass=True)
                            await self._maybe_process_batches()
                        except Exception as e:
                            retryable = self._is_cuda_oom(e)
                            if retryable:
                                self._maybe_reinit_gpu("oom")
                            await self._emit_pipeline_error(
                                internal_stage="run.reprocess_message",
                                exc=e,
                                lane_second_pass=True,
                                task=msg.task,
                                source_etag=msg.source_etag,
                                retryable=retryable,
                                attempt=1,
                            )

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

                        # enqueue + maybe process a batch
                        try:
                            await self._enqueue_decoded_frame(msg, second_pass=False)
                            await self._maybe_process_batches()
                        except Exception as e:
                            retryable = self._is_cuda_oom(e)
                            if retryable:
                                self._maybe_reinit_gpu("oom")
                            await self._emit_pipeline_error(
                                internal_stage="run.init_message",
                                exc=e,
                                lane_second_pass=False,
                                task=msg.task,
                                source_etag=msg.source_etag,
                                retryable=retryable,
                                attempt=1,
                            )

                # If neither lane had work, still give the batcher a chance to flush partial batches.
                # This is important to keep latency bounded when traffic is low.
                if not took_any:
                    await self._maybe_process_batches(force_on_timeout=True)
                    await self._maybe_emit_gpu_pass_1_eos()
                    await asyncio.sleep(0)
                else:
                    # We made progress; it's still a good time to check whether the init lane fully drained.
                    await self._maybe_emit_gpu_pass_1_eos()
        finally:
            # Finalize profiler & emit summaries/traces (if enabled)
            if self._profiler_enabled and self._profiler is not None:
                import logging
                logger = logging.getLogger(__name__)
                try:
                    # Make sure all queued CUDA work is finished so the trace is complete/readable.
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                except Exception:
                    pass

                # Close the profiler context if we opened it in run()
                if self._profiler_started:
                    try:
                        self._profiler.__exit__(None, None, None)
                    except Exception as e:
                        logger.warning(f"Failed to close PyTorch profiler cleanly: {e}")
                    finally:
                        self._profiler_started = False

                # Print operator summaries directly in logs (fastest way to answer 'what dominates?')
                try:
                    ka = self._profiler.key_averages()
                    logger.info("=== PyTorch Profiler: top ops by CUDA time ===\n" +
                                ka.table(sort_by="cuda_time_total", row_limit=30))
                    logger.info("=== PyTorch Profiler: top ops by self CPU time ===\n" +
                                ka.table(sort_by="self_cpu_time_total", row_limit=30))
                    if getattr(self.cfg, "profiler_profile_memory", True):
                        logger.info("=== PyTorch Profiler: top ops by CUDA memory usage ===\n" +
                                    ka.table(sort_by="cuda_memory_usage", row_limit=30))
                except Exception as e:
                    logger.warning(f"Failed to compute profiler summaries: {e}")

                # Optional: export a single Chrome trace (useful, but TensorBoard is usually better)
                try:
                    if getattr(self.cfg, "profiler_export_chrome", True):
                        output_path = getattr(self.cfg, "profiler_trace_output", None) or "pytorch_trace.json"
                        self._profiler.export_chrome_trace(output_path)
                        logger.info(f"PyTorch profiler Chrome trace exported to: {output_path}")
                        logger.info(f"Open in Chrome: chrome://tracing (load {output_path})")
                        prof_dir = getattr(self.cfg, "profiler_dir", "tb_profiler_logs")
                        logger.info(f"TensorBoard traces (recommended): {prof_dir}  (run: tensorboard --logdir {prof_dir})")
                except Exception as e:
                    logger.warning(f"Failed to export profiler trace: {e}")
            
            # Strategic GPU cache clearing at end of processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _has_pending_init_lane_work(self) -> bool:
        """
        Return True if there is any remaining *first-pass lane* work that could still produce
        messages on q_gpu_pass_1_to_post_processor.
        """
        # Pending frames in init lane
        if any((not p.lane_second_pass) for p in self._pending_frames.values()):
            return True

        # Tiles still enqueued for any init-lane pending frame
        for t in self._tile_pool:
            p = self._pending_frames.get(t.frame_id)
            if p is not None and (not p.lane_second_pass):
                return True

        return False

    async def _maybe_emit_gpu_pass_1_eos(self) -> None:
        """
        Break the end-of-stream cycle between batcher and post-processor:

        - PostProcessor only sends EndOfStream(stream="transformed_pass_1") after it learns that
          gpu_pass_1 is finished.
        - Batcher previously only sent gpu_pass_1 EOS at final shutdown, but final shutdown required
          transformed_pass_1 EOS -> deadlock.

        We therefore emit gpu_pass_1 EOS as soon as the init lane is *fully drained*.
        """
        if self._p1_eos_sent:
            return
        if not self._init_done:
            return
        if self._has_pending_init_lane_work():
            return

        await self.q_gpu_pass_1_to_post_processor.put(EndOfStream(stream="gpu_pass_1", producer="LDGpuBatcher"))
        self._p1_eos_sent = True

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

        # Compute padding + tiling starts (CPU-only) and throttle BEFORE allocating GPU tensors.
        x_starts, y_starts, pad_x, pad_y, h_pad, w_pad = self._compute_tiling_geometry(h, w)
        expected_tiles = len(x_starts) * len(y_starts)
        await self._throttle_tile_pool(expected_tiles)

        # Convert to torch on CPU first, then to GPU.
        # Keep it uint8 until we are on GPU to reduce PCIe bandwidth a bit.
        #
        # pin the CPU tensor so the subsequent H2D copy can
        # actually leverage non_blocking=True (async DMA).
        t_u8 = torch.from_numpy(gray)  # [H, W], uint8, CPU (shares memory with numpy)
        if torch.cuda.is_available():
            # pin_memory() returns a new tensor backed by pinned (page-locked) memory.
            t_u8 = t_u8.pin_memory()
        t_u8 = t_u8.unsqueeze(0)       # [1, H, W]

        # Move to GPU and normalize to [0, 1] float32
        t = t_u8.to(self.device, non_blocking=True).float().div_(255.0)  # [1, H, W] float32

        # Pad with "white" background = 1.0 (original uses pad value 255 for uint8)
        # F.pad order for 3D [C,H,W] is (pad_left, pad_right, pad_top, pad_bottom)
        t_pad = F.pad(t, (0, pad_x, 0, pad_y), value=1.0)  # [1, H_pad, W_pad]

        # Allocate accumulator for stitching predictions on GPU.
        # We'll use max() in overlaps (very robust for segmentation probabilities).
        accum = torch.zeros((1, h_pad, w_pad), device=self.device, dtype=self._accum_dtype)

        frame_id = self._next_frame_id
        self._next_frame_id += 1

        pending = _PendingFrame(
            frame_id=frame_id,
            lane_second_pass=second_pass,

            task=dec_frame.task,
            source_etag=dec_frame.source_etag,
            first_pass=bool(dec_frame.first_pass),
            rotation_angle=dec_frame.rotation_angle,
            tps_data=dec_frame.tps_data,

            frame=gray,
            is_binary=is_binary,

            orig_h=int(dec_frame.orig_h),
            orig_w=int(dec_frame.orig_w),
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

        Run when we have tiles_batch_n tiles (or if force_on_timeout and any tiles exist).
        If we have many tiles queued, drain multiple full batches in one call to reduce
        scheduling overhead (fewer context switches / Python overhead).
        """
        if not self._tile_pool:
            return

        if force_on_timeout and len(self._tile_pool) > 0:
            await self._process_one_tiles_batch(min(self.cfg.tiles_batch_n, len(self._tile_pool)))
            return

        while len(self._tile_pool) >= self.cfg.tiles_batch_n:
            await self._process_one_tiles_batch(self.cfg.tiles_batch_n)

    async def _process_one_tiles_batch(self, batch_size: int) -> None:
        """
        Pop up to batch_size tiles across any frames, run model, scatter + stitch.

        Strategy A on failure: emit PipelineError for each impacted frame and drop its partial state.
        Retry: if CUDA OOM, split into smaller batches down to 1 tile.
        """
        if batch_size <= 0 or not self._tile_pool:
            return

        items: List[_TileWorkItem] = []
        for _ in range(min(batch_size, len(self._tile_pool))):
            items.append(self._tile_pool.popleft())

        impacted_frame_ids = sorted({it.frame_id for it in items})

        try:
            with torch.profiler.record_function("tile_batch_prepare"):
                tiles_1 = torch.stack([it.tile_1ch for it in items], dim=0)
                tiles_3 = tiles_1.expand(-1, 3, -1, -1)

            with torch.profiler.record_function("tile_batch_infer"):
                soft = self._infer_tiles_to_soft(tiles_3).to(dtype=self._accum_dtype)

            # Advance profiler schedule once per inference call.
            if self._profiler_enabled and self._profiler is not None and self._profiler_started:
                try:
                    self._profiler.step()
                    self._profiler_step += 1
                except Exception:
                    pass

            for i, it in enumerate(items):
                pending = self._pending_frames.get(it.frame_id)
                if pending is None:
                    continue
                x0, y0 = it.x0, it.y0
                ps = pending.patch_size
                pending.accum_soft[:, y0:y0+ps, x0:x0+ps] = torch.maximum(
                    pending.accum_soft[:, y0:y0+ps, x0:x0+ps],
                    soft[i],
                )
                pending.received_tiles += 1

        except Exception as e:
            retryable = self._is_cuda_oom(e)
            if retryable:
                self._maybe_reinit_gpu("oom")
                for it in reversed(items):
                    self._tile_pool.appendleft(it)
                if batch_size > 1:
                    left = max(1, batch_size // 2)
                    right = max(1, batch_size - left)
                    await self._process_one_tiles_batch(left)
                    await self._process_one_tiles_batch(right)
                    return

            for fid in impacted_frame_ids:
                pending = self._pending_frames.get(fid)
                if pending is None:
                    continue
                await self._emit_pipeline_error(
                    internal_stage="infer.tiles_batch",
                    exc=e,
                    lane_second_pass=bool(pending.lane_second_pass),
                    task=pending.task,
                    source_etag=pending.source_etag,
                    retryable=bool(retryable),
                    attempt=1,
                )
                self._drop_frame(fid)
            return

        await self._finalize_completed_frames()

        # Ensure all async D2H emissions are complete before shutdown.
        while self._d2h_inflight > 0:
            await asyncio.sleep(0)


    def _infer_tiles_to_soft(self, tiles_3: torch.Tensor) -> torch.Tensor:
        """
        Runs the model on tiles and returns sigmoid probabilities for the selected class.

        tiles_3: [B, 3, 512, 512] float32 on device
        returns: [B, 1, 512, 512] float32 on device
        """
        with torch.inference_mode():
            # IMPORTANT: we keep the profiler context open for the whole run() loop and only
            # call prof.step() once per invocation (see _process_tile_batch).
            # Here we just add named ranges to make the trace readable.
            prof_sync = bool(getattr(self.cfg, "profiler_sync_cuda", False))
            if self._profiler_enabled and prof_sync and torch.cuda.is_available():
                torch.cuda.synchronize()

            with torch.profiler.record_function("model_forward"):
                logits = self.model(tiles_3)

            if self._profiler_enabled and prof_sync and torch.cuda.is_available():
                torch.cuda.synchronize()

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

            with torch.profiler.record_function("logits_to_probs"):
                soft = torch.sigmoid(sel).to(torch.float32)
            return soft


    async def _finalize_completed_frames(self) -> None:
        """
        Emit InferredFrame for any pending frames whose tiles are all received.

        This function is also an error boundary: any exception becomes a PipelineError for that frame.
        """
        done_ids: List[int] = []
        for frame_id, pending in list(self._pending_frames.items()):
            if pending.received_tiles >= pending.expected_tiles:
                done_ids.append(frame_id)

        for frame_id in done_ids:
            pending = self._pending_frames.pop(frame_id, None)
            if pending is None:
                continue

            try:
                mask_soft = pending.accum_soft
                if pending.pad_y > 0:
                    mask_soft = mask_soft[:, : pending.orig_h, :]
                if pending.pad_x > 0:
                    mask_soft = mask_soft[:, :, : pending.orig_w]


                with torch.profiler.record_function("threshold_to_packedbits"):
                    # Boolean mask on GPU
                    mask_bool = (mask_soft > self.cfg.class_threshold).squeeze(0)  # [H,W] bool

                    # Pack bits along width to reduce D2H bandwidth by ~8x.
                    # We pack little-endian to match np.unpackbits(..., bitorder="little") downstream.
                    h, w = mask_bool.shape
                    pad = (-int(w)) % 8
                    if pad:
                        mask_bool = F.pad(mask_bool, (0, pad), value=False)

                    mask_u8 = mask_bool.to(torch.uint8)  # 0/1
                    w8 = (int(w) + pad) // 8
                    mask_u8 = mask_u8.view(h, w8, 8)

                    weights = (1 << torch.arange(8, device=mask_u8.device, dtype=torch.uint8)).view(1, 1, 8)
                    packed = (mask_u8 * weights).sum(dim=2).contiguous()  # [H, W8] uint8

                # Async D2H on a dedicated stream to overlap with next batch compute.
                with torch.profiler.record_function("packedbits_d2h_async"):
                    if self._d2h_stream is None:
                        packed_host = packed.cpu()
                        done_evt = None
                    else:
                        while self._d2h_inflight >= self._d2h_inflight_limit:
                            await asyncio.sleep(0)

                        packed_host = torch.empty_like(packed, device="cpu", pin_memory=True)
                        done_evt = torch.cuda.Event()
                        self._d2h_inflight += 1
                        with torch.cuda.stream(self._d2h_stream):
                            packed_host.copy_(packed, non_blocking=True)
                            done_evt.record(self._d2h_stream)

                # Emit in background after the copy completes (does not block the main batching loop).
                asyncio.create_task(
                    self._emit_inferred_after_d2h(
                        pending=pending,
                        packed_host=packed_host,
                        done_evt=done_evt,
                        h=int(h),
                        w=int(w),
                        pad=int(pad),
                    )
                )
                # Free GPU memory after emitting
                del pending.accum_soft

            except Exception as e:
                retryable = self._is_cuda_oom(e)
                if retryable:
                    self._maybe_reinit_gpu("oom")
                await self._emit_pipeline_error(
                    internal_stage="finalize",
                    exc=e,
                    lane_second_pass=bool(pending.lane_second_pass),
                    task=pending.task,
                    source_etag=pending.source_etag,
                    retryable=bool(retryable),
                    attempt=1,
                )
                # Free GPU memory on error too
                if hasattr(pending, 'accum_soft') and pending.accum_soft is not None:
                    del pending.accum_soft

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
        while self._tile_pool:
            # Process all remaining tiles in chunks
            n = min(self.cfg.tiles_batch_n, len(self._tile_pool))
            if n == 0:
                break
            await self._process_one_tiles_batch(n)

        # Safety net: if any frames are somehow left (shouldn't happen), try to finalize.
        await self._finalize_completed_frames()

        # Ensure all async D2H emissions are complete before shutdown.
        while self._d2h_inflight > 0:
            await asyncio.sleep(0)