"""
TileBatcher: Prepares batches of tiled frames for GPU inference.

This component sits between Decoder/PostProcessor and LDInferenceRunner,
handling binarization, tiling, and batch assembly. Similar to PyTorch
DataLoader's worker + collate_fn role.
"""

import asyncio
import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .types_common import DecodedFrame, PipelineError, EndOfStream, TiledBatch
from .img_helpers import adaptive_binarize

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Tiling functions (from utils_alt.py)
# -----------------------------------------------------------------------------

def pad_to_multiple(img: torch.Tensor, patch_size: int = 512, value: float = 255.0) -> Tuple[torch.Tensor, int, int]:
    """
    Pad image to make dimensions divisible by patch_size.
    
    Args:
        img: [C, H, W] tensor
        patch_size: tile size
        value: padding value
    
    Returns:
        (padded_img, pad_w, pad_h)
    """
    _, H, W = img.shape
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    # F.pad order: (left, right, top, bottom)
    img = F.pad(img, (0, pad_w, 0, pad_h), value=value)
    return img, pad_w, pad_h


def tile_image(img: torch.Tensor, patch_size: int = 512) -> Tuple[torch.Tensor, int, int]:
    """
    Tile image using unfold (no overlap).
    
    Args:
        img: [C, H, W] tensor (H, W must be divisible by patch_size)
    
    Returns:
        (tiles, x_steps, y_steps) where tiles is [N, C, patch_size, patch_size]
    """
    C, H, W = img.shape
    y_steps = H // patch_size
    x_steps = W // patch_size
    
    tiles = img.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    # tiles shape: [C, y_steps, x_steps, patch_size, patch_size]
    tiles = tiles.permute(1, 2, 0, 3, 4).contiguous()
    # tiles shape: [y_steps, x_steps, C, patch_size, patch_size]
    tiles = tiles.view(-1, C, patch_size, patch_size)
    # tiles shape: [N, C, patch_size, patch_size] where N = y_steps * x_steps
    
    return tiles, x_steps, y_steps


# -----------------------------------------------------------------------------
# Precision helpers
# -----------------------------------------------------------------------------

def get_tile_dtype(precision: str) -> torch.dtype:
    """
    Get torch dtype from precision string.
    
    Args:
        precision: "fp32", "fp16", "bf16", or "auto"
    
    Returns:
        torch.dtype
    """
    if precision == "fp16":
        return torch.float16
    elif precision == "bf16":
        return torch.bfloat16
    elif precision == "auto":
        # Auto: use fp16 if CUDA available, else fp32
        if torch.cuda.is_available():
            return torch.float16
        return torch.float32
    else:  # fp32 or default
        return torch.float32


# -----------------------------------------------------------------------------
# Synchronous tiling function (can run in thread pool)
# -----------------------------------------------------------------------------

def tile_frame_sync(
    gray: np.ndarray,
    patch_size: int,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, int, int, int, int]:
    """
    Tile a grayscale frame.
    
    Args:
        gray: [H, W] uint8 numpy array
        patch_size: tile size
        dtype: torch dtype for output tiles (fp32, fp16, bf16)
    
    Returns:
        (tiles, x_steps, y_steps, pad_x, pad_y)
        - tiles: [N, 3, patch_size, patch_size] tensor in specified dtype (on CPU)
    """
    # Convert to torch [1, H, W] - start with float32 for precision in padding/normalize
    img = torch.from_numpy(gray).unsqueeze(0).float()
    
    # Pad to multiple of patch_size (white background = 255)
    img, pad_x, pad_y = pad_to_multiple(img, patch_size, value=255.0)
    
    # Tile the image
    tiles, x_steps, y_steps = tile_image(img, patch_size)
    
    # Normalize to [0, 1]
    tiles = tiles.div_(255.0)
    
    # Expand grayscale to 3 channels for model input
    # tiles shape: [N, 1, H, W] -> [N, 3, H, W]
    tiles = tiles.expand(-1, 3, -1, -1).contiguous()
    
    # Convert to target dtype (fp16/bf16 saves memory)
    if dtype != torch.float32:
        tiles = tiles.to(dtype)
    
    return tiles, x_steps, y_steps, pad_x, pad_y


# -----------------------------------------------------------------------------
# Main class
# -----------------------------------------------------------------------------

class TileBatcher:
    """
    Tiles frames and batches them for GPU inference.
    
    Inputs (load-balanced, pass-2 has priority):
      - q_from_decoder: DecodedFrame (pass-1)
      - q_from_postprocessor: DecodedFrame (pass-2)
    
    Output:
      - q_to_inference: TiledBatch (short queue for GPU)
    """

    def __init__(
        self,
        cfg,
        q_from_decoder: asyncio.Queue,
        q_from_postprocessor: asyncio.Queue,
        q_to_inference: asyncio.Queue,
    ):
        self.cfg = cfg
        self.q_from_decoder = q_from_decoder
        self.q_from_postprocessor = q_from_postprocessor
        self.q_to_inference = q_to_inference

        # Configuration
        self.batch_size: int = getattr(cfg, "batch_size", 8)
        self.batch_timeout_s: float = cfg.batch_timeout_ms / 1000.0
        self.patch_size: int = getattr(cfg, "patch_size", 512)
        self.tile_workers: int = getattr(cfg, "tile_workers", 4)
        
        # Tile precision (fp16/bf16 saves ~50% memory)
        precision = getattr(cfg, "precision", "fp32")
        self.tile_dtype: torch.dtype = get_tile_dtype(precision)

        # State tracking
        self._decoder_done = False
        self._postprocessor_done = False
        self._pass1_eos_sent = False  # Track if we've sent pass-1 EOS

        # Thread pool for parallel tiling (optional, can be disabled)
        self._use_parallel_tiling = getattr(cfg, "parallel_tiling", True)
        if self._use_parallel_tiling:
            self._tile_executor = ThreadPoolExecutor(max_workers=self.tile_workers)
            logger.info(
                f"[TileBatcher] Initialized with parallel tiling ({self.tile_workers} workers), "
                f"batch_size={self.batch_size}, patch_size={self.patch_size}, dtype={self.tile_dtype}"
            )
        else:
            self._tile_executor = None
            logger.info(
                f"[TileBatcher] Initialized with sequential tiling, "
                f"batch_size={self.batch_size}, patch_size={self.patch_size}, dtype={self.tile_dtype}"
            )

        # Buffer for accumulating tiled frames before emitting a batch
        # Each entry: dict with tiles, metadata, etc.
        self._buffer: List[Dict[str, Any]] = []
        
        # Pending tiling tasks (for parallel tiling)
        self._pending_tiles: List[asyncio.Task] = []
        
        # Output buffer removed - see explanation below

    # -------------------------------------------------------------------------
    # Error handling
    # -------------------------------------------------------------------------

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
    ) -> None:
        import logging
        logger = logging.getLogger(__name__)
        
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        err = PipelineError(
            stage="TileBatcher",
            task=task,
            source_etag=source_etag,
            error_type=type(exc).__name__,
            message=f"[{internal_stage}] {exc}",
            traceback=tb,
            retryable=bool(retryable),
            attempt=int(attempt),
        )
        
        # Errors flow through to inference runner, which will route them
        try:
            await asyncio.wait_for(self.q_to_inference.put(err), timeout=5.0)
        except asyncio.TimeoutError:
            logger.critical(
                f"Failed to emit error: queue full. "
                f"Dropping error for {task.img_filename if task else 'unknown'}"
            )

    # -------------------------------------------------------------------------
    # Frame preprocessing
    # -------------------------------------------------------------------------

    def _preprocess_frame(self, dec_frame: DecodedFrame) -> np.ndarray:
        """
        Preprocess a decoded frame: validate and optionally binarize.
        
        Returns grayscale uint8 numpy array [H, W].
        """
        gray = dec_frame.frame
        if not isinstance(gray, np.ndarray) or gray.ndim != 2 or gray.dtype != np.uint8:
            raise ValueError("DecodedFrame.frame must be a 2D numpy uint8 array (H, W)")

        # Optional binarization
        is_binary = bool(dec_frame.is_binary)
        if not is_binary:
            gray = adaptive_binarize(
                gray,
                block_size=self.cfg.binarize_block_size,
                c=self.cfg.binarize_c,
            )
        
        return gray

    async def _tile_frame(self, gray: np.ndarray) -> Tuple[torch.Tensor, int, int, int, int]:
        """
        Tile a frame, optionally using thread pool for parallelism.
        """
        if self._use_parallel_tiling and self._tile_executor is not None:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self._tile_executor,
                tile_frame_sync,
                gray,
                self.patch_size,
                self.tile_dtype,
            )
        else:
            return tile_frame_sync(gray, self.patch_size, self.tile_dtype)

    async def _tile_frame_async(
        self,
        dec_frame: DecodedFrame,
        gray: np.ndarray,
        is_pass2: bool,
    ) -> Dict[str, Any]:
        """
        Tile a frame and return the complete entry for the buffer.
        
        This is designed to be run as a task for parallel tiling.
        """
        t0 = time.perf_counter()
        
        if self._use_parallel_tiling and self._tile_executor is not None:
            loop = asyncio.get_event_loop()
            tiles, x_steps, y_steps, pad_x, pad_y = await loop.run_in_executor(
                self._tile_executor,
                tile_frame_sync,
                gray,
                self.patch_size,
                self.tile_dtype,
            )
        else:
            tiles, x_steps, y_steps, pad_x, pad_y = tile_frame_sync(
                gray, self.patch_size, self.tile_dtype
            )
        
        tile_time = time.perf_counter() - t0
        
        return {
            "dec_frame": dec_frame,
            "gray": gray,
            "second_pass": is_pass2,
            "tiles": tiles,
            "x_steps": x_steps,
            "y_steps": y_steps,
            "pad_x": pad_x,
            "pad_y": pad_y,
            "tile_time": tile_time,
        }

    def _collect_completed_tiles(self) -> Tuple[int, float]:
        """
        Check pending tile tasks and move completed ones to buffer.
        
        Returns:
            (n_completed, total_tile_time)
        """
        completed = 0
        total_time = 0.0
        still_pending = []
        
        for task in self._pending_tiles:
            if task.done():
                try:
                    entry = task.result()
                    self._buffer.append(entry)
                    completed += 1
                    total_time += entry.get("tile_time", 0.0)
                except Exception as e:
                    # Task failed - log error but continue
                    logger.error(f"[TileBatcher] Tile task failed: {e}")
            else:
                still_pending.append(task)
        
        self._pending_tiles = still_pending
        return completed, total_time

    # -------------------------------------------------------------------------
    # Batch preparation (multi_image_collate_fn logic)
    # -------------------------------------------------------------------------

    def _prepare_batch(self) -> TiledBatch:
        """
        Combine buffered tiled frames into a single TiledBatch.
        
        This is the collate_fn equivalent from the reference code.
        """
        all_tiles = []
        tile_ranges = []
        metas = []
        offset = 0

        for entry in self._buffer:
            tiles = entry["tiles"]
            n_tiles = tiles.shape[0]
            
            tile_ranges.append((offset, offset + n_tiles))
            all_tiles.append(tiles)
            
            metas.append({
                "dec_frame": entry["dec_frame"],
                "gray": entry["gray"],
                "second_pass": entry["second_pass"],
                "x_steps": entry["x_steps"],
                "y_steps": entry["y_steps"],
                "pad_x": entry["pad_x"],
                "pad_y": entry["pad_y"],
            })
            
            offset += n_tiles

        # Stack all tiles into single tensor
        all_tiles_tensor = torch.cat(all_tiles, dim=0)
        
        # Clear buffer
        self._buffer = []
        
        return TiledBatch(
            all_tiles=all_tiles_tensor,
            tile_ranges=tile_ranges,
            metas=metas,
        )

    # -------------------------------------------------------------------------
    # Queue helpers
    # -------------------------------------------------------------------------

    def _try_get_nowait(self, q: asyncio.Queue):
        """Try to get an item from queue without waiting. Returns None if empty."""
        try:
            return q.get_nowait()
        except asyncio.QueueEmpty:
            return None
    
    async def _pop_one(self, q: asyncio.Queue, timeout_s: float):
        """Pop one item from queue with timeout. Returns None on timeout."""
        try:
            return await asyncio.wait_for(q.get(), timeout=timeout_s)
        except asyncio.TimeoutError:
            return None

    async def _maybe_emit_pass1_eos(self) -> None:
        """
        Emit pass-1 EOS when decoder lane is fully drained.
        
        This breaks the EOS cycle between TileBatcher and PostProcessor:
        - TileBatcher sends tiled_pass_1 EOS
        - LDInferenceRunner forwards as gpu_pass_1 EOS
        - PostProcessor receives gpu_pass_1 EOS, can finish pass-1 processing
        - PostProcessor sends transformed_pass_1 EOS back to TileBatcher
        - TileBatcher can now finish
        """
        if self._pass1_eos_sent:
            return
        if not self._decoder_done:
            return
        
        # Check if any frames in buffer are from pass-1
        has_pass1_frames = any(not entry["second_pass"] for entry in self._buffer)
        if has_pass1_frames:
            return

        await self.q_to_inference.put(
            EndOfStream(stream="tiled_pass_1", producer="TileBatcher")
        )
        self._pass1_eos_sent = True

    # -------------------------------------------------------------------------
    # Main run loop
    # -------------------------------------------------------------------------

    async def run(self) -> None:
        """
        Main loop: receive frames, tile them in parallel, batch, and emit.
        
        Key optimization: Fire off multiple tiling tasks in parallel instead
        of awaiting each one sequentially. This keeps the thread pool busy
        while we fetch more frames.
        
        Load balancing: prefer pass-2 (postprocessor) over pass-1 (decoder).
        """
        # Timing stats
        loop_count = 0
        total_wait_time = 0.0
        total_tile_time = 0.0
        total_batch_emit_time = 0.0
        frames_submitted = 0
        batches_emitted = 0
        
        # Max frames to have in-flight (tiling) at once
        max_inflight = self.tile_workers * 2  # 2x workers for good overlap
        
        total_put_wait_time = 0.0
        
        try:
            while True:
                loop_start = time.perf_counter()
                loop_count += 1
                
                # --- Collect completed tiling tasks ---
                n_completed, tile_time = self._collect_completed_tiles()
                total_tile_time += tile_time
                
                # --- Check termination ---
                all_done = self._decoder_done and self._postprocessor_done
                no_pending = len(self._pending_tiles) == 0
                
                if all_done and no_pending:
                    logger.info(
                        f"[TileBatcher] DONE - loops={loop_count}, frames_submitted={frames_submitted}, "
                        f"batches={batches_emitted}, total_wait={total_wait_time:.2f}s, "
                        f"total_tile={total_tile_time:.2f}s, total_emit={total_batch_emit_time:.2f}s, "
                        f"put_wait={total_put_wait_time:.2f}s"
                    )
                    # Flush remaining buffer
                    if self._buffer:
                        batch = self._prepare_batch()
                        batches_emitted += 1
                        await self.q_to_inference.put(batch)
                        logger.info(
                            f"[TileBatcher] Emitted final batch #{batches_emitted}: "
                            f"{len(batch.metas)} frames, {batch.all_tiles.shape[0]} tiles"
                        )
                    
                    # Send pass-1 EOS if not already sent
                    if not self._pass1_eos_sent:
                        await self.q_to_inference.put(
                            EndOfStream(stream="tiled_pass_1", producer="TileBatcher")
                        )
                        self._pass1_eos_sent = True
                    
                    # Send final pass-2 EOS
                    await self.q_to_inference.put(
                        EndOfStream(stream="tiled_pass_2", producer="TileBatcher")
                    )
                    break

                # --- Emit batch when full ---
                if len(self._buffer) >= self.batch_size:
                    emit_start = time.perf_counter()
                    n_frames = len(self._buffer)
                    batch = self._prepare_batch()
                    n_tiles = batch.all_tiles.shape[0]
                    
                    put_start = time.perf_counter()
                    await self.q_to_inference.put(batch)
                    put_wait = time.perf_counter() - put_start
                    total_put_wait_time += put_wait
                    
                    emit_time = time.perf_counter() - emit_start
                    total_batch_emit_time += emit_time
                    batches_emitted += 1
                    
                    logger.info(
                        f"[TileBatcher] Batch #{batches_emitted}: {n_frames} frames, {n_tiles} tiles, "
                        f"prepare={emit_time-put_wait:.3f}s, put_wait={put_wait:.3f}s, "
                        f"pending={len(self._pending_tiles)}"
                    )

                # --- Fetch more frames if we have capacity ---
                # Keep fetching while we have room for more in-flight tasks
                frames_fetched_this_loop = 0
                max_fetch_per_loop = max_inflight  # Don't spin forever
                
                while (len(self._pending_tiles) < max_inflight and 
                       frames_fetched_this_loop < max_fetch_per_loop):
                    
                    msg = None
                    is_pass2 = False
                    
                    # --- First try NON-BLOCKING gets from both queues ---
                    # This avoids wasting 25ms on empty queues
                    
                    # Prefer pass-2 (from PostProcessor)
                    if not self._postprocessor_done:
                        msg = self._try_get_nowait(self.q_from_postprocessor)
                        if msg is not None:
                            is_pass2 = True
                            if isinstance(msg, EndOfStream) and msg.stream == "transformed_pass_1":
                                self._postprocessor_done = True
                                logger.debug("[TileBatcher] Received transformed_pass_1 EOS")
                                msg = None
                            elif isinstance(msg, PipelineError):
                                await self.q_to_inference.put(msg)
                                msg = None

                    # Then pass-1 (from Decoder)
                    if msg is None and not self._decoder_done:
                        msg = self._try_get_nowait(self.q_from_decoder)
                        if msg is not None:
                            is_pass2 = False
                            if isinstance(msg, EndOfStream) and msg.stream == "decoded":
                                self._decoder_done = True
                                logger.debug("[TileBatcher] Received decoded EOS")
                                msg = None
                            elif isinstance(msg, PipelineError):
                                await self.q_to_inference.put(msg)
                                msg = None
                    
                    # --- If both queues empty, wait briefly on decoder queue ---
                    if msg is None and not self._decoder_done:
                        wait_start = time.perf_counter()
                        msg = await self._pop_one(self.q_from_decoder, self.batch_timeout_s)
                        wait_time = time.perf_counter() - wait_start
                        total_wait_time += wait_time
                        
                        if msg is not None:
                            is_pass2 = False
                            if isinstance(msg, EndOfStream) and msg.stream == "decoded":
                                self._decoder_done = True
                                logger.debug("[TileBatcher] Received decoded EOS")
                                msg = None
                            elif isinstance(msg, PipelineError):
                                await self.q_to_inference.put(msg)
                                msg = None
                    
                    if msg is None:
                        # No frame available, break out of fetch loop
                        break
                    
                    frames_fetched_this_loop += 1
                    
                    # --- Start tiling task (don't await!) ---
                    if isinstance(msg, DecodedFrame):
                        try:
                            # Preprocess synchronously (fast)
                            gray = self._preprocess_frame(msg)
                            
                            # Fire off async tiling task
                            task = asyncio.create_task(
                                self._tile_frame_async(msg, gray, is_pass2)
                            )
                            self._pending_tiles.append(task)
                            frames_submitted += 1
                            
                        except Exception as e:
                            await self._emit_pipeline_error(
                                internal_stage="preprocess",
                                exc=e,
                                lane_second_pass=is_pass2,
                                task=msg.task,
                                source_etag=msg.source_etag,
                                retryable=False,
                                attempt=1,
                            )

                # --- Timeout flush: emit partial batch if nothing is coming ---
                # Only flush if we have frames AND we couldn't fetch any new ones
                # AND we have no pending tiles (everything that was in-flight is now in buffer)
                if (frames_fetched_this_loop == 0 and 
                    self._buffer and 
                    len(self._pending_tiles) == 0):
                    
                    emit_start = time.perf_counter()
                    n_frames = len(self._buffer)
                    batch = self._prepare_batch()
                    n_tiles = batch.all_tiles.shape[0]
                    
                    put_start = time.perf_counter()
                    await self.q_to_inference.put(batch)
                    put_wait = time.perf_counter() - put_start
                    total_put_wait_time += put_wait
                    
                    emit_time = time.perf_counter() - emit_start
                    total_batch_emit_time += emit_time
                    batches_emitted += 1
                    
                    logger.info(
                        f"[TileBatcher] Batch #{batches_emitted} (flush): {n_frames} frames, {n_tiles} tiles, "
                        f"prepare={emit_time-put_wait:.3f}s, put_wait={put_wait:.3f}s"
                    )

                # --- Check if we should send pass-1 EOS ---
                await self._maybe_emit_pass1_eos()

                # Log progress periodically
                loop_time = time.perf_counter() - loop_start
                if loop_count % 100 == 0:
                    logger.debug(
                        f"[TileBatcher] Loop #{loop_count}: {loop_time*1000:.1f}ms, "
                        f"buffer={len(self._buffer)}, pending={len(self._pending_tiles)}, "
                        f"submitted={frames_submitted}"
                    )

                # Yield to event loop (allow pending tasks to progress)
                await asyncio.sleep(0)

        except asyncio.CancelledError:
            logger.info("TileBatcher cancelled, cleaning up...")
            # Cancel pending tasks
            for task in self._pending_tiles:
                task.cancel()
            raise
        except KeyboardInterrupt:
            logger.info("TileBatcher interrupted by user")
            raise
        finally:
            # Shutdown thread pool
            if self._tile_executor is not None:
                self._tile_executor.shutdown(wait=False)

