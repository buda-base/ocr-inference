"""
LDInferenceRunner: Runs GPU inference on pre-tiled batches.

This is a simplified GPU component that receives TiledBatch from TileBatcher,
runs the model, stitches results, and emits InferredFrames.

The heavy lifting (tiling, batching) is done by TileBatcher, so this
component is straightforward: receive batch → GPU forward → stitch → emit.
"""

import asyncio
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .types_common import (
    DecodedFrame,
    PipelineError,
    EndOfStream,
    InferredFrame,
    TiledBatch,
)


# -----------------------------------------------------------------------------
# Stitching functions (from utils_alt.py)
# -----------------------------------------------------------------------------

def stitch_tiles(preds: torch.Tensor, x_steps: int, y_steps: int, patch_size: int = 512) -> torch.Tensor:
    """
    Reconstruct full image from tiles.
    
    Args:
        preds: [N, C, H, W] tensor of tile predictions
        x_steps: number of horizontal tiles
        y_steps: number of vertical tiles
        patch_size: tile size
    
    Returns:
        [C, H_full, W_full] tensor
    """
    N, C, H, W = preds.shape
    assert H == patch_size and W == patch_size
    assert N == x_steps * y_steps
    
    # [N, C, H, W] → [y_steps, x_steps, C, H, W]
    tiles = preds.view(y_steps, x_steps, C, H, W)
    
    # Stitch width (concat along W dimension)
    rows = []
    for y in range(y_steps):
        rows.append(torch.cat(list(tiles[y]), dim=-1))  # [C, H, W*x_steps]
    
    # Stitch height (concat along H dimension)
    full = torch.cat(rows, dim=-2)  # [C, H*y_steps, W*x_steps]
    
    return full


def crop_padding(mask: torch.Tensor, pad_x: int, pad_y: int) -> torch.Tensor:
    """
    Remove padding from mask.
    
    Args:
        mask: [C, H, W] tensor
        pad_x: horizontal padding to remove
        pad_y: vertical padding to remove
    
    Returns:
        Cropped [C, H-pad_y, W-pad_x] tensor
    """
    if pad_y > 0:
        mask = mask[:, :-pad_y, :]
    if pad_x > 0:
        mask = mask[:, :, :-pad_x]
    return mask


# -----------------------------------------------------------------------------
# Inference function (from utils_alt.py infer_batch)
# -----------------------------------------------------------------------------

def infer_batch(
    model: torch.nn.Module,
    batch: TiledBatch,
    class_threshold: float,
    device: str,
    patch_size: int = 512,
) -> List[Tuple[Dict[str, Any], np.ndarray]]:
    """
    Run inference on a TiledBatch and stitch results.
    
    This is the core inference function, nearly identical to utils_alt.py infer_batch.
    
    Args:
        model: segmentation model
        batch: TiledBatch with pre-tiled frames
        class_threshold: threshold for binary mask
        device: "cuda" or "cpu"
        patch_size: tile size
    
    Returns:
        List of (meta, mask_np) tuples where mask_np is uint8 [H, W] with values {0, 255}
    """
    all_tiles = batch.all_tiles.to(device, non_blocking=True)
    
    with torch.inference_mode():
        preds = model(all_tiles)
        soft = torch.sigmoid(preds)

    results = []
    for (start, end), meta in zip(batch.tile_ranges, batch.metas):
        preds_img = soft[start:end]
        
        stitched = stitch_tiles(preds_img, meta["x_steps"], meta["y_steps"], patch_size)
        stitched = crop_padding(stitched, meta["pad_x"], meta["pad_y"])
        
        # Output as grayscale uint8 {0, 255}
        binary = (stitched > class_threshold).to(torch.uint8) * 255
        mask_np = binary.squeeze(0).cpu().numpy()
        
        results.append((meta, mask_np))
    
    return results


# -----------------------------------------------------------------------------
# Main class
# -----------------------------------------------------------------------------

class LDInferenceRunner:
    """
    Runs GPU inference on pre-tiled batches.
    
    Input:
      - q_from_tilebatcher: TiledBatch (short queue, e.g. size 8)
    
    Outputs:
      - q_gpu_pass_1_to_post_processor: InferredFrame (pass-1 results)
      - q_gpu_pass_2_to_post_processor: InferredFrame (pass-2 results)
    """

    def __init__(
        self,
        cfg,
        q_from_tilebatcher: asyncio.Queue,
        q_gpu_pass_1_to_post_processor: asyncio.Queue,
        q_gpu_pass_2_to_post_processor: asyncio.Queue,
    ):
        self.cfg = cfg
        self.q_in = q_from_tilebatcher
        self.q_gpu_pass_1_to_post_processor = q_gpu_pass_1_to_post_processor
        self.q_gpu_pass_2_to_post_processor = q_gpu_pass_2_to_post_processor

        # Configuration
        self.patch_size: int = getattr(cfg, "patch_size", 512)
        self.class_threshold: float = getattr(cfg, "class_threshold", 0.5)

        # Device/model configuration
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = getattr(cfg, "model", None)
        if self.model is None:
            raise ValueError("cfg.model must be set (torch.nn.Module)")
        if not isinstance(self.model, torch.nn.Module):
            raise TypeError(f"cfg.model must be torch.nn.Module, got {type(self.model)}")
        self.model.to(self.device)
        self.model.eval()

        # State tracking
        self._done = False
        self._p1_eos_sent = False
        self._p2_eos_sent = False

    # -------------------------------------------------------------------------
    # Error handling
    # -------------------------------------------------------------------------

    @staticmethod
    def _is_cuda_oom(exc: BaseException) -> bool:
        msg = str(exc).lower()
        return "out of memory" in msg or "cuda out of memory" in msg

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
            stage="LDInferenceRunner",
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
            logger.critical(
                f"Failed to emit error after {timeout_s}s: queue full. "
                f"Dropping error for {task.img_filename if task else 'unknown'}"
            )

    # -------------------------------------------------------------------------
    # Result emission
    # -------------------------------------------------------------------------

    async def _emit_result(self, meta: Dict[str, Any], mask_np: np.ndarray) -> None:
        """Emit InferredFrame to appropriate output queue."""
        dec_frame: DecodedFrame = meta["dec_frame"]
        gray: np.ndarray = meta["gray"]
        second_pass: bool = meta["second_pass"]

        out = InferredFrame(
            task=dec_frame.task,
            source_etag=dec_frame.source_etag,
            frame=gray,
            orig_h=dec_frame.orig_h,
            orig_w=dec_frame.orig_w,
            is_binary=True,  # After binarization in TileBatcher
            first_pass=dec_frame.first_pass,
            rotation_angle=dec_frame.rotation_angle,
            tps_data=dec_frame.tps_data,
            line_mask=mask_np,  # grayscale uint8, {0, 255}
        )

        if second_pass:
            await self.q_gpu_pass_2_to_post_processor.put(out)
        else:
            await self.q_gpu_pass_1_to_post_processor.put(out)

    # -------------------------------------------------------------------------
    # Batch processing
    # -------------------------------------------------------------------------

    async def _process_batch(self, batch: TiledBatch) -> None:
        """
        Process a TiledBatch: run GPU inference, stitch, emit results.
        """
        try:
            results = infer_batch(
                self.model,
                batch,
                self.class_threshold,
                self.device,
                self.patch_size,
            )

            for meta, mask_np in results:
                await self._emit_result(meta, mask_np)

        except Exception as e:
            # On error, emit PipelineError for each frame in the batch
            retryable = self._is_cuda_oom(e)
            for meta in batch.metas:
                await self._emit_pipeline_error(
                    internal_stage="inference",
                    exc=e,
                    lane_second_pass=meta["second_pass"],
                    task=meta["dec_frame"].task,
                    source_etag=meta["dec_frame"].source_etag,
                    retryable=retryable,
                    attempt=1,
                )

    # -------------------------------------------------------------------------
    # Main run loop
    # -------------------------------------------------------------------------

    async def run(self) -> None:
        """
        Main loop: receive TiledBatch, run inference, emit results.
        
        This is intentionally simple - all the batching complexity
        is handled by TileBatcher.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            while True:
                # Receive from TileBatcher
                msg = await self.q_in.get()

                # Handle different message types
                if isinstance(msg, EndOfStream):
                    if msg.stream == "tiled_pass_1":
                        # Pass-1 is done - forward EOS to PostProcessor
                        # This allows PostProcessor to finish pass-1 and send transformed_pass_1 EOS
                        if not self._p1_eos_sent:
                            await self.q_gpu_pass_1_to_post_processor.put(
                                EndOfStream(stream="gpu_pass_1", producer="LDInferenceRunner")
                            )
                            self._p1_eos_sent = True
                    
                    elif msg.stream == "tiled_pass_2":
                        # Pass-2 is done - send final EOS and exit
                        self._done = True
                        # Send pass-1 EOS if not already sent (edge case)
                        if not self._p1_eos_sent:
                            await self.q_gpu_pass_1_to_post_processor.put(
                                EndOfStream(stream="gpu_pass_1", producer="LDInferenceRunner")
                            )
                            self._p1_eos_sent = True
                        # Send pass-2 EOS
                        await self.q_gpu_pass_2_to_post_processor.put(
                            EndOfStream(stream="gpu_pass_2", producer="LDInferenceRunner")
                        )
                        break
                
                elif isinstance(msg, PipelineError):
                    # Forward errors from TileBatcher to appropriate queue
                    # We don't know which lane, so send to pass-1 by default
                    # (TileBatcher should have set the lane info in the error)
                    await self.q_gpu_pass_1_to_post_processor.put(msg)
                
                elif isinstance(msg, TiledBatch):
                    # Process the batch
                    await self._process_batch(msg)
                
                else:
                    logger.warning(f"LDInferenceRunner received unexpected message type: {type(msg)}")

                # Yield to event loop
                await asyncio.sleep(0)

        except asyncio.CancelledError:
            logger.info("LDInferenceRunner cancelled, cleaning up...")
            raise
        except KeyboardInterrupt:
            logger.info("LDInferenceRunner interrupted by user")
            raise
        finally:
            # Clear GPU cache on exit
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

