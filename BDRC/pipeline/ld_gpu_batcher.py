import asyncio
from typing import Optional, Tuple

from .types_common import DecodedFrame, PipelineError, EndOfStream, InferredFrame

class LDGpuBatcher:
    """
    Two-lane GPU micro-batcher and inference runner.

    Inputs:
      - q_init: DecodedFrameMsg (first pass)
      - q_re:   DecodedFrameMsg (second pass; higher priority)
    Outputs:
      - q_first and q_second: InferredFrameMsg
    """

    def __init__(
        self,
        cfg,
        q_decoder_to_gpu_pass_1: asyncio.Queue,
        q_post_processor_to_gpu_pass_2: asyncio.Queue,
        q_first: asyncio.Queue,
        q_second: asyncio.Queue,
    ):
        self.cfg = cfg
        self.q_init = q_decoder_to_gpu_pass_1
        self.q_re = q_post_processor_to_gpu_pass_2
        self.q_first = q_first
        self.q_second = q_second

        self._init_done = False
        self._re_done = False

    async def _pop_one(
        self, q: asyncio.Queue, timeout_s: float
    ):
        try:
            return await asyncio.wait_for(q.get(), timeout=timeout_s)
        except asyncio.TimeoutError:
            return None  # means "no item right now", NOT end-of-stream

    async def run(self):
        # You can tune these weights; goal: prioritize reprocess, but don't starve init
        reprocess_budget = getattr(self.cfg, "reprocess_budget", 3)
        init_budget = 1

        while True:
            # termination condition: both lanes ended (and any internal buffers flushed)
            if self._init_done and self._re_done:
                await self._flush()
                await self.q_first.put(EndOfStream(stream="gpu_pass_1", producer="LDGpuBatcher"))
                await self.q_second.put(EndOfStream(stream="gpu_pass_2", producer="LDGpuBatcher"))
                return

            took = False

            # --- prefer reprocess lane (from LDPostProcessor) ---
            if not self._re_done:
                for _ in range(reprocess_budget):
                    msg = await self._pop_one(self.q_re, self.cfg.batch_timeout_ms / 1000.0)
                    if msg is None:
                        break  # just empty right now
                    if isinstance(msg, EndOfStream) and msg.stream == "transformed_pass_1":
                        self._re_done = True
                        took = True
                        break
                    if isinstance(msg, PipelineError):
                        # forward errors (policy decision)
                        await self.q_second.put(msg)
                        took = True
                        continue

                    await self._infer_and_summarize(msg, second_pass=True)
                    took = True

            if took:
                continue

            # --- then init lane (from Decoder) ---
            if not self._init_done:
                for _ in range(init_budget):
                    msg = await self._pop_one(self.q_init, self.cfg.batch_timeout_ms / 1000.0)
                    if msg is None:
                        break
                    if isinstance(msg, EndOfStream) and msg.stream == "decoded":
                        self._init_done = True
                        took = True
                        break
                    if isinstance(msg, PipelineError):
                        await self.q_first.put(msg)
                        took = True
                        continue

                    await self._infer_and_summarize(msg, second_pass=False)
                    took = True

            # If neither lane had work, loop again (don’t “idle shutdown”)
            # You can add a tiny sleep to reduce spin if desired:
            if not took:
                await asyncio.sleep(0)

    async def _flush(self):
        # in case there's some tiles left in the batch, execute the batch
        return

    async def _infer_and_summarize(self, msg: DecodedFrame, second_pass=False) -> InferredFrame:
        # GPU stuff goes here
        # transfer on GPU
        # binarize if not binarized yet
        # tile in 3, 512, 512 patches
        # put patches in constant size tile batch (configurable, 8, 16, 32?)
        # wait for batch to fill
        # execute batch
        # untile results
        # get line mask
        # package line mask in InferredFrame
        return