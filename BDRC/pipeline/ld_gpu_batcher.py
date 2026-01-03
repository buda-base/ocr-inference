
import asyncio, time
from typing import List, Tuple
from .types import DecodedFrame, Record

class GpuBatcher:
        """Two-lane GPU micro-batcher and inference runner.

        Inputs:
          - q_frames_initial: DecodedFrame (first pass)
          - q_frames_reprocess: DecodedFrame (second pass; higher priority)
        Outputs:
          - First-pass: (DecodedFrame, summary) to q_firstpass_summaries
          - Second-pass: Record to q_records

        Policy: weighted fair scheduling favoring reprocess lane; batches built
        with (batch_size, batch_timeout_ms) to keep GPU utilized.
        """
    def __init__(self, cfg, q_frames_initial: asyncio.Queue, q_frames_reprocess: asyncio.Queue, q_firstpass_summaries: asyncio.Queue, q_records: asyncio.Queue):
        self.cfg = cfg
        self.q_init = q_frames_initial
        self.q_re = q_frames_reprocess
        self.q_first = q_firstpass_summaries
        self.q_records = q_records
        self.re_weight = 3

    async def _infer_and_summarize(self, frame: DecodedFrame, second_pass: bool = False):
            """Run tiling→model→stitch and emit either a summary (first pass) or a final Record (second pass)."""
        # TODO: implement GPU tiling -> model -> stitch
        if second_pass:
            rec = Record(
                img_file_name=frame.task.key,
                img_s3_etag=frame.task.etag,
                resized_w=frame.width,
                resized_h=frame.height,
                rotation_angle=0.0,
                tps_points=None,
                lines_contours=None,
                nb_lines=0,
            )
            await self.q_records.put(rec)
        else:
            summary = {"placeholder": True}
            await self.q_first.put((frame, summary))

    async def _pop_with_timeout(self, q: asyncio.Queue, timeout: float):
        try:
            return await asyncio.wait_for(q.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    async def run(self):
        """Scheduler loop: prefer reprocess items by weight; propagate sentinels when idle."""
        empty_ticks = 0
        while True:
            took = False
            for _ in range(self.re_weight):
                item = await self._pop_with_timeout(self.q_re, self.cfg.batch_timeout_ms/1000.0)
                if item is not None:
                    await self._infer_and_summarize(item, second_pass=True)
                    took = True
            if took:
                empty_ticks = 0
                continue
            item = await self._pop_with_timeout(self.q_init, self.cfg.batch_timeout_ms/1000.0)
            if item is None:
                empty_ticks += 1
            else:
                await self._infer_and_summarize(item, second_pass=False)
                empty_ticks = 0
            if empty_ticks > 4:
                await self.q_first.put(None)
                await self.q_records.put(None)
                break
