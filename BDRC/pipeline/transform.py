
import asyncio
from .types import DecodedFrame, Record

class TransformController:
        """Consumes first-pass summaries, decides rotation/TPS, and routes.

        Input: (DecodedFrame, summary) from GPU first pass.
        Output: Either a final Record to the writer, or a transformed
        DecodedFrame enqueued to the **reprocess** lane.
        """
    def __init__(self, cfg, q_first_results: asyncio.Queue, q_reprocess_frames: asyncio.Queue, q_records: asyncio.Queue):
        self.cfg = cfg
        self.q_first_results = q_first_results
        self.q_reprocess_frames = q_reprocess_frames
        self.q_records = q_records

    async def run(self):
        """Main loop: read summaries, decide, write or re-enqueue; propagate sentinel."""
        while True:
            item = await self.q_first_results.get()
            if item is None:
                await self.q_reprocess_frames.put(None)
                break
            frame, summary = item
            need = False  # TODO: real criterion based on summary
            if not need:
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
                # TODO: apply transform here and enqueue transformed frame
                await self.q_reprocess_frames.put(frame)
