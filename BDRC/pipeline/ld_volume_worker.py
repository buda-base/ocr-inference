
import asyncio, os
from .config import PipelineConfig
from .types_common import ImageTask
from .s3ctx import S3Context
from .prefetch import Prefetcher
from .decoder import Decoder
from .transform import TransformController
from .batcher import GpuBatcher
from .writer import S3ParquetWriter

class LDVolumeWorker:
    """Owns a single volume and runs all stages concurrently.

    Wires the queues, starts:
      - Prefetcher → Decoder → GpuBatcher → TransformController → S3ParquetWriter
    All queues are bounded to enforce backpressure.
    """
    def __init__(self, cfg: PipelineConfig, s3: S3Context, volume_id: str, tasks):
        self.cfg = cfg
        self.s3 = s3
        self.volume_id = volume_id
        self.tasks = tasks

        self.q_bytes = asyncio.Queue(maxsize=cfg.max_q_bytes)
        self.q_frames = asyncio.Queue(maxsize=cfg.max_q_frames)
        self.q_firstpass = asyncio.Queue(maxsize=cfg.max_q_firstpass)
        self.q_reprocess = asyncio.Queue(maxsize=cfg.max_q_reprocess)
        self.q_records = asyncio.Queue(maxsize=cfg.max_q_records)

        self.prefetch = Prefetcher(cfg, s3, tasks, self.q_bytes)
        self.decode = Decoder(cfg, self.q_bytes, self.q_frames)
        self.controller = TransformController(cfg, self.q_firstpass, self.q_reprocess, self.q_records)
        self.batch = GpuBatcher(cfg, self.q_frames, self.q_reprocess, self.q_firstpass, self.q_records)
        self.writer = S3ParquetWriter(cfg, self.q_records, volume_id)

    async def run(self):
        await asyncio.gather(
            self.prefetch.run(),
            self.decode.run(),
            self.batch.run(),
            self.controller.run(),
            self.writer.run(),
        )
