
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
      - Prefetcher → Decoder → GpuBatcher → LDPostProcessor → S3ParquetWriter
    All queues are bounded to enforce backpressure.
    """
    def __init__(self, cfg: PipelineConfig, s3: S3Context, volume_id: str, tasks):
        self.cfg = cfg
        self.s3 = s3
        self.volume_id = volume_id
        self.tasks = tasks

        self.q_prefetcher_to_decoder: asyncio.Queue[FetchedBytesMsg] = asyncio.Queue(maxsize=cfg.max_q_prefetcher_to_decoder)
        self.q_decoder_to_gpu_pass_1: asyncio.Queue[DecodedFrameMsg] = asyncio.Queue(maxsize=cfg.max_q_decoder_to_gpu_pass_1)
        self.q_gpu_pass_1_to_post_processor: asyncio.Queue[InferredFrameMsg] = asyncio.Queue(maxsize=cfg.max_q_gpu_pass_1_to_post_processor)
        self.q_post_processor_to_gpu_pass_2: asyncio.Queue[DecodedFrameMsg] = asyncio.Queue(maxsize=cfg.max_q_post_processor_to_gpu_pass_2)
        self.q_gpu_pass_2_to_post_processor: asyncio.Queue[InferredFrameMsg] = asyncio.Queue(maxsize=cfg.max_q_gpu_pass_2_to_post_processor)
        self.q_post_processor_to_writer: asyncio.Queue[RecordMsg] = asyncio.Queue(maxsize=cfg.max_q_post_processor_to_writer)

        self.prefetch = Prefetcher(cfg, s3, tasks, self.q_prefetcher_to_decoder)
        self.decode = Decoder(cfg, self.q_prefetcher_to_decoder, self.q_decoder_to_gpu_pass_1)
        self.batch = GpuBatcher(cfg, self.q_decoder_to_gpu_pass_1, self.max_q_post_processor_to_gpu_pass_2, self.q_gpu_pass_1_to_post_processor, self.q_gpu_pass_2_to_post_processor)
        self.postprocess = LDPostProcessor(cfg, self.q_gpu_pass_1_to_post_processor, self.q_gpu_pass_2_to_post_processor, self.q_post_processor_to_gpu_pass_2, self.q_post_processor_to_writer)
        self.writer = S3ParquetWriter(cfg, self.q_post_processor_to_writer, volume_id)

    async def run(self):
        await asyncio.gather(
            self.prefetch.run(),
            self.decode.run(),
            self.batch.run(),
            self.postprocess.run(),
            self.writer.run(),
        )