import asyncio, os
from .config import PipelineConfig
from .types_common import *
from .prefetch import BasePrefetcher, LocalPrefetcher, S3Prefetcher
from .decoder import Decoder
from .ld_postprocessor import LDPostProcessor
from .ld_gpu_batcher import LDGpuBatcher
from .parquet_writer import ParquetWriter
from .s3ctx import S3Context

class LDVolumeWorker:
    """Owns a single volume and runs all stages concurrently.

    Wires the queues, starts:
      - Prefetcher → Decoder → GpuBatcher → LDPostProcessor → S3ParquetWriter
    All queues are bounded to enforce backpressure.
    """
    def __init__(self, cfg: PipelineConfig, volume_task: VolumeTask, progress: Optional[ProgressHook] = None, s3ctx: Optional[S3Context]=None):
        self.cfg: PipelineConfig = cfg
        self.volume_task: VolumeTask = volume_task
        self.s3ctx: Optional[S3Context] = s3ctx

        self.q_prefetcher_to_decoder: asyncio.Queue[FetchedBytesMsg] = asyncio.Queue(maxsize=cfg.max_q_prefetcher_to_decoder)
        self.q_decoder_to_gpu_pass_1: asyncio.Queue[DecodedFrameMsg] = asyncio.Queue(maxsize=cfg.max_q_decoder_to_gpu_pass_1)
        self.q_gpu_pass_1_to_post_processor: asyncio.Queue[InferredFrameMsg] = asyncio.Queue(maxsize=cfg.max_q_gpu_pass_1_to_post_processor)
        self.q_post_processor_to_gpu_pass_2: asyncio.Queue[DecodedFrameMsg] = asyncio.Queue(maxsize=cfg.max_q_post_processor_to_gpu_pass_2)
        self.q_gpu_pass_2_to_post_processor: asyncio.Queue[InferredFrameMsg] = asyncio.Queue(maxsize=cfg.max_q_gpu_pass_2_to_post_processor)
        self.q_post_processor_to_writer: asyncio.Queue[RecordMsg] = asyncio.Queue(maxsize=cfg.max_q_post_processor_to_writer)

        if volume_task.io_mode == "local":
            self.prefetcher: BasePrefetcher = LocalPrefetcher(cfg, volume_task, self.q_prefetcher_to_decoder)
        else:
            self.prefetcher: BasePrefetcher = S3Prefetcher(cfg, self.s3ctx, volume_task, self.q_prefetcher_to_decoder)
        self.decoder = Decoder(cfg, self.q_prefetcher_to_decoder, self.q_decoder_to_gpu_pass_1)
        self.batcher = LDGpuBatcher(cfg, self.q_decoder_to_gpu_pass_1, self.q_post_processor_to_gpu_pass_2, self.q_gpu_pass_1_to_post_processor, self.q_gpu_pass_2_to_post_processor)
        self.postprocessor = LDPostProcessor(cfg, self.q_gpu_pass_1_to_post_processor, self.q_gpu_pass_2_to_post_processor, self.q_post_processor_to_gpu_pass_2, self.q_post_processor_to_writer)
        self.writer = ParquetWriter(cfg, self.q_post_processor_to_writer, volume_task.output_parquet_uri, volume_task.output_jsonl_uri, progress=progress)

    async def run(self):
        """Run all pipeline stages concurrently with proper exception handling."""
        import logging
        logger = logging.getLogger(__name__)
        
        tasks = [
            asyncio.create_task(self.prefetcher.run(), name="prefetcher"),
            asyncio.create_task(self.decoder.run(), name="decoder"),
            asyncio.create_task(self.batcher.run(), name="batcher"),
            asyncio.create_task(self.postprocessor.run(), name="postprocessor"),
            asyncio.create_task(self.writer.run(), name="writer"),
        ]
        
        # Wait for all tasks, but handle exceptions per-task to avoid cascading failures
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Log any exceptions
        stage_names = ["prefetcher", "decoder", "batcher", "postprocessor", "writer"]
        for name, result in zip(stage_names, results):
            if isinstance(result, Exception):
                logger.error(f"Stage {name} failed: {result}", exc_info=result)
        
        # Re-raise writer failure as it's critical (data loss risk)
        if isinstance(results[4], Exception):
            raise RuntimeError(f"Writer stage failed: {results[4]}") from results[4]
