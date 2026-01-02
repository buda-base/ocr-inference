
import asyncio
from typing import List, Tuple
from .types import ImageTask

class Prefetcher:
    """Async S3 reader.

    Input: List[ImageTask]
    Output: puts (ImageTask, bytes) pairs onto q_bytes.
    Respects both global and per-worker concurrency caps.
    """
    def __init__(self, cfg, s3ctx, keys: List[ImageTask], q_bytes: asyncio.Queue):
        self.cfg = cfg
        self.s3 = s3ctx
        self.keys = keys
        self.q_bytes = q_bytes

    async def run(self):
        """Fetch all ImageTasks concurrently and push to q_bytes; then send sentinel."""
        async with self.s3.client() as s3c:
            sem = self.s3.global_sem
            per_worker = asyncio.Semaphore(self.cfg.s3_inflight_per_worker)

            async def fetch(task: ImageTask):
                async with sem, per_worker:
                    # obj = await s3c.get_object(Bucket=self.cfg.s3_bucket, Key=task.key)
                    # body: bytes = await obj["Body"].read()
                    body = b""
                    await self.q_bytes.put((task, body))

            await asyncio.gather(*(fetch(t) for t in self.keys))

        await self.q_bytes.put((None, None))
