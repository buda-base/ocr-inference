import asyncio
import traceback
from typing import Iterable, Optional

from .types import ImageTask, FetchedBytes, PipelineError, FetchedBytesMsg, EndOfStream, normalize_etag


class Prefetcher:
    """Async S3 reader.

    Input: iterable of ImageTask
    Output: puts FetchedBytesMsg onto q_bytes
    Respects both global and per-worker concurrency caps.
    """

    def __init__(self, cfg, s3ctx, keys: Iterable[ImageTask], q_bytes: asyncio.Queue[BytesMsg]):
        self.cfg = cfg
        self.s3 = s3ctx
        self.keys = list(keys)
        self.q_bytes = q_bytes

    async def _fetch_one(self, s3c, task: ImageTask) -> BytesMsg:
        attempt = 1
        max_attempts = 3

        while True:
            try:
                async with self.s3.global_sem, self._per_worker_sem:
                    obj = await s3c.get_object(Bucket=self.cfg.s3_bucket, Key=task.s3_key)
                    etag = normalize_etag(obj.get("ETag", ""))
                    body: bytes = await obj["Body"].read()
                return FetchedBytes(task=task, s3_etag=etag, body=body)

            except Exception as e:
                tb = traceback.format_exc()
                retryable = False  # keep simple; add classification later if desired

                err = PipelineError(
                    stage="Prefetcher",
                    task=task,
                    s3_etag=None,
                    error_type=type(e).__name__,
                    message=str(e),
                    traceback=tb,
                    retryable=retryable,
                    attempt=attempt,
                )

                if attempt >= max_attempts or not retryable:
                    return err

                attempt += 1
                # basic backoff (simple). If you want: add jitter.
                await asyncio.sleep(0.2 * attempt)

    async def run(self) -> None:
        # Per-prefetcher cap (this instance).
        self._per_worker_sem = asyncio.Semaphore(self.cfg.s3_inflight_per_worker)

        work_q: asyncio.Queue[Optional[ImageTask]] = asyncio.Queue()
        for t in self.keys:
            work_q.put_nowait(t)

        # Number of async workers pulling tasks; derived for simplicity.
        n_workers = min(len(self.keys), max(1, self.cfg.s3_inflight_per_worker))

        async with self.s3.client() as s3c:
            async def worker() -> None:
                while True:
                    task = await work_q.get()
                    if task is None:
                        work_q.task_done()
                        return
                    msg = await self._fetch_one(s3c, task)
                    await self.q_bytes.put(msg)
                    work_q.task_done()

            workers = [asyncio.create_task(worker()) for _ in range(n_workers)]
            for _ in range(n_workers):
                work_q.put_nowait(None)

            await work_q.join()
            for w in workers:
                await w

        # End-of-stream marker for the (single) decoder consumer coroutine
        await self.q_bytes.put(EndOfStream(stream="prefetched", producer="Prefetcher"))


def normalize_etag(etag: str) -> str:
    """Normalize S3 ETag by stripping a single pair of surrounding quotes."""
    if len(etag) >= 2 and etag[0] == '"' and etag[-1] == '"':
        return etag[1:-1]
    return etag