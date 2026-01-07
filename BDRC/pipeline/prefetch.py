import asyncio
import traceback
from typing import Iterable, Optional

from pathlib import Path
from typing import List, Optional, Union
from urllib.parse import urlparse, unquote

from .types_common import *

class BasePrefetcher:
    """
    Shared worker pool + error wrapping.
    Subclasses implement _fetch_impl(task) -> (source_etag, bytes).
    """

    def __init__(self, cfg, volume_task: VolumeTask, q_prefetcher_to_decoder: asyncio.Queue[FetchedBytesMsg]):
        self.cfg = cfg
        self.volume_task = volume_task
        self.image_tasks: List[ImageTask] = list(volume_task.image_tasks)
        self.q_prefetcher_to_decoder = q_prefetcher_to_decoder
        self._per_worker_sem: asyncio.Semaphore

    def _is_retryable(self, exc: Exception) -> bool:
        return False  # keep simple; classify later if desired

    async def _fetch_impl(self, task: ImageTask) -> tuple[Optional[str], bytes]:
        raise NotImplementedError

    async def _fetch_one(self, task: ImageTask) -> FetchedBytesMsg:
        attempt = 1
        max_attempts = 3

        while True:
            try:
                async with self._per_worker_sem:
                    source_etag, body = await self._fetch_impl(task)
                return FetchedBytes(task=task, source_etag=source_etag, file_bytes=body)

            except Exception as e:
                tb = traceback.format_exc()
                retryable = self._is_retryable(e)

                err = PipelineError(
                    stage="Prefetcher",
                    task=task,
                    source_etag=None,
                    error_type=type(e).__name__,
                    message=str(e),
                    traceback=tb,
                    retryable=retryable,
                    attempt=attempt,
                )

                if attempt >= max_attempts or not retryable:
                    return err

                attempt += 1
                await asyncio.sleep(0.2 * attempt)

    async def run(self) -> None:
        self._per_worker_sem = asyncio.Semaphore(self.cfg.inflight_per_worker)

        work_q: asyncio.Queue[Optional[ImageTask]] = asyncio.Queue()
        for t in self.image_tasks:
            work_q.put_nowait(t)

        n_workers = min(len(self.image_tasks), max(1, self.cfg.inflight_per_worker))

        async def worker() -> None:
            while True:
                task = await work_q.get()
                if task is None:
                    work_q.task_done()
                    return
                msg = await self._fetch_one(task)
                await self.q_prefetcher_to_decoder.put(msg)
                work_q.task_done()

        workers = [asyncio.create_task(worker()) for _ in range(n_workers)]
        for _ in range(n_workers):
            work_q.put_nowait(None)

        await work_q.join()
        for w in workers:
            await w

        await self.q_prefetcher_to_decoder.put(
            EndOfStream(stream="prefetched", producer=type(self).__name__)
        )


# --- S3 implementation ---

class S3Prefetcher(BasePrefetcher):
    def __init__(self, cfg, s3ctx, volume_task: VolumeTask, q_prefetcher_to_decoder: asyncio.Queue[FetchedBytesMsg]):
        super().__init__(cfg, volume_task, q_prefetcher_to_decoder)
        self.s3 = s3ctx

    async def _fetch_impl(self, task: ImageTask) -> tuple[Optional[str], bytes]:
        bucket, key = _parse_s3_uri(task.source_uri)

        async with self.s3.global_sem:
            async with self.s3.client() as s3c:
                obj = await s3c.get_object(Bucket=bucket, Key=key)
                etag = normalize_etag(obj.get("ETag", ""))
                body: bytes = await obj["Body"].read()
                return etag, body


# --- Local implementation ---

class LocalPrefetcher(BasePrefetcher):
    def __init__(self, cfg, volume_task: VolumeTask, q_prefetcher_to_decoder: asyncio.Queue[FetchedBytesMsg]):
        super().__init__(cfg, volume_task, q_prefetcher_to_decoder)

    async def _fetch_impl(self, task: ImageTask) -> tuple[Optional[str], bytes]:
        p = _parse_file_uri(task.source_uri)

        # Offload blocking disk IO to the default thread pool
        data: bytes = await asyncio.to_thread(p.read_bytes)

        # “etag” equivalent: cheap fingerprint for debugging/caching consistency
        st = await asyncio.to_thread(p.stat)
        pseudo_etag = f"local:{st.st_size}:{st.st_mtime_ns}"
        return pseudo_etag, data


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    """
    Accepts s3://bucket/key and returns (bucket, key).
    """
    u = urlparse(uri)
    if u.scheme != "s3":
        raise ValueError(f"Not an s3:// URI: {uri}")
    bucket = u.netloc
    key = u.path.lstrip("/")
    if not bucket or not key:
        raise ValueError(f"Invalid s3:// URI (need bucket and key): {uri}")
    return bucket, key

def _parse_file_uri(uri: str) -> Path:
    """
    Accepts file:///abs/path (or file:/abs/path) and returns a Path.
    """
    u = urlparse(uri)
    if u.scheme != "file":
        raise ValueError(f"Not a file:// URI: {uri}")

    # urlparse on file URIs:
    # - path is percent-encoded; unquote it
    # - netloc is typically "" for local files; we ignore netloc here
    p = unquote(u.path)
    if not p:
        raise ValueError(f"Invalid file:// URI: {uri}")
    return Path(p)

def normalize_etag(etag: str) -> str:
    """Normalize S3 ETag by stripping a single pair of surrounding quotes."""
    if len(etag) >= 2 and etag[0] == '"' and etag[-1] == '"':
        return etag[1:-1]
    return etag