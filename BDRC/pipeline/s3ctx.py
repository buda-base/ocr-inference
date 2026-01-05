import asyncio
import aioboto3
import contextlib
from typing import Optional

class S3Context:
    """Holds global S3 settings + concurrency semaphore.

    Use `client()` as an async context manager to obtain an S3 client
    (aioboto3 in real code). The global semaphore limits total in-flight
    GETs across **all** workers.
    """

    def __init__(self, cfg, global_sem: asyncio.Semaphore):
        self.cfg = cfg
        self.global_sem = global_sem
        self._session = aioboto3.Session()

    @contextlib.asynccontextmanager
    async def client(self):
        """Yield an async S3 client."""
        async with self._session.client("s3", region_name=self.cfg.s3_region) as c:
             yield c