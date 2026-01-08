import asyncio
import contextlib
from aiobotocore.session import get_session

class S3Context:
    """Holds global S3 settings + concurrency semaphore.

    Use `client()` as an async context manager to obtain an S3 client
    (aiobotocore). The global semaphore limits total in-flight
    GETs across **all** workers.
    """

    def __init__(self, cfg, global_sem: asyncio.Semaphore):
        self.cfg = cfg
        self.global_sem = global_sem
        self._session = get_session()

    @contextlib.asynccontextmanager
    async def client(self):
        """Yield an async S3 client."""
        async with self._session.create_client("s3", region_name=self.cfg.s3_region) as c:
            yield c