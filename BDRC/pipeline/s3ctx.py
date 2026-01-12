import asyncio
import contextlib
import logging
from aiobotocore.session import get_session

logger = logging.getLogger(__name__)


class S3Context:
    """Holds global S3 settings + concurrency semaphore.

    Uses a SINGLE shared S3 client for all requests (connection pooling).
    This is much more efficient than creating a new client per request.
    
    Credentials are resolved via the default credential chain:
    - Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    - Shared credentials file (~/.aws/credentials)
    - IAM role (for EC2 instances via instance metadata service)
    """

    def __init__(self, cfg, global_sem: asyncio.Semaphore):
        self.cfg = cfg
        self.global_sem = global_sem
        self._session = get_session()
        self._client = None
        self._client_ctx = None

    async def _ensure_client(self):
        """Lazily create the shared S3 client on first use."""
        if self._client is not None:
            return
        
        client_config = {
            "region_name": self.cfg.s3_region,
        }
        
        # Only use a profile if explicitly set to a non-default value
        aws_profile = getattr(self.cfg, "aws_profile", None)
        if aws_profile and aws_profile != "default":
            client_config["profile_name"] = aws_profile
        
        logger.info(f"[S3Context] Creating shared S3 client (region={self.cfg.s3_region})")
        self._client_ctx = self._session.create_client("s3", **client_config)
        self._client = await self._client_ctx.__aenter__()

    async def close(self):
        """Close the shared S3 client. Call this when done with all S3 operations."""
        if self._client_ctx is not None:
            await self._client_ctx.__aexit__(None, None, None)
            self._client = None
            self._client_ctx = None
            logger.info("[S3Context] Closed shared S3 client")

    @contextlib.asynccontextmanager
    async def client(self):
        """Yield the shared S3 client.
        
        This reuses a single client for all requests, which is much more
        efficient than creating a new client per request.
        
        The client uses connection pooling internally, so concurrent
        requests share connections efficiently.
        """
        await self._ensure_client()
        yield self._client