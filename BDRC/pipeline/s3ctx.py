import asyncio
import contextlib
from aiobotocore.session import get_session

class S3Context:
    """Holds global S3 settings + concurrency semaphore.

    Use `client()` as an async context manager to obtain an S3 client
    (aiobotocore). The global semaphore limits total in-flight
    GETs across **all** workers.
    
    Credentials are resolved via the default credential chain:
    - Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    - Shared credentials file (~/.aws/credentials)
    - IAM role (for EC2 instances via instance metadata service)
    """

    def __init__(self, cfg, global_sem: asyncio.Semaphore):
        self.cfg = cfg
        self.global_sem = global_sem
        self._session = get_session()

    @contextlib.asynccontextmanager
    async def client(self):
        """Yield an async S3 client.
        
        Uses the default credential provider chain, which will automatically
        pick up IAM role credentials from EC2 instance metadata when running
        on EC2 instances with IAM roles attached.
        
        The credential chain is (in order):
        1. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
        2. Shared credentials file (~/.aws/credentials) - only if profile specified
        3. IAM role (for EC2 instances via instance metadata service)
        """
        # Create client without explicit credentials to use default credential chain
        # This will automatically use IAM role credentials on EC2 instances
        client_config = {
            "region_name": self.cfg.s3_region,
            # Don't pass credentials - let it use the default provider chain
            # which includes: env vars, ~/.aws/credentials (if profile set), IAM role (EC2)
        }
        
        # Only use a profile if explicitly set to a non-default value
        # For EC2 instances with IAM roles, we want to skip profile and use
        # the instance metadata service directly
        aws_profile = getattr(self.cfg, "aws_profile", None)
        if aws_profile and aws_profile != "default":
            client_config["profile_name"] = aws_profile
        # If aws_profile is None or "default", don't pass profile_name
        # This allows the default credential chain to work, including IAM roles
        
        async with self._session.create_client("s3", **client_config) as c:
            yield c