import gzip
import hashlib
import io
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import TypedDict

import boto3
import cv2
import numpy as np
import numpy.typing as npt
from botocore.client import BaseClient
from botocore.config import Config

logger = logging.getLogger(__name__)


class ImageInfo(TypedDict):
    filename: str
    height: int
    width: int
    pilmode: str


@lru_cache(maxsize=1)
def get_s3_client() -> BaseClient:
    return boto3.client("s3", config=Config(max_pool_connections=50, retries={"max_attempts": 3, "mode": "adaptive"}))


def get_s3_folder_prefix(w_id: str, i_id: str) -> str:
    md5_2 = hashlib.md5(w_id.encode(), usedforsecurity=False).hexdigest()[:2]
    return f"Works/{md5_2}/{w_id}/images/{w_id}-{i_id}/"


def gets3blob(bucket: str, key: str) -> io.BytesIO | None:
    try:
        return io.BytesIO(get_s3_client().get_object(Bucket=bucket, Key=key)["Body"].read())
    except Exception:
        logger.exception("Failed to get s3://%s/%s", bucket, key)
        return None


def get_image_list(bucket: str, w_id: str, i_id: str) -> list[ImageInfo] | None:
    blob = gets3blob(bucket, get_s3_folder_prefix(w_id, i_id) + "dimensions.json")
    return json.loads(gzip.decompress(blob.getvalue()).decode("utf8")) if blob else None


def download_image(bucket: str, key: str) -> npt.NDArray | None:
    try:
        data = get_s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        logger.exception("Failed to download s3://%s/%s", bucket, key)
        return None


def upload_bytes(bucket: str, key: str, data: bytes, content_type: str = "text/plain") -> None:
    get_s3_client().put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)


class AsyncS3Uploader:
    def __init__(self, bucket: str, max_workers: int = 10) -> None:
        self.bucket = bucket
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.futures: list = []

    def upload(self, key: str, data: bytes, content_type: str = "text/plain") -> None:
        self.futures.append(self.executor.submit(upload_bytes, self.bucket, key, data, content_type))

    def wait_all(self) -> int:
        failures = 0
        for f in self.futures:
            if (exc := f.exception()) is not None:
                logger.error("Upload failed", exc_info=exc)
                failures += 1
        self.futures.clear()
        return failures

    def shutdown(self) -> None:
        self.wait_all()
        self.executor.shutdown(wait=True)
