import json
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy.typing as npt

from bdrc.s3_client import AsyncS3Uploader


class StorageBackend(ABC):
    @property
    @abstractmethod
    def uri_prefix(self) -> str: ...
    @abstractmethod
    def write_text(self, path: str, text: str) -> str: ...
    @abstractmethod
    def write_json(self, path: str, data: object) -> str: ...
    @abstractmethod
    def write_image(self, path: str, image: npt.NDArray) -> str: ...
    @abstractmethod
    def ensure_dir(self, path: str) -> None: ...
    @abstractmethod
    def shutdown(self) -> None: ...

    def finalize(self) -> int:
        return 0


class LocalStorage(StorageBackend):
    def __init__(self, base_path: str) -> None:
        self.base_path = Path(base_path)

    @property
    def uri_prefix(self) -> str:
        return str(self.base_path)

    def _full_path(self, path: str) -> Path:
        full = self.base_path / path
        full.parent.mkdir(parents=True, exist_ok=True)
        return full

    def write_text(self, path: str, text: str) -> str:
        full_path = self._full_path(path)
        full_path.write_text(text, encoding="utf-8")
        return str(full_path)

    def write_json(self, path: str, data: object) -> str:
        full_path = self._full_path(path)
        with full_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        return str(full_path)

    def write_image(self, path: str, image: npt.NDArray) -> str:
        full_path = self._full_path(path)
        cv2.imwrite(str(full_path), image)
        return str(full_path)

    def ensure_dir(self, path: str) -> None:
        (self.base_path / path).mkdir(parents=True, exist_ok=True)


class S3Storage(StorageBackend):
    def __init__(self, bucket: str, base_prefix: str = "", max_workers: int = 10) -> None:
        self.bucket = bucket
        self.base_prefix = base_prefix.rstrip("/")
        self.uploader = AsyncS3Uploader(bucket, max_workers=max_workers)

    @property
    def uri_prefix(self) -> str:
        return f"s3://{self.bucket}/{self.base_prefix}"

    def _full_key(self, path: str) -> str:
        return f"{self.base_prefix}/{path}" if self.base_prefix else path

    def _upload(self, path: str, data: bytes, content_type: str) -> str:
        key = self._full_key(path)
        self.uploader.upload(key, data, content_type)
        return f"s3://{self.bucket}/{key}"

    def write_text(self, path: str, text: str) -> str:
        return self._upload(path, text.encode("utf-8"), "text/plain; charset=utf-8")

    def write_json(self, path: str, data: object) -> str:
        return self._upload(path, json.dumps(data, indent=2, default=str).encode("utf-8"), "application/json")

    def write_image(self, path: str, image: npt.NDArray) -> str:
        ext = Path(path).suffix.lower()
        _, encoded = cv2.imencode(ext, image)
        return self._upload(path, encoded.tobytes(), "image/png" if ext == ".png" else "image/jpeg")

    def ensure_dir(self, path: str) -> None:
        pass

    def finalize(self) -> int:
        return self.uploader.wait_all()

    def shutdown(self) -> None:
        self.uploader.shutdown()
