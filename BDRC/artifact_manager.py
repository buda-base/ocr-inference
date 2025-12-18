import os
from datetime import UTC, datetime
from typing import Any

import numpy.typing as npt

from bdrc.storage import LocalStorage, StorageBackend

SUBDIR_NAMES = {"detection", "dewarping", "lines", "results"}


class ArtifactManager:
    def __init__(
        self,
        storage: StorageBackend | None = None,
        base_output_dir: str | None = None,
        job_id: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.job_id = job_id or f"{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S')}_{os.urandom(3).hex()}"
        if storage is None:
            if base_output_dir is None:
                raise ValueError("Either storage or base_output_dir must be provided")
            storage = LocalStorage(f"{base_output_dir}/{self.job_id}")
        self.storage = storage
        self.config = config or {}
        self.manifest: list[dict[str, str]] = []
        self.page_metrics: dict[str, dict[str, Any]] = {}
        self.current_page: str | None = None
        self._current_prefix: str = ""

    @property
    def job_dir(self) -> str:
        return self.storage.uri_prefix

    def create_directory_structure(self) -> None:
        self.storage.ensure_dir("")

    def set_current_page(self, page_name: str) -> None:
        self.current_page = page_name
        self._current_prefix = page_name
        self.storage.ensure_dir(page_name)

    def _get_subdir_path(self, subdir: str) -> str:
        if subdir not in SUBDIR_NAMES:
            raise ValueError(f"Unknown subdirectory: {subdir}")
        return f"{self._current_prefix}/{subdir}" if self._current_prefix else subdir

    def _add_to_manifest(self, name: str, artifact_type: str, path: str) -> None:
        self.manifest.append(
            {"name": name, "type": artifact_type, "path": path, "timestamp": datetime.now(tz=UTC).isoformat()}
        )

    def save_config(self) -> None:
        self._add_to_manifest("config.json", "configuration", self.storage.write_json("config.json", self.config))

    def save_image(self, name: str, image: npt.NDArray, subdir: str, fmt: str = "png") -> str:
        path = self.storage.write_image(f"{self._get_subdir_path(subdir)}/{name}.{fmt}", image)
        self._add_to_manifest(name, "image", path)
        return path

    def save_json(self, name: str, data: object, subdir: str) -> str:
        path = self.storage.write_json(f"{self._get_subdir_path(subdir)}/{name}.json", data)
        self._add_to_manifest(name, "json", path)
        return path

    def save_text(self, name: str, text: str, subdir: str, ext: str = "txt") -> str:
        path = self.storage.write_text(f"{self._get_subdir_path(subdir)}/{name}.{ext}", text)
        self._add_to_manifest(name, "text", path)
        return path

    def generate_manifest(self) -> str:
        return self.storage.write_json(
            "manifest.json",
            {"job_id": self.job_id, "created": datetime.now(tz=UTC).isoformat(), "artifacts": self.manifest},
        )

    def save_metrics(self, metrics: dict[str, Any]) -> str:
        if self.current_page:
            self.page_metrics[self.current_page] = metrics
            return f"{self.job_dir}/metrics.json"
        return self.storage.write_json("metrics.json", metrics)

    def save_aggregate_metrics(self) -> str:
        pm = self.page_metrics
        total_duration = sum(m.get("total_duration_ms", 0) for m in pm.values())
        return self.storage.write_json(
            "metrics.json",
            {
                "job_summary": {
                    "total_pages": len(pm),
                    "successful_pages": sum(1 for m in pm.values() if m.get("status") != "failed"),
                    "total_duration_ms": total_duration,
                    "avg_duration_per_page_ms": total_duration / len(pm) if pm else 0,
                    "total_lines_detected": sum(m.get("lines_detected", 0) for m in pm.values()),
                },
                "per_page_metrics": pm,
            },
        )

    def finalize(self) -> int:
        return self.storage.finalize()

    def shutdown(self) -> None:
        self.storage.shutdown()
