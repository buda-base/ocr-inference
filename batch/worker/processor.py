from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bdrc.artifact_manager import ArtifactManager
from bdrc.ocr_processor import (
    ImageInput,
    OCRConfig,
    finalize_artifacts,
    load_pipeline,
    process_images,
)
from bdrc.s3_client import download_image, get_image_list, get_s3_folder_prefix
from bdrc.storage import S3Storage

if TYPE_CHECKING:
    from bdrc.inference import OCRPipeline

logger = logging.getLogger(__name__)


@dataclass
class WorkerConfig:
    """Infrastructure config for the worker (not job-specific)."""

    model_dir: str
    input_bucket: str
    output_bucket: str


@dataclass
class ProcessingConfig:
    """Full processing config combining worker infrastructure + job type settings."""

    model_path: str
    output_bucket: str
    input_bucket: str
    job_type_name: str
    encoding: str = "unicode"
    k_factor: float = 2.5
    bbox_tolerance: float = 4.0
    merge_lines: bool = False
    use_tps: bool = False
    line_mode: str = "line"
    artifact_granularity: str = "standard"

    @classmethod
    def from_worker_and_job_type(
        cls,
        worker_config: WorkerConfig,
        model_name: str,
        job_type_name: str,
        *,
        encoding: str = "unicode",
        line_mode: str = "line",
        k_factor: float = 2.5,
        bbox_tolerance: float = 4.0,
        merge_lines: bool = False,
        use_tps: bool = False,
        artifact_granularity: str = "standard",
    ) -> ProcessingConfig:
        model_path = Path(worker_config.model_dir) / model_name
        return cls(
            model_path=str(model_path),
            output_bucket=worker_config.output_bucket,
            input_bucket=worker_config.input_bucket,
            job_type_name=job_type_name,
            encoding=encoding,
            k_factor=k_factor,
            bbox_tolerance=bbox_tolerance,
            merge_lines=merge_lines,
            use_tps=use_tps,
            line_mode=line_mode,
            artifact_granularity=artifact_granularity,
        )


@dataclass
class ProcessingResult:
    success: bool
    output_prefix: str
    image_count: int
    processed_count: int
    failed_count: int
    duration_ms: float
    error: str | None = None
    metrics: dict[str, Any] | None = None


class TaskProcessor:
    def __init__(self, config: ProcessingConfig) -> None:
        self.config = config
        self._pipeline: OCRPipeline | None = None

    def _ensure_pipeline_loaded(self) -> OCRPipeline:
        if self._pipeline is None:
            self._pipeline = load_pipeline(self.config.model_path, self.config.line_mode)
            logger.info("Pipeline loaded from %s", self.config.model_path)
        return self._pipeline

    def process_volume(self, work_id: str, image_group: str, version_name: str) -> ProcessingResult:
        start_time = time.perf_counter()
        output_prefix = f"{self.config.job_type_name}/{work_id}-{image_group}-{version_name}"

        try:
            pipeline = self._ensure_pipeline_loaded()
        except (FileNotFoundError, RuntimeError) as exc:
            return ProcessingResult(
                success=False,
                output_prefix="",
                image_count=0,
                processed_count=0,
                failed_count=0,
                duration_ms=(time.perf_counter() - start_time) * 1000,
                error=f"Model load failed: {exc}",
            )

        image_list = get_image_list(self.config.input_bucket, work_id, image_group)
        if not image_list:
            return ProcessingResult(
                success=False,
                output_prefix="",
                image_count=0,
                processed_count=0,
                failed_count=0,
                duration_ms=(time.perf_counter() - start_time) * 1000,
                error=f"No images found for {work_id}-{image_group}",
            )

        s3_prefix = get_s3_folder_prefix(work_id, image_group)

        images: list[ImageInput] = []
        for img_info in image_list:
            filename = img_info["filename"]
            s3_key = s3_prefix + filename
            img = download_image(self.config.input_bucket, s3_key)
            if img is None:
                logger.warning("Failed to download: s3://%s/%s", self.config.input_bucket, s3_key)
                continue
            images.append(ImageInput(name=filename, image=img))

        if not images:
            return ProcessingResult(
                success=False,
                output_prefix="",
                image_count=len(image_list),
                processed_count=0,
                failed_count=len(image_list),
                duration_ms=(time.perf_counter() - start_time) * 1000,
                error="Failed to download any images",
            )

        storage = S3Storage(
            bucket=self.config.output_bucket,
            base_prefix=output_prefix,
        )

        config_dict: dict[str, Any] = {
            "work_id": work_id,
            "image_group": image_group,
            "image_count": len(images),
            "k_factor": self.config.k_factor,
            "bbox_tolerance": self.config.bbox_tolerance,
            "merge_lines": self.config.merge_lines,
            "dewarp": self.config.use_tps,
            "encoding": self.config.encoding,
            "line_mode": self.config.line_mode,
        }

        artifact_manager = ArtifactManager(storage=storage, config=config_dict)
        artifact_manager.create_directory_structure()
        artifact_manager.save_config()

        ocr_config = OCRConfig(
            encoding=self.config.encoding,
            k_factor=self.config.k_factor,
            bbox_tolerance=self.config.bbox_tolerance,
            merge_lines=self.config.merge_lines,
            use_tps=self.config.use_tps,
            line_mode=self.config.line_mode,
            artifact_granularity=self.config.artifact_granularity,
        )

        stats = process_images(
            pipeline=pipeline,
            images=images,
            config=ocr_config,
            artifact_manager=artifact_manager,
        )

        failures, metrics = finalize_artifacts(artifact_manager, is_batch=True)
        if failures > 0:
            logger.warning("Storage had %d upload failures", failures)

        duration_ms = (time.perf_counter() - start_time) * 1000
        download_failures = len(image_list) - len(images)
        total_failed = stats.failed_count + download_failures
        success = total_failed == 0 and stats.processed_count > 0

        return ProcessingResult(
            success=success,
            output_prefix=output_prefix,
            image_count=len(image_list),
            processed_count=stats.processed_count,
            failed_count=total_failed,
            duration_ms=duration_ms,
            error=None if success else f"Failed {total_failed}/{len(image_list)} images",
            metrics=metrics,
        )
