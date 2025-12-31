from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bdrc.data import ArtifactConfig, Encoding, LayoutDetectionConfig, LineDetectionConfig
from bdrc.inference import OCRPipeline
from bdrc.pipeline import run_ocr_with_artifacts
from bdrc.utils import import_local_model

if TYPE_CHECKING:
    import numpy.typing as npt

    from bdrc.artifact_manager import ArtifactManager
    from bdrc.audit_logger import AuditLogger

logger = logging.getLogger(__name__)


@dataclass
class OCRConfig:
    encoding: str = "unicode"
    k_factor: float = 2.5
    bbox_tolerance: float = 4.0
    merge_lines: bool = False
    use_tps: bool = False
    line_mode: str = "line"
    artifact_granularity: str = "standard"


@dataclass
class ImageInput:
    name: str
    image: npt.NDArray


@dataclass
class ProcessingStats:
    total_images: int
    processed_count: int
    failed_count: int


def load_pipeline(model_path: str, line_mode: str = "line") -> OCRPipeline:
    config_path = Path(model_path) / "model_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Model config not found: {config_path}")

    ocr_model = import_local_model(str(Path(model_path)))
    if ocr_model is None:
        msg = f"Failed to load OCR model from: {model_path}"
        raise RuntimeError(msg)

    if line_mode == "line":
        line_config = LineDetectionConfig(
            model_file="Models/Lines/PhotiLines.onnx",
            patch_size=512,
        )
    else:
        line_config = LayoutDetectionConfig(
            model_file="Models/Layout/photi.onnx",
            patch_size=512,
            classes=["background", "image", "line", "caption", "margin"],
        )

    return OCRPipeline(ocr_model.config, line_config)


def process_images(
    pipeline: OCRPipeline,
    images: list[ImageInput],
    config: OCRConfig,
    artifact_manager: ArtifactManager | None = None,
    audit_logger: AuditLogger | None = None,
) -> ProcessingStats:
    target_encoding = Encoding.UNICODE if config.encoding == "unicode" else Encoding.WYLIE

    artifact_config = None
    if artifact_manager:
        is_standard = config.artifact_granularity == "standard"
        artifact_config = ArtifactConfig(
            enabled=True,
            granularity=config.artifact_granularity,
            save_detection=is_standard,
            save_dewarping=is_standard,
        )

    processed_count = 0
    failed_count = 0
    is_batch = len(images) > 1

    for image_input in images:
        if artifact_manager and is_batch:
            artifact_manager.set_current_page(image_input.name)

        try:
            run_ocr_with_artifacts(
                pipeline=pipeline,
                image=image_input.image,
                image_name=Path(image_input.name).stem,
                k_factor=config.k_factor,
                bbox_tolerance=config.bbox_tolerance,
                merge_lines=config.merge_lines,
                use_tps=config.use_tps,
                target_encoding=target_encoding,
                artifact_manager=artifact_manager,
                audit_logger=audit_logger,
                artifact_config=artifact_config,
            )
            processed_count += 1
            logger.info("Processed: %s", image_input.name)
        except Exception:
            logger.exception("OCR failed for %s", image_input.name)
            failed_count += 1
            if audit_logger:
                audit_logger.log_error(f"Pipeline failed for {image_input.name}")

    return ProcessingStats(
        total_images=len(images),
        processed_count=processed_count,
        failed_count=failed_count,
    )


def finalize_artifacts(
    artifact_manager: ArtifactManager,
    *,
    is_batch: bool = True,
) -> tuple[int, dict[str, Any] | None]:
    metrics = None
    if is_batch:
        metrics = artifact_manager.get_aggregate_metrics()
    artifact_manager.generate_manifest()
    failures = artifact_manager.finalize()
    artifact_manager.shutdown()
    return failures, metrics
