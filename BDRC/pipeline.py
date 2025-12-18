import time

import numpy.typing as npt

from bdrc.artifact_manager import ArtifactManager
from bdrc.audit_logger import AuditLogger
from bdrc.data import ArtifactConfig, Encoding, Line, OCRError
from bdrc.exporter import PageXMLExporter
from bdrc.inference import OCRPipeline
from bdrc.utils import get_text_bbox


def serialize_contours(contours: list[npt.NDArray]) -> list:
    return [c.tolist() for c in contours]


def serialize_lines(lines: list[Line]) -> list:
    return [
        {
            "guid": str(ln.guid),
            "bbox": {"x": ln.bbox.x, "y": ln.bbox.y, "w": ln.bbox.w, "h": ln.bbox.h},
            "center": ln.center,
        }
        for ln in lines
    ]


def run_ocr_with_artifacts(
    pipeline: OCRPipeline,
    image: npt.NDArray,
    image_name: str,
    *,
    k_factor: float = 2.5,
    bbox_tolerance: float = 4.0,
    merge_lines: bool = True,
    use_tps: bool = False,
    tps_threshold: float = 0.25,
    target_encoding: Encoding = Encoding.UNICODE,
    artifact_manager: ArtifactManager | None = None,
    audit_logger: AuditLogger | None = None,
    artifact_config: ArtifactConfig | None = None,
) -> tuple[npt.NDArray, list, list, float]:
    pipeline_start = time.perf_counter()
    save_det = artifact_manager and artifact_config and artifact_config.save_detection
    save_dew = artifact_manager and artifact_config and artifact_config.save_dewarping

    def log_start(stage: str, meta: dict | None = None) -> None:
        if audit_logger:
            audit_logger.log_stage_start(stage, metadata=meta)

    def log_end(stage: str, meta: dict | None = None, status: str = "success") -> None:
        if audit_logger:
            audit_logger.log_stage_end(stage, status=status, metadata=meta)

    def log_err(msg: object, stage: str) -> None:
        if audit_logger:
            audit_logger.log_error(str(msg), stage=stage)

    log_start(
        "ocr_pipeline",
        {
            "image_name": image_name,
            "image_shape": image.shape,
            "k_factor": k_factor,
            "bbox_tolerance": bbox_tolerance,
            "merge_lines": merge_lines,
            "use_tps": use_tps,
            "target_encoding": str(target_encoding),
        },
    )

    if artifact_manager:
        artifact_manager.create_directory_structure()
        artifact_manager.save_config()

    try:
        # STAGE 1: Line/Layout Detection
        log_start("line_detection")
        line_mask = pipeline.detect_lines(image)
        if save_det and artifact_manager:
            artifact_manager.save_image("line_mask", line_mask, "detection")
        log_end("line_detection", {"mask_shape": line_mask.shape})

        # STAGE 2: Build Line Data
        log_start("build_line_data")
        rot_img, rot_mask, line_contours, filtered_contours, page_angle = pipeline.build_lines(image, line_mask)
        if save_det and artifact_manager:
            artifact_manager.save_image("rotated_mask", rot_mask, "detection")
            artifact_manager.save_json(
                "contours_raw",
                {"count": len(line_contours), "contours": serialize_contours(line_contours)},
                "detection",
            )
            artifact_manager.save_json(
                "contours_filtered",
                {"count": len(filtered_contours), "contours": serialize_contours(filtered_contours)},
                "detection",
            )
        log_end(
            "build_line_data",
            {
                "rotation_angle": page_angle,
                "contour_count": len(line_contours),
                "filtered_count": len(filtered_contours),
            },
        )

        # STAGE 3: TPS Dewarping
        log_start("dewarping")
        dewarp_result = pipeline.apply_dewarping(
            rot_img, rot_mask, filtered_contours, page_angle, use_tps=use_tps, tps_threshold=tps_threshold
        )
        if save_dew and artifact_manager and dewarp_result.tps_ratio is not None:
            artifact_manager.save_json(
                "tps_analysis",
                {"ratio": float(dewarp_result.tps_ratio), "threshold": tps_threshold, "applied": dewarp_result.applied},
                "dewarping",
            )
            if dewarp_result.applied and dewarp_result.dewarped_mask is not None:
                artifact_manager.save_image("dewarped_mask", dewarp_result.dewarped_mask, "dewarping")
        log_end("dewarping", {"tps_ratio": dewarp_result.tps_ratio, "dewarping_applied": dewarp_result.applied})

        # STAGE 4: Extract Lines
        log_start("extract_lines")
        sorted_lines, line_images = pipeline.extract_lines(
            dewarp_result.work_img,
            rot_mask,
            dewarp_result.filtered_contours,
            merge_lines=merge_lines,
            k_factor=k_factor,
            bbox_tolerance=bbox_tolerance,
        )
        if artifact_manager and artifact_config:
            artifact_manager.save_json(
                "lines", {"count": len(sorted_lines), "lines": serialize_lines(sorted_lines)}, "lines"
            )
        log_end("extract_lines", {"lines_extracted": len(sorted_lines)})

        # STAGE 5: OCR Inference
        log_start("ocr_inference")
        ocr_lines = pipeline.run_text_recognition(line_images, sorted_lines, target_encoding=target_encoding)
        if audit_logger:
            for idx in range(len(ocr_lines)):
                audit_logger.log_operation(f"ocr_line_{idx + 1}", stage="ocr_inference")
        log_end("ocr_inference", {"lines_processed": len(ocr_lines)})

        # STAGE 6: Save Results
        if artifact_manager:
            artifact_manager.save_text(image_name, "\n".join(line.text for line in ocr_lines), "results", ext="txt")
            xml_exporter = PageXMLExporter("")
            xml_doc = xml_exporter.build_xml_document(
                image,
                image_name,
                xml_exporter.get_bbox_points(get_text_bbox(sorted_lines)),
                [xml_exporter.get_text_points(ln.contour) for ln in sorted_lines],
                ocr_lines,
            )
            artifact_manager.save_text(image_name, xml_doc, "results", ext="xml")

        # Pipeline Complete
        pipeline_duration = (time.perf_counter() - pipeline_start) * 1000
        log_end("ocr_pipeline")

        if artifact_manager:
            artifact_manager.save_metrics(
                {
                    "total_duration_ms": pipeline_duration,
                    "lines_detected": len(sorted_lines),
                    "lines_processed": len(ocr_lines),
                    "dewarping_applied": dewarp_result.applied,
                    "rotation_angle": page_angle,
                    "image_name": image_name,
                }
            )

    except OCRError:
        log_end("ocr_pipeline", status="failure")
        raise
    except Exception as e:
        log_err(f"OCR pipeline failed: {e}", "ocr_pipeline")
        log_end("ocr_pipeline", status="failure")
        raise OCRError(f"OCR pipeline failed: {e}") from e
    else:
        return rot_mask, sorted_lines, ocr_lines, page_angle
