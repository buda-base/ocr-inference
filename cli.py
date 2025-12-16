import argparse
import logging
import sys
from pathlib import Path

import cv2

from bdrc.artifact_manager import ArtifactManager
from bdrc.audit_logger import AuditLogger
from bdrc.data import ArtifactConfig, Encoding, LayoutDetectionConfig, LineDetectionConfig
from bdrc.exporter import TextExporter
from bdrc.inference import OCRPipeline
from bdrc.pipeline import run_ocr_with_artifacts
from bdrc.utils import import_local_model

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Tibetan OCR inference on images.")
    parser.add_argument("--model", required=True, help="Path to OCR model directory")
    parser.add_argument("--image", help="Path to a single image file")
    parser.add_argument("--folder", help="Path to a folder containing images")
    parser.add_argument("--output", required=True, help="Output directory for results")
    parser.add_argument("--encoding", choices=["unicode", "wylie"], default="unicode", help="Output encoding")
    parser.add_argument("--k-factor", type=float, default=2.5, help="Line extraction k-factor")
    parser.add_argument("--bbox-tolerance", type=float, default=4.0, help="Bounding box tolerance")
    parser.add_argument("--merge-lines", action="store_true", help="Merge line chunks")
    parser.add_argument("--dewarp", action="store_true", help="Apply TPS dewarping")
    parser.add_argument("--line-mode", choices=["line", "layout"], default="line", help="Line detection mode")
    parser.add_argument("--save-artifacts", action="store_true", help="Enable artifact saving")
    parser.add_argument("--artifact-output", default="output", help="Base directory for artifacts")
    parser.add_argument(
        "--artifact-granularity",
        choices=["minimal", "standard"],
        default="standard",
        help="Level of artifact detail to save",
    )
    args = parser.parse_args()

    if args.image and args.folder:
        parser.error("--image and --folder cannot be used together.")
    if not args.image and not args.folder:
        parser.error("You must specify either --image or --folder.")

    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Load model
    config_path = Path(args.model) / "model_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Model config not found: {config_path}")
    ocr_model = import_local_model(str(Path(args.model)))
    if ocr_model is None:
        raise RuntimeError(f"Failed to load OCR model from: {args.model}")

    # Line detection config
    if args.line_mode == "line":
        line_config = LineDetectionConfig(model_file="Models/Lines/PhotiLines.onnx", patch_size=512)
    else:
        line_config = LayoutDetectionConfig(
            model_file="Models/Layout/photi.onnx",
            patch_size=512,
            classes=["background", "image", "line", "caption", "margin"],
        )

    pipeline = OCRPipeline(ocr_model.config, line_config)
    target_encoding = Encoding.UNICODE if args.encoding == "unicode" else Encoding.WYLIE

    # Collect images
    is_batch_mode = bool(args.folder)
    if args.folder:
        image_paths = [str(p) for p in Path(args.folder).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
        if not image_paths:
            logger.warning("No images found in %s", args.folder)
            sys.exit(1)
    else:
        image_paths = [args.image]

    # Artifact setup
    artifact_manager = None
    audit_logger = None
    artifact_config = None

    if args.save_artifacts:
        is_standard = args.artifact_granularity == "standard"
        artifact_config = ArtifactConfig(
            enabled=True, granularity=args.artifact_granularity, save_detection=is_standard, save_dewarping=is_standard
        )

        artifact_manager = ArtifactManager(
            base_output_dir=args.artifact_output,
            job_id=None,
            config={
                "image_count": len(image_paths),
                "image_paths": [Path(p).name for p in image_paths],
                "k_factor": args.k_factor,
                "bbox_tolerance": args.bbox_tolerance,
                "merge_lines": args.merge_lines,
                "dewarp": args.dewarp,
                "encoding": args.encoding,
                "line_mode": args.line_mode,
                "artifact_granularity": args.artifact_granularity,
            },
        )
        artifact_manager.create_directory_structure()
        artifact_manager.save_config()

        if is_standard:
            audit_logger = AuditLogger(artifact_manager.job_id, artifact_manager.job_dir / "audit.log")

    # Process images
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            logger.error("Failed to load image: %s", img_path)
            if audit_logger:
                audit_logger.log_error(f"Failed to load image: {img_path}")
            continue

        page_name = Path(img_path).name
        base = Path(img_path).stem

        if artifact_manager and is_batch_mode:
            artifact_manager.set_current_page(page_name)

        try:
            _, _, ocr_lines, _ = run_ocr_with_artifacts(
                pipeline=pipeline,
                image=img,
                image_name=base,
                k_factor=args.k_factor,
                bbox_tolerance=args.bbox_tolerance,
                merge_lines=args.merge_lines,
                use_tps=args.dewarp,
                target_encoding=target_encoding,
                artifact_manager=artifact_manager,
                audit_logger=audit_logger,
                artifact_config=artifact_config,
            )
            if not artifact_manager:
                TextExporter(args.output).export_lines(base, ocr_lines)
                logger.info("Text output: %s", args.output)
        except Exception as e:
            logger.exception("OCR failed for %s", img_path)
            if audit_logger:
                audit_logger.log_error(f"Pipeline failed for {page_name}: {e}")

    # Finalize
    if artifact_manager:
        if is_batch_mode:
            artifact_manager.save_aggregate_metrics()
        artifact_manager.generate_manifest()


if __name__ == "__main__":
    main()
