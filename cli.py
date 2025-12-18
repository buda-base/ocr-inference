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
from bdrc.s3_client import download_image, get_image_list, get_s3_folder_prefix
from bdrc.storage import S3Storage
from bdrc.utils import import_local_model

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run Tibetan OCR inference on images.")
    parser.add_argument("--model", required=True, help="Path to OCR model directory")
    parser.add_argument("--image", help="Path to a single image file")
    parser.add_argument("--folder", help="Path to a folder containing images")
    parser.add_argument("--output", default="output", help="Output directory for results")
    parser.add_argument("--encoding", choices=["unicode", "wylie"], default="unicode", help="Output encoding")
    parser.add_argument("--k-factor", type=float, default=2.5, help="Line extraction k-factor")
    parser.add_argument("--bbox-tolerance", type=float, default=4.0, help="Bounding box tolerance")
    parser.add_argument("--merge-lines", action="store_true", help="Merge line chunks")
    parser.add_argument("--dewarp", action="store_true", help="Apply TPS dewarping")
    parser.add_argument("--line-mode", choices=["line", "layout"], default="line", help="Line detection mode")
    parser.add_argument("--save-artifacts", action="store_true", help="Enable artifact saving")
    parser.add_argument(
        "--artifact-granularity",
        choices=["minimal", "standard"],
        default="standard",
        help="Level of artifact detail to save",
    )
    # S3 mode arguments
    parser.add_argument("--input-bucket", help="S3 bucket for input images")
    parser.add_argument("--work-id", help="BDRC Work ID (W_id)")
    parser.add_argument("--image-group", help="BDRC Image Group ID (I_id)")
    parser.add_argument("--output-bucket", help="S3 bucket for output (defaults to input-bucket)")
    parser.add_argument("--output-prefix", default="ocr-output", help="S3 prefix for output artifacts")
    parser.add_argument("--upload-workers", type=int, default=10, help="Parallel S3 upload workers")
    args = parser.parse_args()

    # Determine mode: S3 input or local input
    s3_input = bool(args.input_bucket and args.work_id and args.image_group)
    local_input = bool(args.image or args.folder)
    # S3 output only if output_bucket is explicitly provided
    s3_output = bool(args.output_bucket)

    if s3_input and local_input:
        parser.error(
            "Cannot mix S3 input (--input-bucket, --work-id, --image-group) with local input (--image, --folder)."
        )
    if not s3_input and not local_input:
        parser.error(
            "You must specify either S3 input (--input-bucket, --work-id, --image-group) "
            "or local input (--image or --folder)."
        )
    if args.image and args.folder:
        parser.error("--image and --folder cannot be used together.")

    if not s3_output:
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
    image_list = []  # For S3 mode: list of ImageInfo dicts
    image_paths = []  # For local mode: list of file paths
    s3_prefix = ""

    if s3_input:
        image_list = get_image_list(args.input_bucket, args.work_id, args.image_group)
        if not image_list:
            logger.error("No images found for W=%s, I=%s", args.work_id, args.image_group)
            sys.exit(1)
        s3_prefix = get_s3_folder_prefix(args.work_id, args.image_group)
        is_batch_mode = len(image_list) > 1
    elif args.folder:
        image_paths = [str(p) for p in Path(args.folder).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
        if not image_paths:
            logger.warning("No images found in %s", args.folder)
            sys.exit(1)
        is_batch_mode = True
    else:
        image_paths = [args.image]
        is_batch_mode = False

    # Artifact setup
    artifact_manager = None
    audit_logger = None
    artifact_config = None

    if args.save_artifacts:
        is_standard = args.artifact_granularity == "standard"
        artifact_config = ArtifactConfig(
            enabled=True, granularity=args.artifact_granularity, save_detection=is_standard, save_dewarping=is_standard
        )

        image_count = len(image_list) if s3_input else len(image_paths)
        image_names = [img["filename"] for img in image_list] if s3_input else [Path(p).name for p in image_paths]

        config_dict = {
            "image_count": image_count,
            "image_paths": image_names,
            "k_factor": args.k_factor,
            "bbox_tolerance": args.bbox_tolerance,
            "merge_lines": args.merge_lines,
            "dewarp": args.dewarp,
            "encoding": args.encoding,
            "line_mode": args.line_mode,
            "artifact_granularity": args.artifact_granularity,
        }

        if s3_input:
            config_dict["work_id"] = args.work_id
            config_dict["image_group"] = args.image_group
            config_dict["input_bucket"] = args.input_bucket

        if s3_output:
            config_dict["output_bucket"] = args.output_bucket
            storage = S3Storage(
                bucket=args.output_bucket,
                base_prefix=args.output_prefix,
                max_workers=args.upload_workers,
            )
            artifact_manager = ArtifactManager(
                storage=storage,
                config=config_dict,
            )
        else:
            artifact_manager = ArtifactManager(
                base_output_dir=args.output,
                config=config_dict,
            )

        artifact_manager.create_directory_structure()
        artifact_manager.save_config()

        if is_standard and not s3_output:
            audit_logger = AuditLogger(artifact_manager.job_id, Path(artifact_manager.job_dir) / "audit.log")

        logger.info("Artifacts will be saved to: %s", artifact_manager.job_dir)

    # Process images
    if s3_input:
        # S3 mode: iterate over image_list from dimensions.json
        for img_info in image_list:
            filename = img_info["filename"]
            s3_key = s3_prefix + filename
            img = download_image(args.input_bucket, s3_key)
            if img is None:
                logger.error("Failed to download image: s3://%s/%s", args.input_bucket, s3_key)
                continue

            page_name = filename
            base = Path(filename).stem

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
                    artifact_manager=artifact_manager,  # type: ignore[arg-type]
                    audit_logger=audit_logger,
                    artifact_config=artifact_config,
                )
                logger.info("Processed: %s", filename)
            except Exception:
                logger.exception("OCR failed for %s", filename)
    else:
        # Local mode: iterate over file paths
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
        failures = artifact_manager.finalize()
        if failures > 0:
            logger.warning("Storage had %d failures", failures)
        artifact_manager.shutdown()
        logger.info("Artifacts saved to: %s", artifact_manager.job_dir)


if __name__ == "__main__":
    main()
