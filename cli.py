import argparse
import logging
import sys
from pathlib import Path

import cv2

from bdrc.artifact_manager import ArtifactManager
from bdrc.audit_logger import AuditLogger
from bdrc.ocr_processor import (
    ImageInput,
    OCRConfig,
    finalize_artifacts,
    load_pipeline,
    process_images,
)
from bdrc.s3_client import download_image, get_image_list, get_s3_folder_prefix
from bdrc.storage import S3Storage

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

    # Load pipeline using shared logic
    pipeline = load_pipeline(args.model, args.line_mode)

    # Collect images
    images: list[ImageInput] = []
    s3_prefix = ""

    if s3_input:
        image_list = get_image_list(args.input_bucket, args.work_id, args.image_group)
        if not image_list:
            logger.error("No images found for W=%s, I=%s", args.work_id, args.image_group)
            sys.exit(1)
        s3_prefix = get_s3_folder_prefix(args.work_id, args.image_group)
        for img_info in image_list:
            filename = img_info["filename"]
            s3_key = s3_prefix + filename
            img = download_image(args.input_bucket, s3_key)
            if img is None:
                logger.error("Failed to download image: s3://%s/%s", args.input_bucket, s3_key)
                continue
            images.append(ImageInput(name=filename, image=img))
    elif args.folder:
        for path in Path(args.folder).iterdir():
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            img = cv2.imread(str(path))
            if img is None:
                logger.error("Failed to load image: %s", path)
                continue
            images.append(ImageInput(name=path.name, image=img))
        if not images:
            logger.warning("No images found in %s", args.folder)
            sys.exit(1)
    else:
        img = cv2.imread(args.image)
        if img is None:
            logger.error("Failed to load image: %s", args.image)
            sys.exit(1)
        images.append(ImageInput(name=Path(args.image).name, image=img))

    is_batch_mode = len(images) > 1

    # Artifact setup
    artifact_manager = None
    audit_logger = None

    if args.save_artifacts:
        image_names = [img.name for img in images]
        config_dict = {
            "image_count": len(images),
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

        if args.artifact_granularity == "standard" and not s3_output:
            audit_logger = AuditLogger(artifact_manager.job_id, Path(artifact_manager.job_dir) / "audit.log")

        logger.info("Artifacts will be saved to: %s", artifact_manager.job_dir)

    # Process images using shared logic
    ocr_config = OCRConfig(
        encoding=args.encoding,
        k_factor=args.k_factor,
        bbox_tolerance=args.bbox_tolerance,
        merge_lines=args.merge_lines,
        use_tps=args.dewarp,
        line_mode=args.line_mode,
        artifact_granularity=args.artifact_granularity,
    )

    stats = process_images(
        pipeline=pipeline,
        images=images,
        config=ocr_config,
        artifact_manager=artifact_manager,
        audit_logger=audit_logger,
    )

    logger.info("Processed %d/%d images", stats.processed_count, stats.total_images)

    # Finalize
    if artifact_manager:
        failures, _metrics = finalize_artifacts(artifact_manager, is_batch=is_batch_mode)
        if failures > 0:
            logger.warning("Storage had %d failures", failures)
        logger.info("Artifacts saved to: %s", artifact_manager.job_dir)
    elif not args.save_artifacts and len(images) == 1:
        # Simple text export for single image without artifacts
        # Note: process_images doesn't return OCR lines, so this path needs the old approach
        # For now, artifacts are required for output
        logger.info("Use --save-artifacts to save results")


if __name__ == "__main__":
    main()
