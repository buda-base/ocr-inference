from __future__ import annotations

import argparse
import asyncio
import logging

from dotenv import load_dotenv

from batch.worker.processor import WorkerConfig
from batch.worker.sqs_loop import run_sqs_worker

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SQS-based OCR batch worker")
    parser.add_argument("--model-dir", required=True, help="Directory containing OCR models")
    parser.add_argument("--input-bucket", required=True, help="S3 bucket for input images")
    parser.add_argument("--output-bucket", required=True, help="S3 bucket for output")
    parser.add_argument("--output-prefix", default="output", help="S3 prefix for output")
    parser.add_argument("--upload-workers", type=int, default=10)
    parser.add_argument("--visibility-timeout", type=int, default=600, help="SQS visibility timeout in seconds")

    args = parser.parse_args()

    worker_config = WorkerConfig(
        model_dir=args.model_dir,
        input_bucket=args.input_bucket,
        output_bucket=args.output_bucket,
        output_prefix=args.output_prefix,
        upload_workers=args.upload_workers,
    )

    asyncio.run(run_sqs_worker(worker_config, args.visibility_timeout))


if __name__ == "__main__":
    main()
