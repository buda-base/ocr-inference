from __future__ import annotations

import argparse
import asyncio
import logging
import os

from dotenv import load_dotenv

from batch.worker.processor import WorkerConfig
from batch.worker.sqs_loop import run_sqs_worker

load_dotenv()

INPUT_BUCKET = os.environ.get("INPUT_BUCKET", "archive.tbrc.org")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SQS-based OCR batch worker")
    parser.add_argument("--model-dir", required=True, help="Directory containing OCR models")
    parser.add_argument("--output-bucket", default="bec.bdrc.io", help="S3 bucket for output (default: bec.bdrc.io)")
    parser.add_argument("--visibility-timeout", type=int, default=600, help="SQS visibility timeout in seconds")

    args = parser.parse_args()

    worker_config = WorkerConfig(
        model_dir=args.model_dir,
        input_bucket=INPUT_BUCKET,
        output_bucket=args.output_bucket,
    )

    asyncio.run(run_sqs_worker(worker_config, args.visibility_timeout))


if __name__ == "__main__":
    main()
