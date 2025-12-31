from __future__ import annotations

# ruff: noqa: T201
import argparse
import asyncio
import sys

from dotenv import load_dotenv

from batch.db.connection import close_pool, init_pool
from batch.db.models import JobStatus
from batch.db.queries import get_job, update_job_status

load_dotenv()


async def run_cancel_job(job_id: int) -> bool:
    await init_pool()

    try:
        job = await get_job(job_id)
        if job is None:
            print(f"Error: Job {job_id} not found", file=sys.stderr)
            return False

        if job.status in (JobStatus.COMPLETED, JobStatus.CANCELED):
            print(f"Warning: Job {job_id} is already {job.status.value}", file=sys.stderr)
            return False

        await update_job_status(job_id, JobStatus.CANCELED)
        print(f"Job {job_id} canceled")
        return True

    finally:
        await close_pool()


def main() -> None:
    parser = argparse.ArgumentParser(description="Cancel a batch job")
    parser.add_argument("--job-id", type=int, required=True, help="Job ID to cancel")

    args = parser.parse_args()

    success = asyncio.run(run_cancel_job(args.job_id))
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
