from __future__ import annotations

# ruff: noqa: T201
import argparse
import asyncio
import hashlib
import json
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv

from batch.db.connection import close_pool, init_pool
from batch.db.models import JobStatus
from batch.db.queries import (
    create_job_with_tasks,
    get_job_type,
    get_pending_tasks_with_volumes,
    update_job_status,
)
from batch.sqs.client import send_tasks_batch
from batch.sqs.messages import TaskMessage

load_dotenv()


def _generate_job_key(volumes: list[tuple[str, str]]) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    volume_hash = hashlib.sha256(json.dumps(sorted(volumes)).encode()).hexdigest()[:8]
    return f"J{timestamp}_{volume_hash}"


async def run_create_job(
    job_type_id: int,
    volumes: list[tuple[str, str]],
    job_key: str | None = None,
) -> int:
    await init_pool()
    job = None

    try:
        job_type = await get_job_type(job_type_id)
        if job_type is None:
            print(f"Error: Job type {job_type_id} not found", file=sys.stderr)
            sys.exit(1)

        job_key = job_key or _generate_job_key(volumes)
        job, task_count = await create_job_with_tasks(
            job_key=job_key,
            type_id=job_type_id,
            volumes=volumes,
        )
        print(f"Created job: {job.job_key} (id={job.id}) with {task_count} tasks")

        success_count, failure_count = await _publish_tasks(job.id, job.job_key)
        if failure_count > 0:
            print(f"Warning: {failure_count} tasks failed to publish", file=sys.stderr)
        if success_count > 0:
            await update_job_status(job.id, JobStatus.RUNNING)
            print(f"Job {job.id} status changed to running")

    except Exception as exc:  # noqa: BLE001
        if job is not None:
            await update_job_status(job.id, JobStatus.FAILED)
        print(f"Error: Job creation failed: {exc}", file=sys.stderr)
        sys.exit(1)

    else:
        return job.id

    finally:
        await close_pool()


async def _publish_tasks(job_id: int, job_key: str) -> tuple[int, int]:
    rows = await get_pending_tasks_with_volumes(job_id)
    if not rows:
        print("No pending tasks to publish")
        return 0, 0

    task_messages = [
        TaskMessage(
            job_id=job_id,
            job_key=job_key,
            task_id=row["id"],
            volume_id=row["volume_id"],
            bdrc_w_id=row["bdrc_w_id"],
            bdrc_i_id=row["bdrc_i_id"],
            attempt=row["attempts"] + 1,
        )
        for row in rows
    ]
    print(f"Publishing {len(task_messages)} tasks to SQS")
    return send_tasks_batch(task_messages)


def parse_volumes(volumes_arg: str) -> list[tuple[str, str]]:
    result = []
    for item in volumes_arg.split(","):
        volume = item.strip()
        if not volume:
            continue
        if "-" not in volume:
            msg = f"Invalid volume format: {volume}. Expected W_id-I_id"
            raise ValueError(msg)
        w_id, i_id = volume.split("-", 1)
        result.append((w_id.strip(), i_id.strip()))
    if not result:
        raise ValueError("No volumes provided")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a new batch job")
    parser.add_argument(
        "--type-id",
        type=int,
        required=True,
        help="Job type ID from job_types table (defines model, encoding, etc.)",
    )
    parser.add_argument(
        "--volumes",
        required=True,
        help="Comma-separated list of volumes (W_id-I_id format)",
    )
    parser.add_argument(
        "--job-key",
        help="Custom job key (auto-generated if not provided)",
    )
    args = parser.parse_args()

    try:
        volumes = parse_volumes(args.volumes)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    job_id = asyncio.run(
        run_create_job(
            job_type_id=args.type_id,
            volumes=volumes,
            job_key=args.job_key,
        )
    )

    print(f"Job ID: {job_id}")


if __name__ == "__main__":
    main()
