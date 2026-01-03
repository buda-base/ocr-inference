from __future__ import annotations

# ruff: noqa: T201
import argparse
import asyncio
import hashlib
import json
import os
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from batch.db.connection import close_pool, init_pool
from batch.db.models import JobStatus, VolumeInput
from batch.db.queries import (
    create_job_with_tasks,
    get_job_type,
    get_pending_tasks_with_volumes,
    update_job_status,
)
from batch.sqs.client import send_tasks_batch
from batch.sqs.messages import TaskMessage
from bdrc.s3_client import get_manifest_info

load_dotenv()


def _generate_job_key(volumes: list[tuple[str, str]]) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    volume_hash = hashlib.sha256(json.dumps(sorted(volumes)).encode()).hexdigest()[:8]
    return f"J{timestamp}_{volume_hash}"


def _generate_version_name(etag: bytes, existing_versions: set[str]) -> str:
    """Generate version_name from etag (first 6 hex), handle collisions."""
    version = etag.hex()[:6]
    if version not in existing_versions:
        return version
    # Collision: generate random suffix

    return secrets.token_hex(3)


def _fetch_volume_manifests(volumes: list[tuple[str, str]], input_bucket: str) -> list[VolumeInput]:
    """Fetch manifest info for all volumes from S3."""
    result = []
    seen_versions: set[str] = set()

    for w_id, i_id in volumes:
        manifest = get_manifest_info(input_bucket, w_id, i_id)
        if manifest is None:
            print(f"Error: Could not fetch manifest for {w_id}-{i_id}", file=sys.stderr)
            sys.exit(1)

        version_name = _generate_version_name(manifest["etag"], seen_versions)
        seen_versions.add(version_name)

        result.append(
            VolumeInput(
                bdrc_w_id=w_id,
                bdrc_i_id=i_id,
                manifest_etag=manifest["etag"],
                version_name=version_name,
                nb_images=manifest["nb_images"],
                manifest_last_modified_at=manifest["last_modified"],
            )
        )

    return result


INPUT_BUCKET = os.environ.get("INPUT_BUCKET", "archive.tbrc.org")


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

        print(f"Fetching manifests for {len(volumes)} volumes...")
        volume_inputs = _fetch_volume_manifests(volumes, INPUT_BUCKET)

        job_key = job_key or _generate_job_key(volumes)
        job, task_count = await create_job_with_tasks(
            job_key=job_key,
            type_id=job_type_id,
            volumes=volume_inputs,
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
            bdrc_w_id=row["bdrc_w_id"],
            bdrc_i_id=row["bdrc_i_id"],
            version_name=row["version_name"],
            attempt=row["attempts"] + 1,
        )
        for row in rows
    ]
    print(f"Publishing {len(task_messages)} tasks to SQS")
    return send_tasks_batch(task_messages)


def parse_volume_line(line: str) -> tuple[str, str] | None:
    """Parse a single volume line. Returns None for empty/comment lines."""
    volume = line.split("#")[0].strip()
    if not volume:
        return None
    if "-" not in volume:
        msg = f"Invalid volume format: {volume}. Expected W_id-I_id"
        raise ValueError(msg)
    w_id, i_id = volume.split("-", 1)
    return (w_id.strip(), i_id.strip())


def parse_volumes_from_string(volumes_arg: str) -> list[tuple[str, str]]:
    """Parse comma-separated volumes string."""
    result = []
    for item in volumes_arg.split(","):
        parsed = parse_volume_line(item)
        if parsed:
            result.append(parsed)
    if not result:
        raise ValueError("No volumes provided")
    return result


def parse_volumes_from_file(file_path: str) -> list[tuple[str, str]]:
    """Parse volumes from file (one per line, # for comments)."""
    result = []
    with Path(file_path).open() as f:
        lines = f.readlines()
    for line_num, line in enumerate(lines, 1):
        try:
            parsed = parse_volume_line(line)
        except ValueError as exc:
            msg = f"Line {line_num}: {exc}"
            raise ValueError(msg) from exc
        if parsed:
            result.append(parsed)
    if not result:
        raise ValueError(f"No volumes found in {file_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a new batch job")
    parser.add_argument(
        "--type-id",
        type=int,
        required=True,
        help="Job type ID from job_types table (defines model, encoding, etc.)",
    )
    volume_group = parser.add_mutually_exclusive_group(required=True)
    volume_group.add_argument(
        "--volumes",
        help="Comma-separated list of volumes (W_id-I_id format)",
    )
    volume_group.add_argument(
        "--volume-file",
        help="Path to file with volumes (one W_id-I_id per line, # for comments)",
    )
    parser.add_argument(
        "--job-key",
        help="Custom job key (auto-generated if not provided)",
    )
    args = parser.parse_args()

    try:
        volumes = parse_volumes_from_string(args.volumes) if args.volumes else parse_volumes_from_file(args.volume_file)
    except (ValueError, FileNotFoundError) as exc:
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
