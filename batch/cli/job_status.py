from __future__ import annotations

# ruff: noqa: T201
import argparse
import asyncio
import sys
from typing import TYPE_CHECKING

from dotenv import load_dotenv

from batch.db.connection import close_pool, init_pool
from batch.db.queries import (
    get_failed_task_errors,
    get_job,
    get_job_by_key,
    get_job_stats,
    get_job_throughput,
    get_job_type,
    list_jobs,
)

if TYPE_CHECKING:
    from batch.db.models import Job

load_dotenv()


def _format_eta(eta_minutes: float) -> str:
    minutes = 60
    if eta_minutes < minutes:
        return f"{eta_minutes} minutes"
    return f"{eta_minutes / minutes:.1f} hours"


async def _print_job_detail(job: Job) -> None:
    """Print detailed status for a single job."""
    stats = await get_job_stats(job.id)
    throughput = await get_job_throughput(job.id)
    errors = await get_failed_task_errors(job.id, limit=5)
    job_type = await get_job_type(job.type_id)

    type_display = f"{job_type.name} (id={job.type_id})" if job_type else f"Unknown (id={job.type_id})"

    print(f"\n{'=' * 60}")
    print(f"Job: {job.job_key} (id={job.id})")
    print(f"{'=' * 60}")
    print(f"Status:     {job.status.value}")
    print(f"Type:       {type_display}")
    print(f"Created:    {job.created_at.strftime('%Y-%m-%d %H:%M:%S') if job.created_at else 'N/A'}")
    if job.started_at:
        print(f"Started:    {job.started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    if job.finished_at:
        print(f"Finished:   {job.finished_at.strftime('%Y-%m-%d %H:%M:%S')}")

    print(f"\nProgress: {job.done_tasks}/{job.total_tasks} done, {job.failed_tasks} failed")
    if stats:
        print(f"  Pending: {stats.get('pending', 0)}, Running: {stats.get('running', 0)}")
        print(f"  Retryable: {stats.get('retryable', 0)}, Terminal: {stats.get('terminal', 0)}")

    if throughput["tasks_per_hour"]:
        eta = _format_eta(throughput["eta_minutes"]) if throughput["eta_minutes"] else "N/A"
        print(f"\nThroughput: {throughput['tasks_per_hour']} tasks/hour, ETA: {eta}")

    if errors:
        print("\nRecent Errors:")
        for err in errors:
            label = "terminal" if err["status"] == "terminal_failed" else "retryable"
            msg = err["error"].get("message", "") if err["error"] else ""
            print(f"  Task {err['task_id']} ({err['volume']}) [{label}]: {msg}")


async def run_job_status(
    job_id: int | None = None,
    job_key: str | None = None,
    status_filter: str | None = None,
    exclude_statuses: list[str] | None = None,
    limit: int = 20,
) -> None:
    await init_pool()

    try:
        if job_id is not None:
            job = await get_job(job_id)
            if job is None:
                print(f"Error: Job {job_id} not found", file=sys.stderr)
                return
            await _print_job_detail(job)
        elif job_key is not None:
            job = await get_job_by_key(job_key)
            if job is None:
                print(f"Error: Job key {job_key} not found", file=sys.stderr)
                return
            await _print_job_detail(job)
        else:
            jobs = await list_jobs(status=status_filter, exclude_statuses=exclude_statuses, limit=limit)
            if not jobs:
                print("No jobs found")
                return
            for job in jobs:
                await _print_job_detail(job)

    finally:
        await close_pool()


def main() -> None:
    parser = argparse.ArgumentParser(description="Get job status (lists all jobs if no ID provided)")
    parser.add_argument("--job-id", type=int, help="Job ID (shows single job)")
    parser.add_argument("--job-key", help="Job key (shows single job)")
    parser.add_argument("--include-status", help="Filter by status (created, running, completed, failed, canceled)")
    parser.add_argument(
        "--exclude-status",
        nargs="+",
        default=["canceled", "failed"],
        help="Exclude jobs with these statuses (default: canceled failed)",
    )
    parser.add_argument("--all", action="store_true", help="Show all jobs without exclusions")
    parser.add_argument("--limit", type=int, default=20, help="Max jobs to show when listing")

    args = parser.parse_args()

    exclude = None if args.all or args.include_status else args.exclude_status

    asyncio.run(
        run_job_status(
            job_id=args.job_id,
            job_key=args.job_key,
            status_filter=args.include_status,
            exclude_statuses=exclude,
            limit=args.limit,
        )
    )


if __name__ == "__main__":
    main()
