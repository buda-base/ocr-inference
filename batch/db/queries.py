from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from .connection import get_connection, get_transaction
from .models import Job, JobStatus, JobType, Task, TaskStatus

if TYPE_CHECKING:
    from asyncpg import Connection, Record
    from asyncpg.pool import PoolConnectionProxy


# ─────────────────────────────────────────────────────────────────────────────
# Job Types
# ─────────────────────────────────────────────────────────────────────────────


def _row_to_job_type(row: Record) -> JobType:
    return JobType(
        id=row["id"],
        name=row["name"],
        model_name=row["model_name"],
        encoding=row["encoding"],
        line_mode=row["line_mode"],
        k_factor=row["k_factor"],
        bbox_tolerance=row["bbox_tolerance"],
        merge_lines=row["merge_lines"],
        use_tps=row["use_tps"],
        artifact_granularity=row["artifact_granularity"],
        description=row["description"],
    )


async def get_job_type(type_id: int) -> JobType | None:
    async with get_connection() as conn:
        row = await conn.fetchrow("SELECT * FROM job_types WHERE id = $1", type_id)
        return _row_to_job_type(row) if row else None


async def get_job_type_by_name(name: str) -> JobType | None:
    async with get_connection() as conn:
        row = await conn.fetchrow("SELECT * FROM job_types WHERE name = $1", name)
        return _row_to_job_type(row) if row else None


async def list_job_types() -> list[JobType]:
    async with get_connection() as conn:
        rows = await conn.fetch("SELECT * FROM job_types ORDER BY name")
        return [_row_to_job_type(row) for row in rows]


# ─────────────────────────────────────────────────────────────────────────────
# Jobs
# ─────────────────────────────────────────────────────────────────────────────


async def create_job_with_tasks(
    job_key: str,
    type_id: int,
    volumes: list[tuple[str, str]],
) -> tuple[Job, int]:
    """Create a job and its tasks in a single transaction."""
    async with get_transaction() as conn:
        job_row = await conn.fetchrow(
            """
            INSERT INTO jobs (job_key, type_id)
            VALUES ($1, $2)
            RETURNING *
            """,
            job_key,
            type_id,
        )
        job = _row_to_job(job_row)

        volume_ids = []
        for w_id, i_id in volumes:
            row = await conn.fetchrow(
                """
                INSERT INTO volumes (bdrc_w_id, bdrc_i_id) VALUES ($1, $2)
                ON CONFLICT (bdrc_w_id, bdrc_i_id) DO UPDATE SET bdrc_w_id = EXCLUDED.bdrc_w_id
                RETURNING id
                """,
                w_id,
                i_id,
            )
            volume_ids.append(row["id"])

        await conn.executemany(
            "INSERT INTO tasks (job_id, volume_id) VALUES ($1, $2)",
            [(job.id, vid) for vid in volume_ids],
        )
        await conn.execute(
            "UPDATE jobs SET total_tasks = $1 WHERE id = $2",
            len(volume_ids),
            job.id,
        )
        job.total_tasks = len(volume_ids)
        return job, len(volume_ids)


async def get_job(job_id: int) -> Job | None:
    async with get_connection() as conn:
        row = await conn.fetchrow("SELECT * FROM jobs WHERE id = $1", job_id)
        return _row_to_job(row) if row else None


async def get_job_by_key(job_key: str) -> Job | None:
    async with get_connection() as conn:
        row = await conn.fetchrow("SELECT * FROM jobs WHERE job_key = $1", job_key)
        return _row_to_job(row) if row else None


async def update_job_status(job_id: int, status: JobStatus) -> None:
    async with get_connection() as conn:
        now = datetime.now(timezone.utc)
        if status == JobStatus.RUNNING:
            await conn.execute("UPDATE jobs SET status = $1, started_at = $2 WHERE id = $3", status.value, now, job_id)
        elif status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELED):
            await conn.execute("UPDATE jobs SET status = $1, finished_at = $2 WHERE id = $3", status.value, now, job_id)
        else:
            await conn.execute("UPDATE jobs SET status = $1 WHERE id = $2", status.value, job_id)


async def list_jobs(
    status: str | None = None,
    exclude_statuses: list[str] | None = None,
    limit: int = 20,
) -> list[Job]:
    """List jobs, optionally filtered by status or excluding certain statuses."""
    async with get_connection() as conn:
        if status:
            rows = await conn.fetch(
                "SELECT * FROM jobs WHERE status = $1 ORDER BY created_at DESC LIMIT $2", status, limit
            )
        elif exclude_statuses:
            placeholders = ", ".join(f"${i + 1}" for i in range(len(exclude_statuses)))
            limit_param = len(exclude_statuses) + 1
            query = (
                f"SELECT * FROM jobs WHERE status NOT IN ({placeholders}) ORDER BY created_at DESC LIMIT ${limit_param}"  # noqa: S608
            )
            rows = await conn.fetch(query, *exclude_statuses, limit)
        else:
            rows = await conn.fetch("SELECT * FROM jobs ORDER BY created_at DESC LIMIT $1", limit)
        return [_row_to_job(row) for row in rows]


async def get_job_stats(job_id: int) -> dict[str, Any]:
    async with get_connection() as conn:
        row = await conn.fetchrow(
            """
            SELECT COUNT(*) FILTER (WHERE status = 'pending') as pending,
                   COUNT(*) FILTER (WHERE status = 'running') as running,
                   COUNT(*) FILTER (WHERE status = 'done') as done,
                   COUNT(*) FILTER (WHERE status = 'retryable_failed') as retryable,
                   COUNT(*) FILTER (WHERE status = 'terminal_failed') as terminal
            FROM tasks WHERE job_id = $1""",
            job_id,
        )
        return dict(row) if row else {}


async def get_failed_task_errors(job_id: int, limit: int = 5) -> list[dict[str, Any]]:
    """Get recent errors from failed tasks."""
    async with get_connection() as conn:
        rows = await conn.fetch(
            """
            SELECT t.id, v.bdrc_w_id, v.bdrc_i_id, t.attempts, t.last_error, t.status
            FROM tasks t JOIN volumes v ON t.volume_id = v.id
            WHERE t.job_id = $1 AND t.status IN ('retryable_failed', 'terminal_failed') AND t.last_error IS NOT NULL
            ORDER BY t.id DESC LIMIT $2""",
            job_id,
            limit,
        )
        return [
            {
                "task_id": row["id"],
                "volume": f"{row['bdrc_w_id']}-{row['bdrc_i_id']}",
                "attempts": row["attempts"],
                "status": row["status"],
                "error": json.loads(row["last_error"]) if row["last_error"] else None,
            }
            for row in rows
        ]


async def get_job_throughput(job_id: int) -> dict[str, Any]:
    """Calculate throughput stats for a job."""
    async with get_connection() as conn:
        row = await conn.fetchrow(
            """
            SELECT COUNT(*) FILTER (WHERE status = 'done') as done_count,
                   MIN(started_at) FILTER (WHERE status = 'done') as first_done,
                   MAX(done_at) FILTER (WHERE status = 'done') as last_done,
                   COUNT(*) FILTER (WHERE status IN ('pending', 'running', 'retryable_failed')) as remaining
            FROM tasks WHERE job_id = $1""",
            job_id,
        )
        if not row or row["done_count"] == 0:
            return {"tasks_per_hour": None, "eta_minutes": None}

        first_done, last_done = row["first_done"], row["last_done"]
        if first_done and last_done and first_done != last_done:
            duration_seconds = (last_done - first_done).total_seconds()
            if duration_seconds > 0:
                tasks_per_hour = (row["done_count"] / duration_seconds) * 3600
                eta_minutes = (row["remaining"] / tasks_per_hour) * 60 if tasks_per_hour > 0 else None
                return {
                    "tasks_per_hour": round(tasks_per_hour, 1),
                    "eta_minutes": round(eta_minutes, 1) if eta_minutes else None,
                }
        return {"tasks_per_hour": None, "eta_minutes": None}


# ─────────────────────────────────────────────────────────────────────────────
# Tasks
# ─────────────────────────────────────────────────────────────────────────────


async def get_pending_tasks_with_volumes(job_id: int) -> list[dict[str, Any]]:
    """Get pending tasks with volume info for publishing to SQS."""
    async with get_connection() as conn:
        rows = await conn.fetch(
            """
            SELECT t.id, t.volume_id, t.attempts, v.bdrc_w_id, v.bdrc_i_id
            FROM tasks t JOIN volumes v ON t.volume_id = v.id
            WHERE t.job_id = $1 AND t.status = 'pending' ORDER BY t.id""",
            job_id,
        )
        return [dict(row) for row in rows]


async def get_task_for_processing(task_id: int) -> Task | None:
    """Get task and increment attempts atomically for processing."""
    async with get_transaction() as conn:
        row = await conn.fetchrow(
            """
            UPDATE tasks SET attempts = attempts + 1, status = 'running', started_at = COALESCE(started_at, now())
            WHERE id = $1
            RETURNING id, job_id, attempts, max_attempts
            """,
            task_id,
        )
        if not row:
            return None
        return Task(
            id=row["id"],
            job_id=row["job_id"],
            attempts=row["attempts"],
            max_attempts=row["max_attempts"],
        )


async def complete_task(task_id: int, output_prefix: str) -> None:
    async with get_transaction() as conn:
        row = await conn.fetchrow(
            """
            UPDATE tasks SET status = 'done', done_at = now(), output_prefix = $1
            WHERE id = $2 RETURNING job_id""",
            output_prefix,
            task_id,
        )
        if row:
            await _update_job_counters(conn, row["job_id"])


async def save_task_metrics(
    task_id: int,
    total_pages: int,
    successful_pages: int,
    total_duration_ms: float,
    total_lines_detected: int,
    page_metrics: list[dict[str, Any]],
) -> None:
    avg_duration = total_duration_ms / total_pages if total_pages > 0 else 0.0
    async with get_transaction() as conn:
        await conn.execute(
            """
            INSERT INTO task_metrics (task_id, total_pages, successful_pages, total_duration_ms,
                                       avg_duration_per_page_ms, total_lines_detected)
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            task_id,
            total_pages,
            successful_pages,
            total_duration_ms,
            avg_duration,
            total_lines_detected,
        )
        if page_metrics:
            await conn.executemany(
                """
                INSERT INTO page_metrics (task_id, image_name, duration_ms, lines_detected,
                                          lines_processed, dewarping_applied, rotation_angle)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                [
                    (
                        task_id,
                        pm.get("image_name", ""),
                        pm.get("total_duration_ms", 0.0),
                        pm.get("lines_detected", 0),
                        pm.get("lines_processed", 0),
                        pm.get("dewarping_applied", False),
                        pm.get("rotation_angle", 0.0),
                    )
                    for pm in page_metrics
                ],
            )


async def fail_task(task_id: int, error: dict[str, Any], *, terminal: bool = False) -> None:
    async with get_transaction() as conn:
        row = await conn.fetchrow("SELECT job_id, attempts, max_attempts FROM tasks WHERE id = $1", task_id)
        if not row:
            return
        status = (
            TaskStatus.TERMINAL_FAILED
            if terminal or row["attempts"] >= row["max_attempts"]
            else TaskStatus.RETRYABLE_FAILED
        )
        await conn.execute(
            "UPDATE tasks SET status = $1, last_error = $2 WHERE id = $3", status.value, json.dumps(error), task_id
        )
        await _update_job_counters(conn, row["job_id"])


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


async def _update_job_counters(conn: Connection | PoolConnectionProxy, job_id: int) -> None:
    """Update job done_tasks/failed_tasks counters and transition status if complete."""
    await conn.execute("SELECT 1 FROM jobs WHERE id = $1 FOR UPDATE", job_id)

    stats = await conn.fetchrow(
        """
        SELECT COUNT(*) FILTER (WHERE status = 'done') as done,
               COUNT(*) FILTER (WHERE status = 'terminal_failed') as failed,
               COUNT(*) as total
        FROM tasks WHERE job_id = $1""",
        job_id,
    )
    if not stats:
        return

    done, failed, total = stats["done"], stats["failed"], stats["total"]
    await conn.execute(
        "UPDATE jobs SET done_tasks = $1, failed_tasks = $2, last_progress_at = now() WHERE id = $3",
        done,
        failed,
        job_id,
    )

    if done + failed >= total:
        new_status = JobStatus.FAILED if failed > 0 else JobStatus.COMPLETED
        await conn.execute("UPDATE jobs SET status = $1, finished_at = now() WHERE id = $2", new_status.value, job_id)


def _row_to_job(row: Record) -> Job:
    return Job(
        id=row["id"],
        job_key=row["job_key"],
        type_id=row["type_id"],
        status=JobStatus(row["status"]),
        created_at=row["created_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        total_tasks=row["total_tasks"],
        done_tasks=row["done_tasks"],
        failed_tasks=row["failed_tasks"],
        last_progress_at=row["last_progress_at"],
    )
