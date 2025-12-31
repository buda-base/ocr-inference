from __future__ import annotations

import asyncio
import logging
import signal
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import asyncpg.exceptions

from batch.db.connection import close_pool, init_pool
from batch.db.models import JobStatus, JobType
from batch.db.queries import (
    complete_task,
    fail_task,
    get_job,
    get_job_type,
    get_task_for_processing,
    save_task_metrics,
)
from batch.sqs.client import delete_message, receive_task
from batch.worker.processor import ProcessingConfig, TaskProcessor, WorkerConfig

if TYPE_CHECKING:
    from batch.sqs.messages import TaskMessage

logger = logging.getLogger(__name__)


MAX_CACHED_PROCESSORS = 10


class SQSWorkerLoop:
    def __init__(self, worker_config: WorkerConfig, visibility_timeout: int = 600) -> None:
        self.worker_config = worker_config
        self.visibility_timeout = visibility_timeout
        self._shutdown_event = asyncio.Event()
        self._job_type_cache: dict[int, JobType] = {}
        self._get_processor = lru_cache(maxsize=MAX_CACHED_PROCESSORS)(self._create_processor)

    def _setup_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()

        def handle_signal() -> None:
            logger.info("Shutdown signal received")
            self._shutdown_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, handle_signal)

    async def _get_job_type(self, type_id: int) -> JobType | None:
        if type_id in self._job_type_cache:
            return self._job_type_cache[type_id]
        job_type = await get_job_type(type_id)
        if job_type:
            self._job_type_cache[type_id] = job_type
        return job_type

    def _create_processor(self, job_type_id: int) -> TaskProcessor:
        job_type = self._job_type_cache[job_type_id]
        model_path = Path(self.worker_config.model_dir) / job_type.model_name
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        config = ProcessingConfig.from_worker_and_job_type(
            worker_config=self.worker_config,
            model_name=job_type.model_name,
            encoding=job_type.encoding,
            line_mode=job_type.line_mode,
            k_factor=job_type.k_factor,
            bbox_tolerance=job_type.bbox_tolerance,
            merge_lines=job_type.merge_lines,
            use_tps=job_type.use_tps,
            artifact_granularity=job_type.artifact_granularity,
        )
        processor = TaskProcessor(config)
        logger.info("Created processor for job type %s (model: %s)", job_type.name, model_path)
        return processor

    async def _process_sqs_message(
        self,
        task_message: TaskMessage,
        receipt_handle: str,
    ) -> None:
        job = await get_job(task_message.job_id)
        if job is None:
            logger.error("Job %d not found, deleting message", task_message.job_id)
            delete_message(receipt_handle)
            return

        if job.status == JobStatus.CANCELED:
            logger.info("Job %d is canceled, skipping task %d", job.id, task_message.task_id)
            delete_message(receipt_handle)
            return

        job_type = await self._get_job_type(job.type_id)
        if job_type is None:
            logger.error("Job type %d not found for job %d, deleting message", job.type_id, job.id)
            delete_message(receipt_handle)
            return

        task = await get_task_for_processing(task_message.task_id)
        if task is None:
            logger.error("Task %d not found, deleting message", task_message.task_id)
            delete_message(receipt_handle)
            return

        logger.info(
            "Processing task %d: %s-%s (attempt %d/%d, type: %s)",
            task_message.task_id,
            task_message.bdrc_w_id,
            task_message.bdrc_i_id,
            task.attempts,
            task.max_attempts,
            job_type.name,
        )

        try:
            processor = self._get_processor(job_type.id)
        except FileNotFoundError:
            logger.exception("Model not found for job type %s", job_type.name)
            await fail_task(
                task_id=task_message.task_id,
                error={"message": "Model not found", "attempt": task.attempts},
                terminal=True,
            )
            delete_message(receipt_handle)
            return

        result = await asyncio.to_thread(
            processor.process_volume, task_message.bdrc_w_id, task_message.bdrc_i_id, job.job_key
        )

        if result.success:
            await complete_task(
                task_id=task_message.task_id,
                output_prefix=result.output_prefix,
            )
            if result.metrics:
                job_summary = result.metrics.get("job_summary", {})
                per_page = result.metrics.get("per_page_metrics", {})
                await save_task_metrics(
                    task_id=task_message.task_id,
                    total_pages=job_summary.get("total_pages", 0),
                    successful_pages=job_summary.get("successful_pages", 0),
                    total_duration_ms=job_summary.get("total_duration_ms", 0.0),
                    total_lines_detected=job_summary.get("total_lines_detected", 0),
                    page_metrics=list(per_page.values()),
                )
            delete_message(receipt_handle)
            logger.info(
                "Task %d completed: %d/%d images in %.1fs",
                task_message.task_id,
                result.processed_count,
                result.image_count,
                result.duration_ms / 1000,
            )
        else:
            is_terminal = task.attempts >= task.max_attempts
            await fail_task(
                task_id=task_message.task_id,
                error={
                    "message": result.error,
                    "attempt": task.attempts,
                    "duration_ms": result.duration_ms,
                },
                terminal=is_terminal,
            )
            if is_terminal:
                delete_message(receipt_handle)
            logger.warning(
                "Task %d failed (attempt %d/%d): %s",
                task_message.task_id,
                task.attempts,
                task.max_attempts,
                result.error,
            )

    async def run(self) -> None:
        await init_pool()
        try:
            self._setup_signal_handlers()
            logger.info(
                "SQS worker loop started (model_dir=%s, input_bucket=%s, output_bucket=%s, output_prefix=%s)",
                self.worker_config.model_dir,
                self.worker_config.input_bucket,
                self.worker_config.output_bucket,
                self.worker_config.output_prefix,
            )

            while not self._shutdown_event.is_set():
                try:
                    task_message, receipt_handle = await asyncio.to_thread(receive_task, self.visibility_timeout, 20)
                    if task_message is None or receipt_handle is None:
                        continue
                    await self._process_sqs_message(task_message, receipt_handle)
                except (
                    asyncpg.exceptions.ConnectionDoesNotExistError,
                    asyncpg.exceptions.InterfaceError,
                    ConnectionResetError,
                ):
                    logger.warning("Database connection lost, reconnecting...")
                    await close_pool()
                    await init_pool()
                except Exception:
                    logger.exception("Error processing SQS message")
                    await asyncio.sleep(1.0)

            logger.info("SQS worker loop finished")
        finally:
            await close_pool()


async def run_sqs_worker(worker_config: WorkerConfig, visibility_timeout: int = 600) -> None:
    worker = SQSWorkerLoop(worker_config=worker_config, visibility_timeout=visibility_timeout)
    await worker.run()
