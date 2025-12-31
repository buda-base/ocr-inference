from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from typing import TYPE_CHECKING

import boto3

if TYPE_CHECKING:
    from botocore.client import BaseClient

from batch.sqs.messages import TaskMessage

logger = logging.getLogger(__name__)

DEFAULT_VISIBILITY_TIMEOUT = 600  # 10 minutes
DEFAULT_WAIT_TIME = 20  # Long polling


@lru_cache(maxsize=1)
def _get_queue_url() -> str:
    url = os.environ.get("SQS_QUEUE_URL")
    if not url:
        raise RuntimeError("SQS_QUEUE_URL environment variable not set")
    return url


@lru_cache(maxsize=1)
def _get_client() -> BaseClient:
    return boto3.client("sqs")


def send_tasks_batch(task_messages: list[TaskMessage]) -> tuple[int, int]:
    """Send multiple task messages to SQS in batches of 10.

    Returns: (success_count, failure_count)
    """
    client = _get_client()
    queue_url = _get_queue_url()

    success_count = 0
    failure_count = 0

    for i in range(0, len(task_messages), 10):
        batch = task_messages[i : i + 10]
        entries = [
            {
                "Id": str(idx),
                "MessageBody": msg.to_json(),
                "MessageGroupId": str(msg.job_id),
                "MessageDeduplicationId": f"{msg.task_id}_{msg.attempt}",
            }
            for idx, msg in enumerate(batch)
        ]
        response = client.send_message_batch(QueueUrl=queue_url, Entries=entries)
        success_count += len(response.get("Successful", []))
        failed = response.get("Failed", [])
        failure_count += len(failed)
        for fail in failed:
            logger.error("Failed to send message %s: %s", fail["Id"], fail.get("Message"))

    return success_count, failure_count


def receive_task(
    visibility_timeout: int = DEFAULT_VISIBILITY_TIMEOUT, wait_time: int = DEFAULT_WAIT_TIME
) -> tuple[TaskMessage | None, str | None]:
    """Receive a single task message from SQS.

    Returns: (task_message, receipt_handle) or (None, None) if no message
    """
    client = _get_client()
    queue_url = _get_queue_url()

    response = client.receive_message(
        QueueUrl=queue_url,
        MaxNumberOfMessages=1,
        VisibilityTimeout=visibility_timeout,
        WaitTimeSeconds=wait_time,
        AttributeNames=["All"],
    )
    messages = response.get("Messages", [])
    if not messages:
        return None, None

    message = messages[0]
    receipt_handle = message["ReceiptHandle"]
    try:
        return TaskMessage.from_json(message["Body"]), receipt_handle
    except (json.JSONDecodeError, KeyError, TypeError):
        logger.exception("Failed to parse SQS message, deleting malformed message")
        delete_message(receipt_handle)
        return None, None


def delete_message(receipt_handle: str) -> None:
    """Delete a message from SQS after successful processing."""
    _get_client().delete_message(QueueUrl=_get_queue_url(), ReceiptHandle=receipt_handle)
