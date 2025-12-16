"""Audit Logging for OCR Pipeline with JSON formatting."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class JSONFormatter(logging.Formatter):
    """Formatter that outputs log records as JSON."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.now(tz=UTC).isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
        }
        for key in ("job_id", "stage", "operation", "status", "metadata"):
            if hasattr(record, key):
                log_obj[key] = getattr(record, key)
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj, default=str)


class AuditLogger:
    """Structured audit logger for OCR pipeline operations."""

    def __init__(self, job_id: str, log_file: Path) -> None:
        self.job_id = job_id
        self.logger = logging.getLogger(f"ocr_audit_{job_id}")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        self.logger.propagate = False
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler.setFormatter(JSONFormatter())
        self.logger.addHandler(handler)

    def log(self, level: str, message: str, *, exc_info: bool = False, **kwargs: object) -> None:
        """Log a message with structured data."""
        extra = {"job_id": self.job_id, **{k: v for k, v in kwargs.items() if v is not None}}
        getattr(self.logger, level.lower())(message, extra=extra, exc_info=exc_info)

    def log_stage_start(self, stage: str, metadata: dict[str, Any] | None = None) -> None:
        """Log the start of a processing stage."""
        self.log("INFO", f"Starting stage: {stage}", stage=stage, operation="stage_start", metadata=metadata)

    def log_stage_end(self, stage: str, status: str = "success", metadata: dict[str, Any] | None = None) -> None:
        """Log the end of a processing stage."""
        self.log(
            "INFO", f"Completed stage: {stage}", stage=stage, operation="stage_end", status=status, metadata=metadata
        )

    def log_operation(self, operation: str, stage: str | None = None, status: str = "success") -> None:
        """Log a single operation."""
        self.log("INFO", f"Operation: {operation}", stage=stage, operation=operation, status=status)

    def log_error(
        self, error_msg: str, stage: str | None = None, operation: str | None = None, *, exc_info: bool = True
    ) -> None:
        """Log an error."""
        self.log("ERROR", error_msg, stage=stage, operation=operation, status="failure", exc_info=exc_info)
