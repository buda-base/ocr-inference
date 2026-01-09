import asyncio
import hashlib
import json
import os
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Union

from .types_common import *
from . import parquet_schemas as schema_mod

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.fs as pafs

from urllib.parse import urlparse, unquote
from pathlib import Path


def _truncate(s: Optional[str], max_len: int) -> Optional[str]:
    if s is None:
        return None
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def _open_filesystem_and_path(uri: str, cfg) -> tuple[Any, str]:
    """
    Returns (filesystem, path_within_fs) for a given uri.
    Supports s3://..., file://..., and plain local paths.
    """
    if pafs is None:
        return None, uri

    if uri.startswith("s3://"):
        fs = pafs.S3FileSystem(region=getattr(cfg, "s3_region", None))
        path = uri[len("s3://"):]
        return fs, path

    # file:// URI
    if uri.startswith("file://"):
        u = urlparse(uri)
        # percent-decoding + normalize to local OS path
        local_path = unquote(u.path)
        fs = pafs.LocalFileSystem()
        return fs, local_path

    # Plain local path
    fs = pafs.LocalFileSystem()
    return fs, uri


class ParquetWriter:
    """
    Writes a single Parquet file and a JSONL error sidecar.

    Input queue: RecordMsg
      - Record rows become ok=True rows in Parquet.
      - PipelineError rows become ok=False rows in Parquet + full JSONL entry.
      - EndOfStream triggers flush/close.

    Notes:
      - No tmp file, no atomic finalize. For small volumes this is usually acceptable.
      - No per-writer success marker. The pipeline can write a run-level success marker separately.
    """

    def __init__(
        self,
        cfg,
        q_post_processor_to_writer: asyncio.Queue,
        parquet_uri: str,
        errors_jsonl_uri: str,
        progress: Optional[ProgressHook] = None,
    ):
        self.cfg = cfg
        self.q_post_processor_to_writer = q_post_processor_to_writer
        self.parquet_uri = parquet_uri
        self.errors_jsonl_uri = errors_jsonl_uri
        self.flush_every = cfg.flush_every
        self.max_error_message_len = cfg.max_error_message_len

        self._schema = schema_mod.ld_build_schema()
        self._writer: Optional[Any] = None
        self._buffer: List[Dict[str, Any]] = []
        self._error_fh = None

        self._fs = None
        self._parquet_path = None
        self._err_fs = None
        self._errors_path = None

        self._progress = progress
        
        # Error tracking for progress reporting
        self._success_count = 0
        self._error_count = 0
        self._error_by_stage: Dict[str, int] = {}
        self._last_summary_time = 0.0
        self._summary_interval = 100  # Emit summary every N items

    def _emit_progress(self, event: Dict[str, Any]) -> None:
        """Best-effort progress emission. Must never break the pipeline."""
        if self._progress is None:
            return
        try:
            self._progress(event)
        except Exception:
            # Never let UI/progress failures crash the worker
            pass

    def _ensure_open(self) -> None:
        """Open Parquet writer + error sidecar output stream."""
        if self._writer is not None:
            return

        try:
            self._fs, self._parquet_path = _open_filesystem_and_path(self.parquet_uri, self.cfg)
            self._err_fs, self._errors_path = _open_filesystem_and_path(self.errors_jsonl_uri, self.cfg)

            # Open output streams (direct write; no tmp)
            parquet_sink = self._fs.open_output_stream(self._parquet_path)
            self._writer = pq.ParquetWriter(
                parquet_sink,
                schema=self._schema,
                compression=getattr(self.cfg, "parquet_compression", "zstd"),
                use_dictionary=getattr(self.cfg, "parquet_dictionary_enabled", True),
                data_page_size=getattr(self.cfg, "parquet_data_page_size", 65536),
            )

            # Error sidecar (JSONL)
            self._error_fh = self._err_fs.open_output_stream(self._errors_path)
        except Exception as e:
            # Surface the problem to the UI immediately; this is the #1 source of "0 persisted".
            hint = None
            if str(self.parquet_uri).startswith("s3://") or str(self.errors_jsonl_uri).startswith("s3://"):
                hint = (
                    "Output is on S3. Check AWS credentials and that you have write permission "
                    "to the destination bucket/prefix (or use --output-folder file:///... to write locally)."
                )
            self._emit_progress(
                {
                    "type": "fatal",
                    "stage": "ParquetWriter._ensure_open",
                    "error": f"{type(e).__name__}: {e}",
                    "parquet_uri": self.parquet_uri,
                    "errors_jsonl_uri": self.errors_jsonl_uri,
                    "hint": hint,
                }
            )
            raise

    def _row_from_record(self, rec: Record) -> Dict[str, Any]:
        # Identity + ok/error summary
        row = {
            "img_file_name": rec.task.img_filename,
            "source_etag": rec.source_etag,
            "ok": True,
            "error_stage": None,
            "error_type": None,
            "error_message": None,
            "error_id": None,
            # Record fields
            "rotation_angle": rec.rotation_angle,
            "tps_data": rec.tps_data,
            "contours": rec.contours,
            "nb_contours": rec.nb_contours,
            "contours_bboxes": rec.contours_bboxes,
        }
        return row

    def _row_from_error(self, err: PipelineError) -> Dict[str, Any]:
        img_file_name = None
        if err.task is not None:
            img_file_name = getattr(err.task, "img_filename", None)

        row = {
            "img_file_name": img_file_name,
            "img_source_etag": getattr(err, "source_etag", None),
            "ok": False,
            "error_stage": err.stage,
            "error_type": err.error_type,
            "error_message": _truncate(err.message, self.max_error_message_len),
            # Feature fields are null for error rows
            "rotation_angle": None,
            "tps_data": None,
            "contours": None,
            "nb_contours": None,
            "contours_bboxes": None,
        }
        return row

    def _write_error_jsonl(self, err: PipelineError) -> None:
        """Write full error details as a JSONL line."""
        # Prefer a stable schema: explicit keys + safe string fields.
        task = err.task
        payload = {
            "source_uri": getattr(task, "source_uri", None),
            "img_filename": getattr(task, "img_filename", None),
            "source_etag": err.source_etag,
            "stage": err.stage,
            "error_type": err.error_type,
            "message": err.message,
            "traceback": getattr(err, "traceback", None),
            "retryable": getattr(err, "retryable", False),
            "attempt": getattr(err, "attempt", 1),
        }
        line = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        self._error_fh.write(line)

    def _flush(self) -> None:
        """Flush buffered rows to Parquet."""
        if not self._buffer:
            return
        if pq is None or pa is None:
            self._buffer.clear()
            return

        self._ensure_open()
        if self._writer is None:
            # No-op environment
            self._buffer.clear()
            return

        rows_len = len(self._buffer)
        self._emit_progress({"type": "flush", "state": "start", "rows": rows_len})
        table = pa.Table.from_pylist(self._buffer, schema=self._schema)
        self._writer.write_table(table)
        self._buffer.clear()
        self._emit_progress({"type": "flush", "state": "end", "rows": rows_len})

    def _close(self) -> None:
        """Close outputs."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self._error_fh is not None:
            self._error_fh.close()
            self._error_fh = None

    async def run(self) -> None:
        """
        Consume messages until EndOfStream, then flush and close.
        """
        try:
            # If pyarrow is absent, we still drain the queue to keep pipeline behavior consistent.
            # (Useful for unit tests / environments without optional deps.)
            while True:
                msg = await self.q_post_processor_to_writer.get()

                if isinstance(msg, EndOfStream):
                    break

                if isinstance(msg, PipelineError):
                    # Write error summary row to Parquet + full JSONL record
                    self._ensure_open()
                    self._buffer.append(self._row_from_error(msg))
                    self._write_error_jsonl(msg)

                    # Track errors
                    self._error_count += 1
                    stage = msg.stage
                    self._error_by_stage[stage] = self._error_by_stage.get(stage, 0) + 1

                    self._emit_progress(
                        {
                            "type": "item",
                            "ok": False,
                            "img": getattr(msg.task, "img_filename", "") if getattr(msg, "task", None) else "",
                            "stage": msg.stage,
                            "error_type": msg.error_type,
                        }
                    )
                else:
                    # Record
                    self._buffer.append(self._row_from_record(msg))
                    self._success_count += 1
                    self._emit_progress({"type": "item", "ok": True, "img": msg.task.img_filename})

                # Emit periodic error summary
                total = self._success_count + self._error_count
                if total > 0 and total % self._summary_interval == 0:
                    self._emit_error_summary()

                # For small volumes, you can increase flush_every or set it very high.
                if len(self._buffer) >= self.flush_every:
                    self._flush()

            # Final flush & close
            self._flush()
            self._close()

            # Emit final error summary
            if self._success_count > 0 or self._error_count > 0:
                self._emit_error_summary(final=True)

            self._emit_progress({"type": "close"})
        except Exception as e:
            self._emit_progress(
                {
                    "type": "fatal",
                    "stage": "ParquetWriter.run",
                    "error": f"{type(e).__name__}: {e}",
                    "parquet_uri": self.parquet_uri,
                    "errors_jsonl_uri": self.errors_jsonl_uri,
                }
            )
            raise
    
    def _emit_error_summary(self, final: bool = False) -> None:
        """Emit error rate summary for monitoring."""
        total = self._success_count + self._error_count
        if total == 0:
            return
        
        error_rate = (self._error_count / total) * 100.0 if total > 0 else 0.0
        
        summary = {
            "type": "error_summary",
            "final": final,
            "total": total,
            "success": self._success_count,
            "errors": self._error_count,
            "error_rate_pct": error_rate,
            "errors_by_stage": dict(self._error_by_stage),
        }
        self._emit_progress(summary)
