import asyncio
import hashlib
import json
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Union

from .types_common import Record, PipelineError, EndOfStream
from . import parquet_schemas as schema_mod

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.fs as pafs


def _truncate(s: Optional[str], max_len: int) -> Optional[str]:
    if s is None:
        return None
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def _open_filesystem_and_path(uri: str, cfg) -> tuple[Any, str]:
    """
    Returns (filesystem, path_within_fs) for a given uri.
    Supports s3://... and local paths.
    """
    if pafs is None:
        return None, uri

    if uri.startswith("s3://"):
        # Parse s3://bucket/key...
        # Use pyarrow's S3FileSystem (credentials from env/instance role/profile depending on setup).
        fs = pafs.S3FileSystem(region=getattr(cfg, "s3_region", None))
        path = uri[len("s3://") :]
        return fs, path

    # Local filesystem path
    fs = pafs.LocalFileSystem()
    return fs, uri


class S3ParquetWriter:
    """
    Writes a *single* Parquet file and a JSONL error sidecar.

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
        errors_jsonl_uri: Optional[str] = None,
        flush_every: int = 4096,
        max_error_message_len: int = 128,
    ):
        self.cfg = cfg
        self.q_post_processor_to_writer = q_post_processor_to_writer
        self.parquet_uri = parquet_uri
        self.errors_jsonl_uri = errors_jsonl_uri or (parquet_uri + ".errors.jsonl")
        self.flush_every = flush_every
        self.max_error_message_len = max_error_message_len

        self._schema = schema_mod.ld_build_schema()
        self._writer: Optional[Any] = None
        self._buffer: List[Dict[str, Any]] = []
        self._error_fh = None

        self._fs = None
        self._parquet_path = None
        self._err_fs = None
        self._errors_path = None

    def _ensure_open(self) -> None:
        """Open Parquet writer + error sidecar output stream."""
        if self._writer is not None:
            return

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

    def _row_from_record(self, rec: Record) -> Dict[str, Any]:
        # Identity + ok/error summary
        row = {
            "img_file_name": rec.task.img_filename,
            "img_s3_etag": rec.s3_etag,
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
            "img_s3_etag": getattr(err, "s3_etag", None),
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
        img_file_name = None
        if err.task is not None:
            img_file_name = getattr(err.task, "img_filename", None)

        # Prefer a stable schema: explicit keys + safe string fields.
        payload = {
            "img_s3_etag": getattr(err, "s3_etag", None),
            "stage": err.stage,
            "error_type": err.error_type,
            "message": err.message,
            "traceback": getattr(err, "traceback", None),
            "retryable": getattr(err, "retryable", False),
            "attempt": getattr(err, "attempt", 1),
            "task": None,
            "s3_etag": err.s3_etag,
        }
        if err.task is not None:
            payload["task"] = {
                "s3_key": getattr(err.task, "s3_key", None),
                "img_filename": getattr(err.task, "img_filename", None),
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

        table = pa.Table.from_pylist(self._buffer, schema=self._schema)
        self._writer.write_table(table)
        self._buffer.clear()

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
            else:
                # Record
                self._buffer.append(self._row_from_record(msg))

            # For small volumes, you can increase flush_every or set it very high.
            if len(self._buffer) >= self.flush_every:
                self._flush()

        # Final flush & close
        self._flush()
        self._close()
