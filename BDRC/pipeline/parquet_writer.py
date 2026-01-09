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
import numpy as np


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
        # Schema-aligned row (see parquet_schemas.py)
        tps_points, tps_alpha = self._serialize_tps(rec.tps_data)
        row = {
            "img_file_name": rec.task.img_filename,
            "source_etag": rec.source_etag,
            "rotation_angle": rec.rotation_angle,
            "tps_points": tps_points,
            "tps_alpha": tps_alpha,
            "contours": self._serialize_contours(rec.contours),
            "nb_contours": int(rec.nb_contours),
            "contours_bboxes": self._serialize_bboxes(rec.contours_bboxes),
            # Hybrid error summary (nullable)
            "ok": True,
            "error_stage": None,
            "error_type": None,
            "error_message": None,
        }
        return row

    def _row_from_error(self, err: PipelineError) -> Dict[str, Any]:
        img_file_name = getattr(getattr(err, "task", None), "img_filename", None)
        row = {
            "img_file_name": img_file_name,
            "source_etag": getattr(err, "source_etag", None),
            "rotation_angle": None,
            "tps_points": None,
            "tps_alpha": None,
            "contours": None,
            "nb_contours": None,
            "contours_bboxes": None,
            "ok": False,
            "error_stage": err.stage,
            "error_type": err.error_type,
            "error_message": _truncate(err.message, self.max_error_message_len),
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
        try:
            table = pa.Table.from_pylist(self._buffer, schema=self._schema)
        except Exception as e:
            # Surface schema/serialization issues clearly in the UI.
            sample = None
            try:
                sample = self._buffer[0]
            except Exception:
                pass
            self._emit_progress(
                {
                    "type": "fatal",
                    "stage": "ParquetWriter._flush",
                    "error": f"{type(e).__name__}: {e}",
                    "sample_row_keys": sorted(list(sample.keys())) if isinstance(sample, dict) else None,
                }
            )
            raise
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
        # If the run had no errors, remove the (possibly empty) errors JSONL sidecar.
        # This keeps downstream consumers simpler and avoids leaving stale empty files around.
        if self._error_count == 0 and self._err_fs is not None and self._errors_path is not None:
            try:
                self._err_fs.delete_file(self._errors_path)
            except Exception:
                # Best-effort: ignore deletion failures (permissions / eventual consistency / etc.)
                pass

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

    # -----------------------------
    # Serialization helpers (PyArrow compatibility)
    # -----------------------------
    @staticmethod
    def _clamp_int16(v: int) -> int:
        if v < -32768:
            return -32768
        if v > 32767:
            return 32767
        return v

    def _serialize_contours(self, contours: Any) -> Any:
        """
        Schema: list<list<struct<x:int16,y:int16>>>
        Input: typically List[np.ndarray] where each ndarray is (N,1,2) or (N,2).
        """
        if contours is None:
            return None
        out = []
        for c in contours:
            if c is None:
                continue
            if isinstance(c, np.ndarray):
                pts = c.reshape(-1, 2)
                pts_list = pts.tolist()
            else:
                pts_list = list(c)
            contour_structs = []
            for xy in pts_list:
                x = int(xy[0])
                y = int(xy[1])
                contour_structs.append({"x": self._clamp_int16(x), "y": self._clamp_int16(y)})
            out.append(contour_structs)
        return out

    def _serialize_bboxes(self, bboxes: Any) -> Any:
        """
        Schema: list<struct<x:int16,y:int16,w:int16,h:int16>>
        Input: typically List[Tuple[int,int,int,int]]
        """
        if bboxes is None:
            return None
        out = []
        for b in bboxes:
            if b is None:
                continue
            x, y, w, h = b
            out.append(
                {
                    "x": self._clamp_int16(int(x)),
                    "y": self._clamp_int16(int(y)),
                    "w": self._clamp_int16(int(w)),
                    "h": self._clamp_int16(int(h)),
                }
            )
        return out

    def _serialize_tps(self, tps_data: Any) -> tuple[Any, Any]:
        """
        Schema:
          - tps_points: list<list<float32>>
          - tps_alpha: float16

        Input (from pipeline): (input_pts[N,2], output_pts[N,2], alpha) or None.

        We store one row per point as [in_y, in_x, out_y, out_x] to keep both mappings.
        """
        if not tps_data:
            return None, None
        try:
            in_pts, out_pts, alpha = tps_data
        except Exception:
            return None, None

        in_arr = np.asarray(in_pts, dtype=np.float32).reshape(-1, 2)
        out_arr = np.asarray(out_pts, dtype=np.float32).reshape(-1, 2)
        n = min(in_arr.shape[0], out_arr.shape[0])
        pts = []
        for i in range(n):
            iy, ix = float(in_arr[i, 0]), float(in_arr[i, 1])
            oy, ox = float(out_arr[i, 0]), float(out_arr[i, 1])
            pts.append([iy, ix, oy, ox])
        return pts, float(alpha)
