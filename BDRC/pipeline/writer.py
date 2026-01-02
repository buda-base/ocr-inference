
import os, json, time, asyncio
from typing import List, Optional
from .types import Record
from . import schema as schema_mod

# Optional imports guarded for environments without pyarrow/s3fs
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.fs as pafs
except Exception:  # pragma: no cover
    pa = None
    pq = None
    pafs = None

class S3ParquetWriter:
        """Single-file Parquet writer with robust S3 staging→finalize flow.

        Input: Records from q_records.
        Behavior: buffers into row groups, writes to staging key, then server-side
        copies to final key and writes a _SUCCESS.json marker.
        """
    def __init__(self, cfg, q_records: asyncio.Queue, volume_id: str):
        self.cfg = cfg
        self.q_records = q_records
        self.volume_id = volume_id
        self.schema = schema_mod.build_schema()
        self.writer = None
        self.buffer: List[Record] = []
        self.flush_every = 4096  # rows per flush
        self.count_rows = 0

        # Paths
        self.staging_key = f"{self.cfg.staging_prefix.rstrip('/')}/{self.volume_id}.parquet.tmp"
        self.final_key   = f"{self.cfg.out_prefix.rstrip('/')}/{self.volume_id}.parquet"
        self.success_key = f"{self.cfg.out_prefix.rstrip('/')}/{self.volume_id}._SUCCESS.json"

    def _ensure_writer(self):
        if pq is None or self.schema is None:
            return
        if self.writer is None:
            fs = pafs.FileSystem.from_uri(self.staging_key)[0]  # resolves filesystem
            # Parquet write options
            props = pq.ParquetWriter(
                where=self.staging_key,
                schema=self.schema,
                filesystem=fs,
                compression=self.cfg.parquet_compression,
            )
            # We keep the writer object simple: recreate using pq.ParquetWriter directly
            self.writer = props

    def _records_to_table(self, rows: List[Record]):
        if pa is None:
            return None
        data = {
            "img_file_name": [r.img_file_name for r in rows],
            "img_s3_etag":  [r.img_s3_etag  for r in rows],
            "resized_w":    [r.resized_w    for r in rows],
            "resized_h":    [r.resized_h    for r in rows],
            "rotation_angle":[r.rotation_angle for r in rows],
            "tps_points":   [r.tps_points   for r in rows],
            "lines_contours":[r.lines_contours for r in rows],
            "nb_lines":     [r.nb_lines     for r in rows],
        }
        return pa.Table.from_pydict(data, schema=self.schema)

    def _finalize_atomic(self):
        if pafs is None:
            return
        fs, final_path = pafs.FileSystem.from_uri(self.final_key)
        fs_stage, stage_path = pafs.FileSystem.from_uri(self.staging_key)
        # S3 has no atomic rename: use server-side copy+delete
        fs.copy_file(stage_path, final_path)
        fs.delete_file(stage_path)
        # write a small success marker with metadata
        meta = {"volume_id": self.volume_id, "rows": self.count_rows, "schema_version": self.cfg.schema_version, "ts": int(time.time())}
        with fs.open_output_stream(self.success_key) as out:
            out.write(json.dumps(meta).encode("utf-8"))

    async def run(self):
            """Stream records to Parquet; flush periodically; finalize atomically on completion."""
        # Best-effort streaming; on environments without pyarrow this becomes a no-op skeleton
        while True:
            rec = await self.q_records.get()
            if rec is None:
                if self.buffer and pq is not None:
                    self._ensure_writer()
                    tbl = self._records_to_table(self.buffer)
                    if tbl is not None:
                        self.writer.write_table(tbl)
                    self.buffer.clear()
                if self.writer is not None:
                    self.writer.close()
                    self._finalize_atomic()
                break
            self.buffer.append(rec)
            self.count_rows += 1
            if len(self.buffer) >= self.flush_every and pq is not None:
                self._ensure_writer()
                tbl = self._records_to_table(self.buffer)
                if tbl is not None:
                    self.writer.write_table(tbl)
                self.buffer.clear()
