
# Pipeline Overview

This directory contains the runtime pieces that make a volume worker efficient, safe, and easy to reason about.
If you are **not** a concurrency expert, start here — the most important concepts are the **coroutines**, **bounded queues**, and the **two‑lane GPU batcher**.

## Big Picture

```
S3 → Prefetcher (async) → Decoder (thread pool) → GPU Batcher (async; two lanes)
       │                                │                 └→ first-pass summary → TransformController
       │                                │                                                     │
       └────────────────────────────────┴─────────────────────────────────────────────────────┘
                                                                                 ↘ (if needed) Reprocess lane
                                                                                                       │
                                                                                                       ↓
                                                                                               S3ParquetWriter
```

- Each arrow between stages is a **bounded asyncio.Queue** that provides **backpressure**.
- Stages are **coroutines** (async tasks) except `Decoder`, which uses a **ThreadPoolExecutor** for CPU-bound OpenCV work that releases the GIL.
- The **GPU batcher** has **two lanes**:
  - the **initial lane** with normal priority (first pass on raw frames)
  - the **reprocess lane** with higher priority (second pass on transformed frames)
  A simple **weighted scheduler** favors the reprocess lane to keep second‑pass latency low without starving first‑pass work.

## Components

- `config.py` — **`PipelineConfig`** centralizes knobs (S3 concurrency caps, queue sizes, batch size/timeout, output prefixes).
- `types.py` — data classes that flow between stages:
  - **`ImageTask`** (what to fetch from S3),
  - **`DecodedFrame`** (a decoded image with size),
  - **`Record`** (a row for Parquet).
- `s3ctx.py` — **`S3Context`** holds the global S3 semaphore and yields an async S3 client (via `aioboto3` in real code).
- `prefetch.py` — **`Prefetcher`** lists/reads S3 keys **asynchronously** and pushes `(task, bytes)` into the first queue.
- `decoder.py` — **`Decoder`** turns bytes into frames in a **thread pool** and pushes `DecodedFrame` objects onward.
- `batcher.py` — **`GpuBatcher`** builds **micro‑batches** of 512×512 patches on GPU and runs the model; it emits first‑pass summaries to the `TransformController` or final `Record`s to the writer.
- `transform.py` — **`TransformController`** inspects first‑pass summaries, computes **rotation/TPS** with OpenCV, and either writes a final `Record` or enqueues a **transformed** frame into the high‑priority **reprocess lane**.
- `writer.py` — **`S3ParquetWriter`** streams rows to a **staging** Parquet on S3 and **publishes** atomically via server‑side copy + `_SUCCESS.json` marker.
- `worker.py` — **`VolumeWorker`** wires everything together with bounded queues and runs all stages concurrently.

## Why Bounded Queues? (Backpressure 101)

A bounded queue has a **maximum size**. When a downstream stage is slow, the upstream `await q.put(...)` **blocks**, naturally slowing the pipeline. This keeps memory stable and avoids flooding any stage.

## Coroutines vs Threads

- **Coroutines** (`async def ...`) are used for I/O‑bound stages (S3 networking, GPU scheduling). They are cheap and cooperative.
- **Threads** are used for CPU‑bound decoding with OpenCV (which releases the GIL during native calls). We keep a small, fixed pool.

## Two‑Lane GPU Batching (non‑obvious but important)

- First pass produces a **summary**; only **some** frames need transform + second pass.
- To keep completion time tight for those frames, the reprocess lane is served with a higher **weight** (e.g., 3:1). The batcher pulls up to `re_weight` items from the reprocess lane before it takes an initial‑lane item.
- Batching (e.g., 8/16 patches per forward) happens **across frames** inside each lane, so the GPU runs efficiently even when a single image has only a few patches.

## Quick Example

```python
import asyncio
from pipeline.config import PipelineConfig
from pipeline.types import ImageTask
from pipeline.s3ctx import S3Context
from pipeline.worker import VolumeWorker

async def main():
    cfg = PipelineConfig(
        s3_bucket="my-bucket",
        out_prefix="s3://my-bucket/out",
        staging_prefix="s3://my-bucket/staging",
    )
    tasks = [ImageTask(key=f"vol_A/img_{i:05d}.jpg", etag="etagA", size=None, volume_id="vol_A") for i in range(600)]
    global_sem = asyncio.Semaphore(cfg.s3_max_inflight_global)
    s3 = S3Context(cfg, global_sem)
    worker = VolumeWorker(cfg, s3, "vol_A", tasks)
    await worker.run()

if __name__ == "__main__":
    asyncio.run(main())
```

> Replace the placeholders in `decoder.py`, `batcher.py`, and `transform.py` with your OpenCV / PyTorch / Kornia logic. Everything else (queues, scheduling, S3 finalize) is ready to go.
