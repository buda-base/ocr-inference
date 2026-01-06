
import asyncio
from .config import PipelineConfig
from .types_common import ImageTask
from .ld_volume_worker import LDVolumeWorker

async def main():
    cfg = PipelineConfig(
        s3_bucket="YOUR_BUCKET",
        s3_region="us-east-1",
        s3_max_inflight_global=256,
        s3_inflight_per_worker=32,
        decode_threads=8,
        use_gpu=True,
        batch_size=16,
        batch_timeout_ms=25,
        out_prefix="s3://YOUR_BUCKET/out",
        staging_prefix="s3://YOUR_BUCKET/staging" )

    # Dummy manifest with two volumes of ~600 each
    volumes = {
        "vol_A": [ImageTask(key=f"vol_A/img_{i:05d}.jpg", filename="") for i in range(600)],
        "vol_B": [ImageTask(key=f"vol_B/img_{i:05d}.jpg", filename="") for i in range(600)],
    }

    global_sem = asyncio.Semaphore(cfg.s3_max_inflight_global)
    s3ctx = S3Context(cfg, global_sem)

    workers = [LDVolumeWorker(cfg, s3ctx, vid, tasks) for vid, tasks in volumes.items()]

    async def run_worker(w: LDVolumeWorker):
        try:
            await w.run()
        except Exception as e:
            print(f"Worker {w.volume_id} failed: {e}")
            raise

    await asyncio.gather(*(run_worker(w) for w in workers))

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
