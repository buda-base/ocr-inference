
from dataclasses import dataclass

@dataclass
class PipelineConfig:
    """Configuration knobs for the pipeline.

    Key fields:
      - S3 concurrency caps (global/per-worker)
      - Bounded queue sizes per stage (backpressure)
      - GPU micro-batching (batch size/timeout)
      - Output prefixes (staging and final) and Parquet options
    """
    # S3
    s3_bucket: str
    s3_region: str = "us-east-1"
    aws_profile: str = "default"
    s3_max_inflight_global: int = 256   # GLOBAL cap across all workers
    s3_inflight_per_worker: int = 32    # per-worker GET concurrency
    s3_get_timeout_s: int = 60

    # Queues (bounded)
    max_q_bytes: int = 256
    max_q_frames: int = 128
    max_q_firstpass: int = 256
    max_q_reprocess: int = 64
    max_q_records: int = 256

    # CPU decode threads
    decode_threads: int = 8

    # GPU batching
    use_gpu: bool = True
    batch_size: int = 16
    batch_timeout_ms: int = 25
    cuda_streams: int = 2

    # Output
    parquet_compression: str = "zstd"
    parquet_data_page_size: int = 65536  # 64KB; tune
    parquet_dictionary_enabled: bool = True
    schema_version: str = "v1"

    # Decoder
    frame_max_width = 4096
    frame_max_height = 2048
    linearize = True
    normalize_background = False

    # LDTransform
    max_angle_deg: float = 5.0
    min_angle_deg: float = 0.3
    tps_add_corners: bool = True
    tps_alpha: float = 0.5
    legacy_tps_detect: bool = False
