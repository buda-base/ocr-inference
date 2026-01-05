
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
    max_q_prefetcher_to_decoder: int = 256
    max_q_decoder_to_gpu_pass_1: int = 128
    max_q_gpu_pass_1_to_post_processor: int = 256
    max_q_post_processor_to_gpu_pass_2: int = 64
    max_q_gpu_pass_2_to_post_processor: int = 128
    max_q_post_processor_to_writer: int = 256


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
    patch_size = 512
    patch_vertical_overlap_px = 78 # about 15%
    snap_extra_patch_row_threshold_px = 78 # if a patch row would only have h=78px, downscale to remove the patch row
    max_patch_rows = 2 # resize so it fits into two patch rows (considering vertical overlap)
    linearize = True # remove gamma from jpeg encoding, should improve binarization
    normalize_background = False

    # LDTransform
    max_angle_deg: float = 5.0
    min_angle_deg: float = 0.3
    tps_add_corners: bool = True
    tps_alpha: float = 0.5
    legacy_tps_detect: bool = False
