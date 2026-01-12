
from dataclasses import dataclass
from typing import Dict, Literal, Optional

Precision = Literal["fp32", "fp16", "bf16", "auto"]

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
    inflight_per_worker: int = 32       # per-worker S3 GET concurrency or local concurrency
    s3_get_timeout_s: int = 60

    # Queues (bounded)
    max_q_prefetcher_to_decoder: int = 64
    max_q_decoder_to_tilebatcher: int = 32
    max_q_tilebatcher_to_inference: int = 2  # Very short! Each batch can be ~200MB+ on GPU
    max_q_gpu_pass_1_to_post_processor: int = 32
    max_q_post_processor_to_tilebatcher: int = 16
    max_q_gpu_pass_2_to_post_processor: int = 16
    max_q_post_processor_to_writer: int = 64


    # CPU decode threads
    decode_threads: int = 8
    tile_workers: int = 8  # ThreadPoolExecutor workers for parallel tiling (increase if CPU-bound)
    parallel_tiling: bool = True  # Enable parallel tiling in TileBatcher

    # GPU batching
    use_gpu: bool = True
    compile_model: bool = False
    precision: Precision = "fp16" # bf16, fp16, fp32 or auto
    # Optional cap for the internal tile pool to limit peak memory under backpressure.
    # 0 disables throttling.
    batch_size: int = 8  # Number of images per batch (8 images × ~8 tiles = ~64 tiles = ~200MB)
    batch_timeout_ms: int = 25
    cuda_streams: int = 2
    binarize_block_size: int = 31
    binarize_c: int = 15
    batch_type: str = "tiles" # "tiles" for batching tiles (images batch mode removed)
    class_threshold: float = 0.85
    gpu_reinit_on_error = False
    gpu_reinit_on_oom = True
    reprocess_budget: int = 3  # Priority weight for reprocess lane (higher = more priority)
    controller_poll_ms: int = 5  # Polling timeout for post-processor

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
    patch_horizontal_overlap_px = 0
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
    add_corners: bool = True

    # Artefact writer
    max_error_message_len = 128
    flush_every = 4096  # Flush Parquet buffer every N records
    
    # Debug
    debug_mode: bool = False
    debug_folder: Optional[str] = None  # Local folder path for debug output
    debug_images: Optional[set[str]] = None  # Set of image filenames to debug (None = all images)
    
    # GPU profiling
    enable_pytorch_profiler: bool = False  # Enable PyTorch profiler for GPU timing analysis
    profiler_trace_output: Optional[str] = None  # Output path for profiler trace (default: pytorch_trace.json)
    detailed_inference_timing: bool = True  # Log H2D/forward/stitch/D2H breakdown for each batch
    
    def __post_init__(self):
        """Validate configuration values."""

        # GPU settings
        if self.use_gpu:
            try:
                import torch
                if not torch.cuda.is_available():
                    raise ValueError("GPU requested but CUDA is not available")
            except ImportError:
                pass  # torch may not be available at config time
        
        # Overlap validations
        if not (0 <= self.patch_vertical_overlap_px < self.patch_size):
            raise ValueError(
                f"patch_vertical_overlap_px must be in [0, patch_size), "
                f"got {self.patch_vertical_overlap_px} (patch_size={self.patch_size})"
            )
        if not (0 <= self.patch_horizontal_overlap_px < self.patch_size):
            raise ValueError(
                f"patch_horizontal_overlap_px must be in [0, patch_size), "
                f"got {self.patch_horizontal_overlap_px} (patch_size={self.patch_size})"
            )
        
        # Timeout validations
        if self.batch_timeout_ms <= 0:
            raise ValueError(f"batch_timeout_ms must be > 0, got {self.batch_timeout_ms}")
        if self.s3_get_timeout_s <= 0:
            raise ValueError(f"s3_get_timeout_s must be > 0, got {self.s3_get_timeout_s}")
        
        # Batch type validation
        if self.batch_type != "tiles":
            raise ValueError(f"batch_type must be 'tiles', got '{self.batch_type}'")