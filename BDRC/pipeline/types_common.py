from dataclasses import dataclass
from typing import Any, Literal, Optional, Union, Tuple, List

# --- Sentinel ---------------------------------------------------------------

@dataclass(frozen=True)
class EndOfStream:
    """
    Explicit end-of-stream marker for multi-lane pipelines.
    """
    stream: Literal["prefetched", "decoded", "gpu_pass_1", "transformed_pass_1", "gpu_pass_2", "transformed_pass_2", "record"]
    producer: Optional[str] = None


# --- Core tasks / payloads --------------------------------------------------


@dataclass(frozen=True)
class ImageTask:
    """Input descriptor for the prefetcher.
    """
    s3_key: str
    img_filename: str

@dataclass(frozen=True)
class FetchedBytes:
    """Input descriptor for the decoder.
    """
    task: ImageTask
    s3_etag: str
    file_bytes: bytes

@dataclass(frozen=True)
class DecodedFrame:
    """Output of the decode stage and transform stage
    """
    task: ImageTask
    s3_etag: str
    frame: Any # grayscale H, W, uint8
    orig_h: int
    orig_w: int
    is_binary: bool # if the value space is {0, 255}
    first_pass: bool # if it's a first pass for an image
    rotation_angle: Optional[float] # if in the second pass, value of the rotation angle in degrees that was applied after first pass, null if first pass
    tps_data: Optional[Any] # (input_pts, output_pts, alpha), if in the second pass, tps data that was applied after first pass, null if first pass

@dataclass(frozen=True)
class InferredFrame:
    """Output of the inference stage.
    """
    task: ImageTask
    s3_etag: str
    frame: Any
    orig_h: int
    orig_w: int
    is_binary: bool
    first_pass: bool
    rotation_angle: Optional[float]
    tps_data: Optional[Any]
    line_mask: Any # result of inference, H, W, uint8, binary {0, 255}, same H, W as frame

@dataclass(frozen=True)
class Record:
    """input for the Parquet writer, output of the transform stage
    """
    task: ImageTask
    s3_etag: str
    rotation_angle: float
    tps_data: Any # should be scaled to original image dimension
    contours: Any # NDArray of (x,y) points, contours of line segments (not final merged lines), scaled to original image dimensions
    nb_contours: int
    contours_bboxes: Any # bboxes (x, y, w, h) of the contours, scaled to original image dimensions

# --- Error envelope ---------------------------------------------------------

@dataclass(frozen=True)
class PipelineError:
    """Error message that can flow through queues."""
    stage: Literal["Prefetcher", "Decoder", "LDGpuBatcher", "LDPostProcessor", "S3ParquetWriter"]
    task: ImageTask
    s3_etag: Optional[str]
    error_type: str
    message: str
    traceback: Optional[str] = None
    retryable: bool = False
    attempt: int = 1


# --- Queue message unions ---------------------------------------------------

FetchedBytesMsg = Union[FetchedBytes, PipelineError, EndOfStream]
DecodedFrameMsg = Union[DecodedFrame, PipelineError, EndOfStream]
InferredFrameMsg = Union[InferredFrame, PipelineError, EndOfStream]
RecordMsg = Union[Record, PipelineError, EndOfStream]