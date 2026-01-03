from dataclasses import dataclass
from typing import Optional, Any, Tuple, List

@dataclass
class ImageTask:
    """Input descriptor for the prefetcher.
    """
    s3_key: str
    img_filename: str

@dataclass
class FetchedBytes:
    """Input descriptor for the decoder.
    """
    img_filename: str
    s3_etag: str
    file_bytes: bytes

@dataclass
class DecodedFrame:
    """Output of the decode stage and transform stage
    """
    img_filename: str
    s3_etag: str
    frame: Any # grayscale H, W, uint8
    is_binary: bool # if the value space is {0, 255}
    first_pass: bool # if it's a first pass for an image
    rotation_angle: Optional[float] # if in the second pass, value of the rotation angle in degrees that was applied after first pass, null if first pass
    tps_data: Optional[Any] # (input_pts, output_pts, alpha), if in the second pass, tps data that was applied after first pass, null if first pass

@dataclass
class InferredFrame:
    """Output of the inference stage.
    """
    img_filename: str
    s3_etag: str
    frame: Any
    is_binary: bool
    first_pass: bool
    rotation_angle: Optional[float]
    tps_data: Optional[Any]
    line_mask: Any # result of inference, H, W, uint8, binary {0, 255}, same H, W as frame

@dataclass
class Record:
    """input for the Parquet writer, output of the transform stage
    """
    img_filename: str
    s3_etag: str
    resized_w: int # from frame
    resized_h: int
    rotation_angle: float
    tps_data: Any
    contours: Any
    nb_contours: int
    contours_bboxes: Any
