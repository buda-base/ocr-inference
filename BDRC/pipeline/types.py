
from dataclasses import dataclass
from typing import Optional, Any, Tuple, List

@dataclass
class ImageTask:
    """Input descriptor for the prefetcher.

    Attributes:
      key: S3 object key to GET.
      etag: Expected S3 ETag (used for idempotency/audit).
      size: Size in bytes if known.
      volume_id: Owning volume identifier.
    """
    s3_key: str                 # s3 key
    s3_img_etag: str                # from manifest if available
    size: Optional[int]      # bytes if known

@dataclass
class DecodedFrame:
    """Output of the decode stage.

    frame: Decoded image (e.g., np.ndarray HxW or HxWxC).
    width/height: Dimensions of the decoded frame.
    task: Original ImageTask (for metadata like key/etag).
    """
    task: ImageTask
    frame: Any               # placeholder for np.ndarray
    width: int
    height: int

@dataclass
class Record:
    """Row destined for the Parquet writer.

    Contains image identifiers, geometry outputs (angles/TPS),
    and extracted line information to persist.
    """
    img_file_name: str
    img_s3_etag: str
    resized_w: int
    resized_h: int
    rotation_angle: float
    tps_points: Any
    lines_contours: Any
    nb_lines: int
