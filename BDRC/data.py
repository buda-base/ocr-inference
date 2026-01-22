"""
Data structures and enumerations for the Tibetan OCR application.

This module contains all the core data types, enums, and dataclasses used
throughout the OCR application for representing OCR data, settings, and
various configuration options.
"""

from dataclasses import dataclass
from enum import Enum
from numpy.typing import NDArray
from pathlib import Path
from typing import Optional
from uuid import UUID


class OpStatus(Enum):
    """Operation status indicators for various OCR operations."""

    SUCCESS = 0
    FAILED = 1


class Encoding(Enum):
    """Text encoding formats for OCR output."""

    UNICODE = 0
    WYLIE = 1


class CharsetEncoder(Enum):
    """Character set encoding methods for OCR models."""

    WYLIE = 0
    STACK = 1


class ExportFormat(Enum):
    """Available export formats for OCR results."""

    TXT = 0
    XML = 1
    JSON = 2


class LineMode(Enum):
    """Line detection modes for OCR processing."""

    LINE = 0
    LAYOUT = 1


class LineMerge(Enum):
    """Methods for merging detected lines."""

    MERGE = 0
    STACK = 1


class LineSorting(Enum):
    """Algorithms for sorting detected lines."""

    THRESHOLD = 0
    PEAKS = 1


class OCRArchitecture(Enum):
    """Supported OCR model architectures."""

    EASTER2 = 0
    CRNN = 1


class TPSMode(Enum):
    """Thin Plate Spline transformation modes for dewarping."""

    GLOBAL = 0
    LOCAL = 1


class Language(Enum):
    """Supported application languages."""

    ENGLISH = 0
    GERMAN = 1
    FRENCH = 2
    TIBETAN = 3
    CHINESE = 4


@dataclass
class ScreenData:
    """Screen dimensions and positioning data for application window."""

    max_width: int
    max_height: int
    start_width: int
    start_height: int
    start_x: int
    start_y: int


@dataclass
class BBox:
    """Bounding box coordinates for rectangular regions."""

    x: int
    y: int
    w: int
    h: int

@dataclass
class RotatedBBox:
    center: tuple[float, float]
    width: float
    height: float
    angle: float
    points: NDArray  # (4,2)

@dataclass
class Line:
    """Detected text line with contour and bounding box information."""

    guid: UUID
    contour: NDArray
    bbox: BBox
    center: tuple[int, int]


@dataclass
class OCRLine:
    """OCR-recognized text line with encoding information."""

    guid: UUID
    text: str
    encoding: str
    ctc_conf: Optional[float] | None
    logits: Optional[list[float]] | None
    lm_scores: Optional[list[float]] | None


@dataclass
class LayoutData:
    """Layout analysis results containing detected regions and predictions."""

    image: NDArray
    rotation: float
    images: list[BBox]
    text_bboxes: list[BBox]
    lines: list[Line]
    captions: list[BBox]
    margins: list[BBox]
    predictions: dict[str, NDArray]


@dataclass
class OCRData:
    """Complete OCR data for a single image including results and metadata."""

    guid: UUID
    image_path: str
    image_name: str
    image: NDArray
    ocr_lines: list[OCRLine] | None
    lines: list[Line] | None
    preview: NDArray | None
    angle: float


@dataclass
class DewarpingResult:
    """Result from dewarping stage."""

    work_img: NDArray
    work_mask: NDArray
    filtered_contours: list
    page_angle: float
    applied: bool
    tps_ratio: Optional[float] = None
    dewarped_img: Optional[NDArray] = None
    dewarped_mask: Optional[NDArray] = None

@dataclass
class LineDetectionConfig:
    """Configuration for line detection model."""

    model_file: str
    patch_size: int


@dataclass
class LayoutDetectionConfig:
    """Configuration for layout detection model."""
    checkpoint: str
    onnx_file: str
    architecture: str
    patch_size: int
    classes: list[str]


@dataclass
class OCRModelConfig:
    """Configuration parameters for OCR model."""

    model_file: str
    architecture: str
    input_width: int
    input_height: int
    input_layer: str
    output_layer: str
    squeeze_channel: bool
    swap_hw: bool
    encoder: CharsetEncoder
    charset: list[str]
    add_blank: bool
    version: str


@dataclass
class LineDataResult:
    """Result container for line detection operations."""

    guid: UUID
    lines: list[Line]


@dataclass
class OCResult:
    """Complete OCR processing result for an image."""

    guid: UUID
    mask: NDArray
    lines: list[Line]
    text: list[OCRLine]
    angle: float


@dataclass
class OCRSample:
    """OCR sample data with metadata for batch processing."""

    cnt: int
    guid: UUID
    name: str
    result: OCResult


@dataclass
class OCRModel:
    """OCR model information and configuration."""

    guid: UUID
    name: str
    path: str
    config: OCRModelConfig


@dataclass
class OCRSettings:
    """User-configurable OCR processing settings."""

    line_mode: LineMode
    line_merge: LineMerge
    line_sorting: LineSorting
    k_factor: float
    bbox_tolerance: float
    dewarping: bool
    merge_lines: bool
    tps_mode: TPSMode
    output_encoding: Encoding


@dataclass
class EvaluationSet:
    distribution: str
    image_paths: list[str]
    label_paths: list[str]
    cer_scores: dict[str, float]


@dataclass
class KenLMConfig:
    kenlm_file: str | Path
    arpa_file: str | Path
    unigrams: list[str]


@dataclass
class ArtifactConfig:
    """Configuration for artifact saving behavior."""

    enabled: bool = True
    granularity: str = "standard"  # "minimal", "standard"
    save_detection: bool = True
    save_dewarping: bool = True
