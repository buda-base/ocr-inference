"""
Utility functions for image processing, OCR operations, and file management.

This module contains a comprehensive set of utility functions for:
- Image processing and transformation
- Screen and platform detection
- OCR model management
- File operations and directory handling

Note: Line detection/sorting and image dewarping functions have been moved to
separate modules and are re-exported here for backward compatibility.
"""

import json
import logging
import math
import os
import torch
import torch.nn.functional as F

from cv2.typing import Rect
from datetime import datetime
from math import ceil
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import onnxruntime as ort

import segmentation_models_pytorch as sm
import pyarrow as pa
import pyarrow.parquet as pq

from BDRC.data import (
    BBox,
    KenLMConfig,
    Line,
    OCRData,
    OCRModel,
    OCRModelConfig,
    LineDetectionConfig,
    LayoutDetectionConfig,
    RotatedBBox
)

# Import functions from specialized modules for backward compatibility
# Import generate_guid from line_detection module for backward compatibility
from BDRC.line_detection import (
    calculate_rotation_angle_from_lines,
    generate_guid,
    mask_n_crop,
    rotate_from_angle,
)
from Config import (
    CHARSETENCODER,
    COLOR_DICT,
    LINE_DETECTION_SCHEMA,
)

from huggingface_hub import snapshot_download


def show_image(
    image: NDArray, cmap: str = "", axis="off", fig_x: int = 24, fix_y: int = 13
) -> None:
    plt.figure(figsize=(fig_x, fix_y))
    plt.axis(axis)

    if cmap != "":
        plt.imshow(image, cmap=cmap)
    else:
        plt.imshow(image)


def show_overlay(
    image: NDArray,
    mask: NDArray,
    alpha=0.4,
    axis="off",
    fig_x: int = 24,
    fix_y: int = 13,
):
    plt.figure(figsize=(fig_x, fix_y))
    plt.axis(axis)
    plt.imshow(image)
    plt.imshow(mask, alpha=alpha)


def get_utc_time():
    """
    Get current UTC time as a formatted string.

    Returns:
        Current UTC time in ISO format (YYYY-MM-DDTHH:MM:SS)
    """
    utc_time = datetime.now()
    utc_time = utc_time.strftime("%Y-%m-%dT%H:%M:%S")

    return utc_time


def download_model(identifier: str) -> str:
    model_path = snapshot_download(
        repo_id=identifier,
        repo_type="model",
        local_dir=f"Models/{identifier}",
        force_download=True,
    )

    model_path = Path(model_path)
    json_files = list(model_path.glob("*.json"))

    if len(json_files) == 0:
        raise FileNotFoundError(
            f"No JSON config file found in {model_path}"
        )
    
    if len(json_files) > 1:
        raise RuntimeError(
            f"Multiple JSON files found in {model_path}: "
            f"{[p.name for p in json_files]} — cannot decide which is the model config"
        )
    
    assert os.path.isfile(json_files[0])

    return str(json_files[0])


def download_kenlm(identifier: str) -> tuple[str, str]:
    lm_path = snapshot_download(
        repo_id=identifier,
        repo_type="model",
        local_dir=f"Models/{identifier}",
        force_download=True,
    )

    lm_dir = Path(lm_path)

    bin_files = list(lm_dir.glob("*.binary"))
    arpa_files = list(lm_dir.glob("*.arpa"))

    if len(bin_files) == 0:
        raise FileNotFoundError(f"No .bin file found in {lm_dir}")
    if len(arpa_files) == 0:
        raise FileNotFoundError(f"No .arpa file found in {lm_dir}")

    if len(bin_files) > 1:
        raise RuntimeError(
            f"Multiple .bin files found in {lm_dir}: {[p.name for p in bin_files]}"
        )
    if len(arpa_files) > 1:
        raise RuntimeError(
            f"Multiple .arpa files found in {lm_dir}: {[p.name for p in arpa_files]}"
        )

    return str(bin_files[0]), str(arpa_files[0])


def read_line_model_config(config_file: str) -> LineDetectionConfig:
    model_dir = os.path.dirname(config_file)
    file = open(config_file, encoding="utf-8")
    json_content = json.loads(file.read())

    checkpoint = f"{model_dir}/{json_content['checkpoint']}"
    onnx_file = f"{model_dir}/{json_content['onnx-model']}"
    architecture = json_content["architecture"]
    patch_size = int(json_content["patch_size"])
    classes = json_content["classes"]

    config = LineDetectionConfig(
        checkpoint,
        onnx_file,
        architecture,
        patch_size,
        classes)

    return config


def read_layout_model_config(config_file: str) -> LayoutDetectionConfig:
    model_dir = os.path.dirname(config_file)
    file = open(config_file, encoding="utf-8")
    json_content = json.loads(file.read())

    checkpoint = f"{model_dir}/{json_content["checkpoint"]}"
    onnx_model_file = f"{model_dir}/{json_content["onnx-model"]}"

    if "architecture" in json_content:
        architecture = f"{model_dir}/{json_content["architecture"]}"
    else:
        architecture = "deeplabv3"

    architecture = f"{model_dir}/{json_content["architecture"]}"
    patch_size = int(json_content["patch_size"])
    classes = json_content["classes"]

    config = LayoutDetectionConfig(
        checkpoint,
        onnx_model_file,
        architecture,
        patch_size,
        classes)

    return config


def get_charset(charset: str | list[str]) -> list[str]:
    if isinstance(charset, str):
        charset = [x for x in charset]

    elif isinstance(charset, list):
        charset = charset

    return [x for x in charset]


def get_execution_providers() -> list[str]:
    """
    Get available ONNX runtime execution providers.

    Returns:
        List of available execution provider names
    """
    available_providers = ort.get_available_providers()
    print(f"Available ONNX providers: {available_providers}")
    return available_providers


def get_filename(file_path: str) -> str:
    """
    Extract filename without extension from a file path.

    Args:
        file_path: Full path to the file

    Returns:
        Filename without extension
    """
    name_segments = os.path.basename(file_path).split(".")[:-1]
    name = "".join(f"{x}." for x in name_segments)
    return name.rstrip(".")


def create_dir(dir_name: str) -> None:
    """
    Create a directory if it doesn't exist.

    Args:
        dir_name: Path of the directory to create
    """
    if not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name)
            print(f"Created directory at  {dir_name}")
        except IOError as e:
            print(f"Failed to create directory at: {dir_name}, {e}")


def build_ocr_data(id_val, file_path: str, target_width: int = 2048):
    """
    Build OCR data from a file path.

    Args:
        id_val: Either an integer or a UUID to use as the identifier
        file_path: Path to the image file
        target_width: Optional width to scale the image to

    Returns:
        OCRData object
    """
    file_name = get_filename(file_path)

    # Generate GUID if id_val is an integer, otherwise use the provided UUID
    if isinstance(id_val, int):
        guid = generate_guid(id_val)
    else:
        guid = id_val

    # Load and scale the image
    image = cv2.imread(file_path)

    if target_width is not None:
        image, _ = resize_to_width(image, target_width)

    ocr_data = OCRData(
        guid=guid,
        image_path=file_path,
        image_name=file_name,
        image=image,
        ocr_lines=None,
        lines=None,
        preview=None,
        angle=0.0,
    )

    return ocr_data


def read_theme_file(file_path: str) -> dict | None:
    """
    Load theme configuration from a JSON file.

    Args:
        file_path: Path to the theme configuration file

    Returns:
        Theme configuration dictionary, or None if file doesn't exist
    """
    if os.path.isfile(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            content = json.load(f)

        return content
    else:
        logging.error("Theme File %s does not exist", file_path)
        return None


def import_local_model(model_path: str):
    """
    Import a single OCR model from a directory.

    Args:
        model_path: Directory path containing a single OCR model

    Returns:
        OCRModel instance or None if model cannot be loaded
    """
    _model = None
    if os.path.isdir(model_path):
        _config_file = os.path.join(model_path, "model_config.json")
        if not os.path.isfile(_config_file):
            return None

        _config = read_ocr_model_config(_config_file)
        _model = OCRModel(
            guid=generate_guid(1),
            name=Path(model_path).name,
            path=model_path,
            config=_config,
        )

    return _model


def read_ocr_model_config(config_file: str):
    """
    Load OCR model configuration from a JSON file.

    Args:
        config_file: Path to the model configuration JSON file

    Returns:
        OCRModelConfig instance with loaded parameters
    """
    model_dir = os.path.dirname(config_file)
    file = open(config_file, encoding="utf-8")
    json_content = json.loads(file.read())

    onnx_model_file = f"{model_dir}/{json_content['onnx-model']}"
    architecture = json_content["architecture"]
    version = json_content["version"]
    input_width = json_content["input_width"]
    input_height = json_content["input_height"]
    input_layer = json_content["input_layer"]
    output_layer = json_content["output_layer"]
    encoder = json_content["encoder"]
    squeeze_channel_dim = (
        True if json_content["squeeze_channel_dim"] == "yes" else False
    )
    swap_hw = True if json_content["swap_hw"] == "yes" else False
    characters = json_content["charset"]
    add_blank = True if json_content["add_blank"] == "yes" else False

    config = OCRModelConfig(
        onnx_model_file,
        architecture,
        input_width,
        input_height,
        input_layer,
        output_layer,
        squeeze_channel_dim,
        swap_hw,
        encoder=CHARSETENCODER[encoder],
        charset=characters,
        add_blank=add_blank,
        version=version,
    )

    return config

def parse_arpa_unigrams(arpa_path: str | Path) -> list[str] | None:
        """
        Extract unigram symbols from a KenLM ARPA file.
        """
        unigrams = []
        in_1grams = False

        with open(arpa_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if line == r"\1-grams:":
                    in_1grams = True
                    continue

                if in_1grams and line.startswith("\\"):
                    break

                if in_1grams:
                    if not line or line.startswith("#"):
                        continue

                    # Format: <logprob> <token> [<backoff>]
                    parts = line.split()
                    if len(parts) >= 2:
                        token = parts[1]
                        unigrams.append(token)

        if not unigrams:
            print("No valid unigrams found")
            return None

        return unigrams

def get_kenlm_config(model_path: str | Path, arpa_file: str | Path) -> KenLMConfig:
        unigrams = parse_arpa_unigrams(arpa_file)

        return KenLMConfig(
            model_path,
            arpa_file,
            unigrams
        )

def resize_image(image: NDArray, target_width: int, target_height: int) -> NDArray:
    return cv2.resize(
        image, (target_width, target_height),
        interpolation=cv2.INTER_LINEAR,
    )

def resize_image_gpu(image: torch.Tensor, target_width: int, target_height: int) -> torch.Tensor:
    image = image.unsqueeze(0).float()
    image = torch.nn.functional.interpolate(
            image,
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )
    image = image.squeeze(0)

    return image


def resize_to_height(image: NDArray, target_height: int) -> tuple[NDArray, float]:
    """
    Resize image to a specific height while maintaining aspect ratio.

    Args:
        image: Input image array
        target_height: Desired height in pixels

    Returns:
        Tuple of (resized_image, scale_ratio)
    """
    scale_ratio = target_height / image.shape[0]
    image = cv2.resize(
        image,
        (int(image.shape[1] * scale_ratio), target_height),
        interpolation=cv2.INTER_LINEAR,
    )
    return image, scale_ratio


def resize_to_width(image: NDArray, target_width: int = 2048) -> tuple[NDArray, float]:
    """
    Resize image to a specific width while maintaining aspect ratio.

    Args:
        image: Input image array
        target_width: Desired width in pixels (default: 2048)

    Returns:
        Tuple of (resized_image, scale_ratio)
    """
    scale_ratio = target_width / image.shape[1]
    image = cv2.resize(
        image,
        (target_width, int(image.shape[0] * scale_ratio)),
        interpolation=cv2.INTER_LINEAR,
    )
    return image, scale_ratio


def calculate_steps(image: NDArray, patch_size: int = 512) -> tuple[int, int]:
    """
    Calculate number of patches needed to tile an image.

    Args:
        image: Input image array
        patch_size: Size of each square patch (default: 512)

    Returns:
        Tuple of (x_steps, y_steps) for tiling
    """
    x_steps = image.shape[1] / patch_size
    y_steps = image.shape[0] / patch_size

    x_steps = math.ceil(x_steps)
    y_steps = math.ceil(y_steps)

    return x_steps, y_steps


def calculate_paddings(
    image: NDArray, x_steps: int, y_steps: int, patch_size: int = 512
) -> tuple[int, int]:
    """
    Calculate padding needed to make image divisible into patches.

    Args:
        image: Input image array
        x_steps: Number of horizontal patches
        y_steps: Number of vertical patches
        patch_size: Size of each patch

    Returns:
        Tuple of (pad_x, pad_y) padding values
    """
    max_x = x_steps * patch_size
    max_y = y_steps * patch_size
    pad_x = max_x - image.shape[1]
    pad_y = max_y - image.shape[0]

    return pad_x, pad_y


def pad_image(image: NDArray, pad_x: int, pad_y: int, pad_value: int = 0) -> NDArray:
    """
    Add padding to an image.

    Args:
        image: Input image array
        pad_x: Horizontal padding to add
        pad_y: Vertical padding to add
        pad_value: Value to use for padding (default: 0)

    Returns:
        Padded image array
    """
    padded_img = np.pad(
        image,
        pad_width=((0, pad_y), (0, pad_x), (0, 0)),
        mode="constant",
        constant_values=pad_value,
    )

    return padded_img


def sigmoid(x) -> float:
    """
    Apply sigmoid activation function.

    Args:
        x: Input value or array

    Returns:
        Sigmoid of input (value between 0 and 1)
    """
    return 1 / (1 + np.exp(-x))


def get_text_area(
    image: NDArray, prediction: NDArray
) -> tuple[NDArray, BBox] | tuple[None, None, None]:
    dil_kernel = np.ones((12, 2))
    dil_prediction = cv2.dilate(prediction, kernel=dil_kernel, iterations=10)

    prediction = cv2.resize(prediction, (image.shape[1], image.shape[0]))
    dil_prediction = cv2.resize(dil_prediction, (image.shape[1], image.shape[0]))

    contours, _ = cv2.findContours(dil_prediction, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if len(contours) > 0:
        area_mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)

        area_sizes = [cv2.contourArea(x) for x in contours]
        biggest_area = max(area_sizes)
        biggest_idx = area_sizes.index(biggest_area)

        x, y, w, h = cv2.boundingRect(contours[biggest_idx])
        color = (255, 255, 255)

        cv2.rectangle(
            area_mask,
            (x, y),
            (x + w, y + h),
            color,
            -1,
        )
        area_mask = cv2.cvtColor(area_mask, cv2.COLOR_BGR2GRAY)

        return prediction, area_mask, contours[biggest_idx]
    else:
        return None, None, None


def get_text_bbox(lines: list[Line]):
    all_bboxes = [x.bbox for x in lines]
    min_x = min(a.x for a in all_bboxes)
    min_y = min(a.y for a in all_bboxes)

    max_w = max(a.w for a in all_bboxes)
    max_h = all_bboxes[-1].y + all_bboxes[-1].h

    bbox = BBox(min_x, min_y, max_w, max_h)

    return bbox


def pol2cart(theta, rho) -> tuple[float, float]:
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y


def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho


def rotate_contour(cnt, center: tuple[int, int], angle: float):
    cx = center[0]
    cy = center[1]

    cnt_norm = cnt - [cx, cy]

    coordinates = cnt_norm[:, 0, :]
    xs, ys = coordinates[:, 0], coordinates[:, 1]
    thetas, rhos = cart2pol(xs, ys)

    thetas = np.rad2deg(thetas)
    thetas = (thetas + angle) % 360
    thetas = np.deg2rad(thetas)

    xs, ys = pol2cart(thetas, rhos)

    cnt_norm[:, 0, 0] = xs
    cnt_norm[:, 0, 1] = ys

    cnt_rotated = cnt_norm + [cx, cy]
    cnt_rotated = cnt_rotated.astype(np.int32)

    return cnt_rotated


def is_inside_rectangle(point: tuple[float, float], rect: tuple[int, int, int, int]) -> bool:
    x, y = point
    xmin, ymin, xmax, ymax = rect
    return xmin <= x <= xmax and ymin <= y <= ymax


def filter_contours(prediction: NDArray, textarea_contour: NDArray) -> list[NDArray]:
    filtered_contours = []
    x, y, w, h = cv2.boundingRect(textarea_contour)
    line_contours, _ = cv2.findContours(
        prediction, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in line_contours:
        center, _, _ = cv2.minAreaRect(cnt)
        is_in_area = is_inside_rectangle(center, [x, y, x + w, y + h])

        if is_in_area:
            filtered_contours.append(cnt)

    return filtered_contours


def post_process_prediction(image: NDArray, prediction: NDArray):
    prediction, text_area, textarea_contour = get_text_area(image, prediction)

    if prediction is not None:
        cropped_prediction = mask_n_crop(prediction, text_area)
        angle = calculate_rotation_angle_from_lines(cropped_prediction)

        rotated_image = rotate_from_angle(image, angle)
        rotated_prediction = rotate_from_angle(prediction, angle)

        moments = cv2.moments(textarea_contour)
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        rotated_textarea_contour = rotate_contour(textarea_contour, (cx, cy), angle)

        return rotated_image, rotated_prediction, rotated_textarea_contour, angle
    else:
        return None, None, None, None


def generate_line_preview(prediction: NDArray, filtered_contours: list[NDArray]) -> NDArray:
    preview = np.zeros(shape=prediction.shape, dtype=np.uint8)

    for cnt in filtered_contours:
        cv2.drawContours(preview, [cnt], -1, color=(255, 0, 0), thickness=-1)

    return preview


def tile_image(padded_img: NDArray, patch_size: int = 512) -> tuple[list[NDArray], int]:
    x_steps = int(padded_img.shape[1] / patch_size)
    y_steps = int(padded_img.shape[0] / patch_size)
    y_splits = np.split(padded_img, y_steps, axis=0)

    patches = [np.split(x, x_steps, axis=1) for x in y_splits]
    patches = [x for xs in patches for x in xs]

    return patches, y_steps


def stitch_predictions(prediction: NDArray, y_steps: int) -> NDArray:
    pred_y_split = np.split(prediction, y_steps, axis=0)
    x_slices = [np.hstack(x) for x in pred_y_split]
    concat_img = np.vstack(x_slices)

    return concat_img


def get_paddings(image: NDArray, patch_size: int = 512) -> tuple[int, int]:
    max_x = ceil(image.shape[1] / patch_size) * patch_size
    max_y = ceil(image.shape[0] / patch_size) * patch_size
    pad_x = max_x - image.shape[1]
    pad_y = max_y - image.shape[0]

    return pad_x, pad_y


def preprocess_image(
    image: NDArray,
    patch_size: int = 512,
    clamp_width: int = 4096,
    clamp_height: int = 2048,
    clamp_size: bool = True,
) -> tuple[NDArray, int, int]:
    """
    Preprocess image for OCR by resizing and padding to patch-compatible dimensions.

    Some dimension checking and resizing to avoid very large inputs on which the line(s)
    on the resulting tiles could be too big and cause troubles with the current line model.

    Args:
        image: Input image array
        patch_size: Target patch size for tiling
        clamp_width: Maximum allowed width
        clamp_height: Maximum allowed height
        clamp_size: Whether to enforce size limits

    Returns:
        Tuple of (processed_image, pad_x, pad_y)
    """
    if clamp_size and image.shape[1] > image.shape[0] and image.shape[1] > clamp_width:
        image, _ = resize_to_width(image, clamp_width)

    elif (
        clamp_size and image.shape[0] > image.shape[1] and image.shape[0] > clamp_height
    ):
        image, _ = resize_to_height(image, clamp_height)

    elif image.shape[0] < patch_size:
        image, _ = resize_to_height(image, patch_size)

    pad_x, pad_y = get_paddings(image, patch_size)
    padded_img = pad_image(image, pad_x, pad_y, pad_value=255)

    return padded_img, pad_x, pad_y


def normalize(image: NDArray) -> NDArray:
    """
    Normalize image pixel values to range [0, 1].

    Args:
        image: Input image array with values in range [0, 255]

    Returns:
        Normalized image array with values in range [0, 1]
    """
    image = image.astype(np.float32)
    image /= 255.0
    return image


def binarize(
    img: NDArray, adaptive: bool = True, block_size: int = 51, c: int = 13
) -> NDArray:
    line_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if adaptive:
        bw = cv2.adaptiveThreshold(
            line_img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            c,
        )

    else:
        _, bw = cv2.threshold(line_img, 120, 255, cv2.THRESH_BINARY)

    bw = cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)
    return bw


def pad_to_width(
    img: NDArray, target_width: int, target_height: int, padding: str
) -> NDArray:
    _, _, channels = img.shape
    tmp_img, _ = resize_to_width(img, target_width)

    height = tmp_img.shape[0]
    middle = (target_height - tmp_img.shape[0]) // 2

    if padding == "white":
        upper_stack = np.ones(shape=(middle, target_width, channels), dtype=np.uint8)
        lower_stack = np.ones(
            shape=(target_height - height - middle, target_width, channels),
            dtype=np.uint8,
        )

        upper_stack *= 255
        lower_stack *= 255
    else:
        upper_stack = np.zeros(shape=(middle, target_width, channels), dtype=np.uint8)
        lower_stack = np.zeros(
            shape=(target_height - height - middle, target_width, channels),
            dtype=np.uint8,
        )

    out_img = np.vstack([upper_stack, tmp_img, lower_stack])

    return out_img


def pad_to_height(
    img: NDArray, target_width: int, target_height: int, padding: str
) -> NDArray:
    _, _, channels = img.shape
    tmp_img, _ = resize_to_height(img, target_height)

    width = tmp_img.shape[1]
    middle = (target_width - width) // 2

    if padding == "white":
        left_stack = np.ones(shape=(target_height, middle, channels), dtype=np.uint8)
        right_stack = np.ones(
            shape=(target_height, target_width - width - middle, channels),
            dtype=np.uint8,
        )

        left_stack *= 255
        right_stack *= 255

    else:
        left_stack = np.zeros(shape=(target_height, middle, channels), dtype=np.uint8)
        right_stack = np.zeros(
            shape=(target_height, target_width - width - middle, channels),
            dtype=np.uint8,
        )

    out_img = np.hstack([left_stack, tmp_img, right_stack])

    return out_img


def pad_ocr_line(
    img: NDArray,
    target_width: int = 3000,
    target_height: int = 80,
    padding: str = "black",
) -> NDArray:

    width_ratio = target_width / img.shape[1]
    height_ratio = target_height / img.shape[0]

    if width_ratio < height_ratio:
        out_img = pad_to_width(img, target_width, target_height, padding)

    elif width_ratio > height_ratio:
        out_img = pad_to_height(img, target_width, target_height, padding)
    else:
        out_img = pad_to_width(img, target_width, target_height, padding)

    return cv2.resize(
        out_img, (target_width, target_height), interpolation=cv2.INTER_LINEAR
    )


def draw_bbox(image, bbox: BBox, color=(0, 255, 0), thickness=2):
    cv2.rectangle(image, (bbox.x, bbox.y), (bbox.x + bbox.w, bbox.y + bbox.h), color, thickness)
    return image


def draw_rotated_bbox(image, obb: RotatedBBox, color=(0, 0, 255), thickness=2):
    cv2.polylines(
        image,
        [obb.points],
        isClosed=True,
        color=color,
        thickness=thickness
    )
    return image


def create_preview_image(
    image: NDArray,
    image_predictions: Optional[list],
    line_predictions: Optional[list],
    caption_predictions: Optional[list],
    margin_predictions: Optional[list],
    alpha: float = 0.4,
) -> NDArray:
    mask = np.zeros(image.shape, dtype=np.uint8)

    if image_predictions is not None and len(image_predictions) > 0:
        color = tuple([int(x) for x in COLOR_DICT["image"].split(",")])

        for idx, _ in enumerate(image_predictions):
            cv2.drawContours(
                mask, image_predictions, contourIdx=idx, color=color, thickness=-1
            )

    if line_predictions is not None:
        color = tuple([int(x) for x in COLOR_DICT["line"].split(",")])

        for idx, _ in enumerate(line_predictions):
            cv2.drawContours(
                mask, line_predictions, contourIdx=idx, color=color, thickness=-1
            )

    if caption_predictions is not None and len(caption_predictions) > 0:
        color = tuple([int(x) for x in COLOR_DICT["caption"].split(",")])

        for idx, _ in enumerate(caption_predictions):
            cv2.drawContours(
                mask, caption_predictions, contourIdx=idx, color=color, thickness=-1
            )

    if margin_predictions is not None and len(margin_predictions) > 0:
        color = tuple([int(x) for x in COLOR_DICT["margin"].split(",")])

        for idx, _ in enumerate(margin_predictions):
            cv2.drawContours(
                mask, margin_predictions, contourIdx=idx, color=color, thickness=-1
            )

    cv2.addWeighted(mask, alpha, image, 1 - alpha, 0, image)

    return image


### Functions for PyTorch-based inference ###

def resize_clamp(
    img: torch.Tensor, patch_size: int = 512, max_w: int = 4096, max_h: int = 2048
):
    _, H, W = img.shape

    scale_x = 1.0
    scale_y = 1.0

    if W > H and W > max_w:
        scale = max_w / W
    elif H > W and H > max_h:
        scale = max_h / H
    elif H < patch_size:
        scale = patch_size / H
    else:
        return img, scale_x, scale_y

    new_h = int(round(H * scale))
    new_w = int(round(W * scale))

    scale_x = new_w / W
    scale_y = new_h / H

    img = img.unsqueeze(0).float()
    img = torch.nn.functional.interpolate(
        img,
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
    )
    img = img.squeeze(0)

    return img, scale_x, scale_y


def pad_to_multiple(img: torch.Tensor, patch_size=512, value=255):
    _, H, W = img.shape

    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size

    # pad = (left, right, top, bottom)
    img = F.pad(img, (0, pad_w, 0, pad_h), value=value)
    return img, pad_w, pad_h


def tile_timage(img: torch.Tensor, patch_size: int = 512):
    C, H, W = img.shape
    y_steps = H // patch_size
    x_steps = W // patch_size

    tiles = img.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)

    tiles = tiles.permute(1, 2, 0, 3, 4).contiguous()
    tiles = tiles.view(-1, C, patch_size, patch_size)

    return tiles, x_steps, y_steps


def stitch_tiles(
    preds: torch.Tensor,
    x_steps: int,
    y_steps: int,
    patch_size: int = 512,
):
    """
    preds: [N, C, H, W]
    returns: [C, H_full, W_full]
    """
    N, C, H, W = preds.shape
    assert H == patch_size and W == patch_size
    assert N == x_steps * y_steps

    # [N, C, H, W] → [y, x, C, H, W]
    tiles = preds.view(y_steps, x_steps, C, H, W)

    # stitch width
    rows = []
    for y in range(y_steps):
        rows.append(torch.cat(list(tiles[y]), dim=-1))  # concat W

    # stitch height
    full = torch.cat(rows, dim=-2)  # concat H

    return full


def contour_to_cv(contour: list[int, int]):
    """
    contour: list[(x, y)]
    returns: np.ndarray [N, 1, 2] int32
    """
    return np.array(contour, dtype=np.int32).reshape(-1, 1, 2)


def contour_to_original(contour: list[tuple[int, int]], scale_x: float, scale_y: float) -> list[tuple[int, int]]:
    return [
        (
            int(round(x / scale_x)),
            int(round(y / scale_y)),
        )
        for x, y in contour
    ]


def bbox_to_original(bbox: Rect, scale_x: float, scale_y: float) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    return (
        int(round(x / scale_x)),
        int(round(y / scale_y)),
        int(round(w / scale_x)),
        int(round(h / scale_y)),
    )

def get_union_bbox(contours: list[NDArray]):   
    if len(contours) == 0:
        return None, None
    
    all_points = np.vstack(contours)

    x, y, w, h = cv2.boundingRect(all_points)
    bbox = BBox(x, y, w, h)

    # Rotated
    center, (width, height), angle = cv2.minAreaRect(all_points)
    points = cv2.boxPoints((center, (width, height), angle))
    points = points.astype(np.int32)
    cx, cy = center
    rot_bbox = RotatedBBox((float(cx), float(cy)), width, height, angle, points)

    return bbox, rot_bbox


def crop_padding(mask: torch.Tensor, pad_x: int, pad_y: int):
    """
    mask: [C, H, W]
    """
    if pad_y > 0:
        mask = mask[:, :-pad_y, :]
    if pad_x > 0:
        mask = mask[:, :, :-pad_x]
    return mask


def bboxes_to_pyarrow(bboxes):
    return [{"x": x, "y": y, "w": w, "h": h} for (x, y, w, h) in bboxes]


def contours_to_arrow(contours):
    return [[{"x": x, "y": y} for x, y in contour] for contour in contours]


def write_result_parquet(result: dict, out_dir: str | Path):
    os.makedirs(out_dir, exist_ok=True)
    base_name, _ = os.path.splitext(result["image_name"])

    table = pa.Table.from_pylist(
        [
            {
                "image_name": result["image_name"],
                "image_width": result["image_width"],
                "image_height": result["image_height"],
                "num_contours": result["num_contours"],
                "contours": contours_to_arrow(result["contours"]),
                "bboxes": bboxes_to_pyarrow(result["bboxes"]),
            }
        ],
        schema=LINE_DETECTION_SCHEMA,
    )

    out_path = os.path.join(out_dir, f"{base_name}.parquet")

    pq.write_table(table, out_path, compression="zstd")


def multi_image_collate_fn(batch):
    all_tiles = []
    tile_ranges = []
    metas = []

    offset = 0

    for img, meta in batch:
        img, sx, sy = resize_clamp(img)
        img, pad_x, pad_y = pad_to_multiple(img)

        tiles, x_steps, y_steps = tile_timage(img)
        tiles = tiles.float().div_(255.0)

        n_tiles = tiles.shape[0]
        tile_ranges.append((offset, offset + n_tiles))
        all_tiles.append(tiles)

        meta["scale_x"] = sx
        meta["scale_y"] = sy
        meta["pad_x"] = pad_x
        meta["pad_y"] = pad_y
        meta["x_steps"] = x_steps
        meta["y_steps"] = y_steps

        metas.append(meta)
        offset += n_tiles

    all_tiles = torch.cat(all_tiles, dim=0)

    return all_tiles, tile_ranges, metas


def load_model(checkpoint_path: str, num_classes: int, device: str = "cuda"):
    checkpoint = torch.load(checkpoint_path)

    model = sm.DeepLabV3Plus(classes=num_classes).to(device)
    model.load_state_dict(checkpoint["state_dict"])

    """model = models.segmentation.deeplabv3_resnet50(
        weights=None,
        num_classes=2
    )"""
    model.to(device)
    model.eval()
    return model


def infer_batch(
    model: torch.nn.Module,
    all_tiles: list[torch.Tensor],
    tile_ranges: list[tuple[int, int]],
    metas: list[dict],
    class_threshold: float = 0.9,
    device: str = "cuda",
):
    all_tiles = all_tiles.to(device, non_blocking=True)
    preds = model(all_tiles)

    soft = torch.sigmoid(preds)

    for (start, end), meta in zip(tile_ranges, metas):
        preds_img = soft[start:end]

        stitched = stitch_tiles(preds_img, meta["x_steps"], meta["y_steps"])
        stitched = crop_padding(stitched, meta["pad_x"], meta["pad_y"])

        binary = (stitched > class_threshold).to(torch.uint8) * 255
        mask_np = binary.squeeze(0).cpu().numpy()
        
        contours, _ = cv2.findContours(mask_np, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        result = {
            "image_name": meta["image_name"],
            "image_width": meta["orig_shape"][1],
            "image_height": meta["orig_shape"][0],
            "num_contours": len(contours),
            "contours": [
                contour_to_original(
                    [(int(x), int(y)) for [[x, y]] in cnt],
                    meta["scale_x"],
                    meta["scale_y"],
                )
                for cnt in contours
            ],
            "bboxes": [
                bbox_to_original(
                    cv2.boundingRect(cnt),
                    meta["scale_x"],
                    meta["scale_y"],
                )
                for cnt in contours
            ],
        }

        return result


def save_ocr_lines_parquet(ocr_lines, out_path):
    """
    Saves a list of OCRLine to Parquet.
    """

    data = {
        "guid": [],
        "text": [],
        "encoding": [],
        "ctc_conf": [],
        "norm_logp": [],
        "n_beams": [],
        "logits": [],
        "lm_scores": []
    }

    for line in ocr_lines:
        data["guid"].append(str(line.guid))
        data["text"].append(line.text)
        data["encoding"].append(line.encoding)
        data["ctc_conf"].append(float(line.ctc_conf))

        # If you add these fields:
        data["norm_logp"].append(float(getattr(line, "norm_logp", 0.0)))
        data["n_beams"].append(len(line.logits))

        data["logits"].append(line.logits)

        if line.lm_scores is None:
            data["lm_scores"].append(None)
        else:
            data["lm_scores"].append(line.lm_scores)

    table = pa.Table.from_pydict(
        data,
        schema=pa.schema([
            ("guid", pa.string()),
            ("text", pa.string()),
            ("encoding", pa.string()),
            ("ctc_conf", pa.float32()),
            ("norm_logp", pa.float32()),
            ("n_beams", pa.int16()),
            ("logits", pa.list_(pa.float32())),
            ("lm_scores", pa.list_(pa.float32()))
        ])
    )

    pq.write_table(
        table,
        out_path,
        compression="zstd",
        compression_level=7
    )
