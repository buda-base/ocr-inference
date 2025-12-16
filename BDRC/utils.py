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
from collections.abc import Sequence
from datetime import UTC, datetime
from math import ceil
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import onnxruntime as ort
from huggingface_hub import snapshot_download

from bdrc.data import BBox, LayoutDetectionConfig, Line, LineDetectionConfig, OCRData, OCRModel, OCRModelConfig

# Import functions from specialized modules for backward compatibility
# Import generate_guid from line_detection module for backward compatibility
from bdrc.line_detection import calculate_rotation_angle_from_lines, generate_guid, mask_n_crop, rotate_from_angle
from config import CHARSETENCODER, COLOR_DICT, OCRARCHITECTURE


def show_image(image: npt.NDArray, cmap: str = "", axis: str = "off", fig_x: int = 24, fix_y: int = 13) -> None:
    plt.figure(figsize=(fig_x, fix_y))
    plt.axis(axis)

    if cmap != "":
        plt.imshow(image, cmap=cmap)
    else:
        plt.imshow(image)


def show_overlay(
    image: npt.NDArray,
    mask: npt.NDArray,
    alpha: float = 0.4,
    axis: str = "off",
    fig_x: int = 24,
    fix_y: int = 13,
) -> None:
    plt.figure(figsize=(fig_x, fix_y))
    plt.axis(axis)
    plt.imshow(image)
    plt.imshow(mask, alpha=alpha)


def get_utc_time() -> str:
    """
    Get current UTC time as a formatted string.

    Returns:
        Current UTC time in ISO format (YYYY-MM-DDTHH:MM:SS)
    """
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S")


def download_model(identifier: str) -> str:
    model_path = snapshot_download(
        repo_id=identifier, repo_type="model", local_dir=f"Models/{identifier}", force_download=True
    )

    model_config = Path(model_path) / "model_config.json"
    if not model_config.is_file():
        msg = f"Model config not found: {model_config}"
        raise FileNotFoundError(msg)

    return str(model_config)


def read_line_model_config(config_file: str) -> LineDetectionConfig:
    model_dir = Path(config_file).parent
    with Path(config_file).open(encoding="utf-8") as f:
        json_content = json.load(f)

    onnx_model_file = f"{model_dir}/{json_content['onnx-model']}"
    patch_size = int(json_content["patch_size"])

    return LineDetectionConfig(onnx_model_file, patch_size)


def read_layout_model_config(config_file: str) -> LayoutDetectionConfig:
    model_dir = Path(config_file).parent
    with Path(config_file).open(encoding="utf-8") as f:
        json_content = json.load(f)

    onnx_model_file = f"{model_dir}/{json_content['onnx-model']}"
    patch_size = int(json_content["patch_size"])
    classes = json_content["classes"]

    return LayoutDetectionConfig(onnx_model_file, patch_size, classes)


def get_charset(charset: str | list[str]) -> list[str]:
    if isinstance(charset, str):
        return list(charset)
    return list(charset)


def get_execution_providers() -> list[str]:
    """
    Get available ONNX runtime execution providers.

    Returns:
        List of available execution provider names
    """
    available_providers = ort.get_available_providers()
    logger = logging.getLogger(__name__)
    logger.info("Available ONNX providers: %s", available_providers)
    return available_providers


def get_filename(file_path: str) -> str:
    """
    Extract filename without extension from a file path.

    Args:
        file_path: Full path to the file

    Returns:
        Filename without extension
    """
    name_segments = Path(file_path).name.split(".")[:-1]
    name = "".join(f"{x}." for x in name_segments)
    return name.rstrip(".")


def create_dir(dir_name: str) -> None:
    """
    Create a directory if it doesn't exist.

    Args:
        dir_name: Path of the directory to create
    """
    logger = logging.getLogger(__name__)
    path = Path(dir_name)
    if not path.exists():
        try:
            path.mkdir(parents=True)
            logger.info("Created directory at %s", dir_name)
        except OSError:
            logger.exception("Failed to create directory at: %s", dir_name)


def build_ocr_data(id_val: int, file_path: str, target_width: int | None = None) -> OCRData:
    """
    Build OCR data from a file path.

    Args:
        id_val: Integer to use for generating the UUID identifier
        file_path: Path to the image file
        target_width: Optional width to scale the image to

    Returns:
        OCRData object
    """
    file_name = get_filename(file_path)
    guid = generate_guid(id_val)

    # Load and scale the image
    image = cv2.imread(file_path)
    if image is None:
        msg = f"Failed to load image: {file_path}"
        raise FileNotFoundError(msg)
    if target_width is not None:
        image, _ = resize_to_width(image, target_width)

    return OCRData(
        guid=guid,
        image_path=file_path,
        image_name=file_name,
        image=image,
        ocr_lines=None,
        lines=None,
        preview=None,
        angle=0.0,
    )


def read_theme_file(file_path: str) -> dict | None:
    """
    Load theme configuration from a JSON file.

    Args:
        file_path: Path to the theme configuration file

    Returns:
        Theme configuration dictionary, or None if file doesn't exist
    """
    path = Path(file_path)
    if path.is_file():
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    logger = logging.getLogger(__name__)
    logger.error("Theme File %s does not exist", file_path)
    return None


def import_local_models(model_path: str) -> list[OCRModel]:
    """
    Import all OCR models from a directory.

    Args:
        model_path: Directory path containing OCR model subdirectories

    Returns:
        List of OCRModel instances loaded from the directory
    """
    tick = 1
    ocr_models: list[OCRModel] = []
    logger = logging.getLogger(__name__)
    model_dir = Path(model_path)

    if model_dir.is_dir():
        for sub_dir in model_dir.iterdir():
            if sub_dir.is_dir():
                _config_file = sub_dir / "model_config.json"
                if not _config_file.is_file():
                    logger.warning("ignore %s", sub_dir)
                    tick += 1
                    continue

                _config = read_ocr_model_config(str(_config_file))
                _model = OCRModel(guid=generate_guid(tick), name=sub_dir.name, path=str(sub_dir), config=_config)
                ocr_models.append(_model)
            tick += 1

    return ocr_models


def import_local_model(model_path: str) -> OCRModel | None:
    """
    Import a single OCR model from a directory.

    Args:
        model_path: Directory path containing a single OCR model

    Returns:
        OCRModel instance or None if model cannot be loaded
    """
    model_dir = Path(model_path)
    if model_dir.is_dir():
        _config_file = model_dir / "model_config.json"
        if not _config_file.is_file():
            return None

        _config = read_ocr_model_config(str(_config_file))
        return OCRModel(guid=generate_guid(1), name=model_dir.name, path=model_path, config=_config)

    return None


def read_ocr_model_config(config_file: str) -> OCRModelConfig:
    """
    Load OCR model configuration from a JSON file.

    Args:
        config_file: Path to the model configuration JSON file

    Returns:
        OCRModelConfig instance with loaded parameters
    """
    model_dir = Path(config_file).parent
    with Path(config_file).open(encoding="utf-8") as f:
        json_content = json.load(f)

    onnx_model_file = f"{model_dir}/{json_content['onnx-model']}"
    architecture = json_content["architecture"]
    version = json_content["version"]
    input_width = json_content["input_width"]
    input_height = json_content["input_height"]
    input_layer = json_content["input_layer"]
    output_layer = json_content["output_layer"]
    encoder = json_content["encoder"]
    squeeze_channel_dim = json_content["squeeze_channel_dim"] == "yes"
    swap_hw = json_content["swap_hw"] == "yes"
    characters = json_content["charset"]
    add_blank = json_content["add_blank"] == "yes"

    return OCRModelConfig(
        onnx_model_file,
        OCRARCHITECTURE[architecture],
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


def resize_to_height(image: npt.NDArray, target_height: int) -> tuple[npt.NDArray, float]:
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


def resize_to_width(image: npt.NDArray, target_width: int = 2048) -> tuple[npt.NDArray, float]:
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


def calculate_steps(image: npt.NDArray, patch_size: int = 512) -> tuple[int, int]:
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


def calculate_paddings(image: npt.NDArray, x_steps: int, y_steps: int, patch_size: int = 512) -> tuple[int, int]:
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


def pad_image(image: npt.NDArray, pad_x: int, pad_y: int, pad_value: int = 0) -> npt.NDArray:
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
    return np.pad(
        image,
        pad_width=((0, pad_y), (0, pad_x), (0, 0)),
        mode="constant",
        constant_values=pad_value,
    )


def sigmoid(x: npt.NDArray) -> npt.NDArray:
    """
    Apply sigmoid activation function.

    Args:
        x: Input value or array

    Returns:
        Sigmoid of input (value between 0 and 1)
    """
    return 1 / (1 + np.exp(-x))


def get_text_area(
    image: npt.NDArray, prediction: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray] | tuple[None, None, None]:
    dil_kernel = np.ones((12, 2))
    dil_prediction = cv2.dilate(prediction, kernel=dil_kernel, iterations=10)

    prediction = cv2.resize(prediction, (image.shape[1], image.shape[0]))
    dil_prediction = cv2.resize(dil_prediction, (image.shape[1], image.shape[0]))

    contours, _ = cv2.findContours(dil_prediction, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if len(contours) > 0:
        area_mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

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
    return None, None, None


def get_text_bbox(lines: list[Line]) -> BBox:
    all_bboxes = [x.bbox for x in lines]
    min_x = min(a.x for a in all_bboxes)
    min_y = min(a.y for a in all_bboxes)

    max_w = max(a.w for a in all_bboxes)
    max_h = all_bboxes[-1].y + all_bboxes[-1].h

    return BBox(min_x, min_y, max_w, max_h)


def pol2cart(theta: npt.NDArray, rho: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y


def cart2pol(x: npt.NDArray, y: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho


def rotate_contour(cnt: npt.NDArray, center: tuple[int, int], angle: float) -> npt.NDArray:
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

    cnt_rotated = cnt_norm + [cx, cy]  # noqa: RUF005
    return cnt_rotated.astype(np.int32)


def is_inside_rectangle(point: Sequence[float], rect: list[int]) -> bool:
    x, y = point
    xmin, ymin, xmax, ymax = rect
    return xmin <= x <= xmax and ymin <= y <= ymax


def filter_contours(prediction: npt.NDArray, textarea_contour: npt.NDArray) -> list[npt.NDArray]:
    filtered_contours = []
    x, y, w, h = cv2.boundingRect(textarea_contour)
    line_contours, _ = cv2.findContours(prediction, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in line_contours:
        center, _, _ = cv2.minAreaRect(cnt)
        is_in_area = is_inside_rectangle(center, [x, y, x + w, y + h])

        if is_in_area:
            filtered_contours.append(cnt)

    return filtered_contours


def post_process_prediction(
    image: npt.NDArray, prediction: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, float] | None:
    processed_prediction, text_area, textarea_contour = get_text_area(image, prediction)

    if processed_prediction is not None and text_area is not None and textarea_contour is not None:
        cropped_prediction = mask_n_crop(processed_prediction, text_area)
        angle = calculate_rotation_angle_from_lines(cropped_prediction)

        rotated_image = rotate_from_angle(image, angle)
        rotated_prediction = rotate_from_angle(prediction, angle)

        moments = cv2.moments(textarea_contour)
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        rotated_textarea_contour = rotate_contour(textarea_contour, (cx, cy), angle)

        return rotated_image, rotated_prediction, rotated_textarea_contour, angle
    return None


def generate_line_preview(prediction: npt.NDArray, filtered_contours: list[npt.NDArray]) -> npt.NDArray:
    preview = np.zeros(shape=prediction.shape, dtype=np.uint8)

    for cnt in filtered_contours:
        cv2.drawContours(preview, [cnt], -1, color=(255, 0, 0), thickness=-1)

    return preview


def tile_image(padded_img: npt.NDArray, patch_size: int = 512) -> tuple[list[npt.NDArray], int]:
    x_steps = int(padded_img.shape[1] / patch_size)
    y_steps = int(padded_img.shape[0] / patch_size)
    y_splits = np.split(padded_img, y_steps, axis=0)

    patches = [np.split(x, x_steps, axis=1) for x in y_splits]
    patches = [x for xs in patches for x in xs]

    return patches, y_steps


def stitch_predictions(prediction: npt.NDArray, y_steps: int) -> npt.NDArray:
    pred_y_split = np.split(prediction, y_steps, axis=0)
    x_slices = [np.hstack(list(x)) for x in pred_y_split]
    return np.vstack(x_slices)


def get_paddings(image: npt.NDArray, patch_size: int = 512) -> tuple[int, int]:
    max_x = ceil(image.shape[1] / patch_size) * patch_size
    max_y = ceil(image.shape[0] / patch_size) * patch_size
    pad_x = max_x - image.shape[1]
    pad_y = max_y - image.shape[0]

    return pad_x, pad_y


def preprocess_image(
    image: npt.NDArray,
    patch_size: int = 512,
    clamp_width: int = 4096,
    clamp_height: int = 2048,
    *,
    clamp_size: bool = True,
) -> tuple[npt.NDArray, int, int]:
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

    elif clamp_size and image.shape[0] > image.shape[1] and image.shape[0] > clamp_height:
        image, _ = resize_to_height(image, clamp_height)

    elif image.shape[0] < patch_size:
        image, _ = resize_to_height(image, patch_size)

    pad_x, pad_y = get_paddings(image, patch_size)
    padded_img = pad_image(image, pad_x, pad_y, pad_value=255)

    return padded_img, pad_x, pad_y


def normalize(image: npt.NDArray) -> npt.NDArray:
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


def binarize(img: npt.NDArray, *, adaptive: bool = True, block_size: int = 51, c: int = 13) -> npt.NDArray:
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

    return cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)


def pad_to_width(img: npt.NDArray, target_width: int, target_height: int, padding: str) -> npt.NDArray:
    _, _, channels = img.shape
    tmp_img, _ = resize_to_width(img, target_width)

    height = tmp_img.shape[0]
    middle = (target_height - tmp_img.shape[0]) // 2

    if padding == "white":
        upper_stack = np.ones(shape=(middle, target_width, channels), dtype=np.uint8)
        lower_stack = np.ones(shape=(target_height - height - middle, target_width, channels), dtype=np.uint8)

        upper_stack *= 255
        lower_stack *= 255
    else:
        upper_stack = np.zeros(shape=(middle, target_width, channels), dtype=np.uint8)
        lower_stack = np.zeros(shape=(target_height - height - middle, target_width, channels), dtype=np.uint8)

    return np.vstack([upper_stack, tmp_img, lower_stack])


def pad_to_height(img: npt.NDArray, target_width: int, target_height: int, padding: str) -> npt.NDArray:
    _, _, channels = img.shape
    tmp_img, _ = resize_to_height(img, target_height)

    width = tmp_img.shape[1]
    middle = (target_width - width) // 2

    if padding == "white":
        left_stack = np.ones(shape=(target_height, middle, channels), dtype=np.uint8)
        right_stack = np.ones(shape=(target_height, target_width - width - middle, channels), dtype=np.uint8)

        left_stack *= 255
        right_stack *= 255

    else:
        left_stack = np.zeros(shape=(target_height, middle, channels), dtype=np.uint8)
        right_stack = np.zeros(shape=(target_height, target_width - width - middle, channels), dtype=np.uint8)

    return np.hstack([left_stack, tmp_img, right_stack])


def pad_ocr_line(
    img: npt.NDArray, target_width: int = 3000, target_height: int = 80, padding: str = "black"
) -> npt.NDArray:
    width_ratio = target_width / img.shape[1]
    height_ratio = target_height / img.shape[0]

    if width_ratio < height_ratio:
        out_img = pad_to_width(img, target_width, target_height, padding)

    elif width_ratio > height_ratio:
        out_img = pad_to_height(img, target_width, target_height, padding)
    else:
        out_img = pad_to_width(img, target_width, target_height, padding)

    return cv2.resize(out_img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)


def create_preview_image(
    image: npt.NDArray,
    image_predictions: list | None,
    line_predictions: list | None,
    caption_predictions: list | None,
    margin_predictions: list | None,
    alpha: float = 0.4,
) -> npt.NDArray:
    mask = np.zeros(image.shape, dtype=np.uint8)

    if image_predictions is not None and len(image_predictions) > 0:
        rgb = [int(x) for x in COLOR_DICT["image"].split(",")]
        image_color: tuple[int, int, int] = (rgb[0], rgb[1], rgb[2])

        for idx, _ in enumerate(image_predictions):
            cv2.drawContours(mask, image_predictions, contourIdx=idx, color=image_color, thickness=-1)

    if line_predictions is not None:
        rgb = [int(x) for x in COLOR_DICT["line"].split(",")]
        line_color: tuple[int, int, int] = (rgb[0], rgb[1], rgb[2])

        for idx, _ in enumerate(line_predictions):
            cv2.drawContours(mask, line_predictions, contourIdx=idx, color=line_color, thickness=-1)

    if caption_predictions is not None and len(caption_predictions) > 0:
        rgb = [int(x) for x in COLOR_DICT["caption"].split(",")]
        caption_color: tuple[int, int, int] = (rgb[0], rgb[1], rgb[2])

        for idx, _ in enumerate(caption_predictions):
            cv2.drawContours(mask, caption_predictions, contourIdx=idx, color=caption_color, thickness=-1)

    if margin_predictions is not None and len(margin_predictions) > 0:
        rgb = [int(x) for x in COLOR_DICT["margin"].split(",")]
        margin_color: tuple[int, int, int] = (rgb[0], rgb[1], rgb[2])

        for idx, _ in enumerate(margin_predictions):
            cv2.drawContours(mask, margin_predictions, contourIdx=idx, color=margin_color, thickness=-1)

    cv2.addWeighted(mask, alpha, image, 1 - alpha, 0, image)

    return image
