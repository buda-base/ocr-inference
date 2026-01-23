from typing import Union

import os
import cv2
import json
import math
import numpy as np
import torch

from numpy.typing import NDArray

from glob import glob
import onnxruntime as ort
import pyewts

from pyctcdecode import build_ctcdecoder
from pyctcdecode.decoder import OutputBeam

from scipy.special import softmax

from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
from tqdm import tqdm
from typing import Optional

from BDRC.data import (
    CharsetEncoder,
    Encoding,
    EvaluationSet,
    DewarpingResult,
    KenLMConfig,
    LayoutDetectionConfig,
    Line,
    LineDetectionConfig,
    OCRLine,
    OCRModelConfig,
    OpStatus,
)
from BDRC.image_dewarping import apply_global_tps, check_for_tps
from BDRC.label_encoder import WylieEncoder

from BDRC.line_detection import (
    build_line_data,
    build_raw_line_data,
    extract_line_images,
    filter_line_contours,
    optimize_countour,
    sort_lines_by_threshold2,
)
from BDRC.utils import (
    binarize,
    crop_padding,
    get_execution_providers,
    get_filename,
    get_union_bbox,
    load_model,
    multi_image_collate_fn,
    normalize,
    pad_to_height,
    pad_to_width,
    preprocess_image,
    read_ocr_model_config,
    resize_image_gpu,
    sigmoid,
    stitch_predictions,
    stitch_tiles,
    tile_image,
)
from Config import COLOR_DICT




class CTCDecoder:
    def __init__(
        self,
        charset: str | list[str],
        add_blank: bool,
        kenlm_config: KenLMConfig | None,
    ):
        self.blank_sign = "<blk>"
        self.ctc_beam_width = 64

        if isinstance(charset, str):
            self.charset = list(charset)
        else:
            self.charset = charset

        self.ctc_vocab = self.charset.copy()

        if add_blank:
            self.ctc_vocab.insert(0, "<blk>")

        if kenlm_config is not None:
            try:
                self.ctc_decoder = build_ctcdecoder(
                    self.ctc_vocab,
                    kenlm_model_path=str(kenlm_config.kenlm_file),
                    unigrams=kenlm_config.unigrams,
                )
            except Exception as e:
                print(f"KenLM disabled: {e}")
                self.ctc_decoder = build_ctcdecoder(self.ctc_vocab)
        else:
            self.ctc_decoder = build_ctcdecoder(self.ctc_vocab)

    def encode(self, label: str):
        return [self.charset.index(x) + 1 for x in label]

    def decode(self, inputs: list[int]) -> str:
        return "".join(self.charset[x - 1] for x in inputs)

    def ctc_decode(self, logits) -> str:
        return self.ctc_decoder.decode(logits).replace(self.blank_sign, "")

    def ctc_beam_decode(self, logits):
        return self.ctc_decoder.decode_beams(logits)


class Detection:
    def __init__(self, config: LineDetectionConfig | LayoutDetectionConfig):
        self.config = config
        self._config_file = config
        self._onnx_model_file = config.onnx_file
        self._patch_size = config.patch_size
        self._execution_providers = get_execution_providers()
        self._inference = ort.InferenceSession(
            self._onnx_model_file, providers=self._execution_providers
        )

    def _preprocess_image(self, image: NDArray, patch_size: int = 512):
        padded_img, pad_x, pad_y = preprocess_image(image, patch_size)
        tiles, y_steps = tile_image(padded_img, patch_size)
        tiles = [binarize(x) for x in tiles]
        tiles = [normalize(x) for x in tiles]
        tiles = np.array(tiles)

        return padded_img, tiles, y_steps, pad_x, pad_y

    def _crop_prediction(
        self, image: NDArray, prediction: NDArray, x_pad: int, y_pad: int
    ) -> NDArray:
        x_lim = prediction.shape[1] - x_pad
        y_lim = prediction.shape[0] - y_pad

        prediction = prediction[:y_lim, :x_lim]
        prediction = cv2.resize(prediction, dsize=(image.shape[1], image.shape[0]))

        return prediction

    def _predict(self, image_batch: NDArray):
        image_batch = np.transpose(image_batch, axes=[0, 3, 1, 2])
        ort_batch = ort.OrtValue.ortvalue_from_numpy(image_batch)
        prediction = self._inference.run_with_ort_values(
            ["output"], {"input": ort_batch}
        )
        prediction = prediction[0].numpy()

        return prediction

    def predict(self, image: NDArray, class_threshold: float = 0.8) -> NDArray:
        pass


class LineDetection(Detection):
    def __init__(self, config: LineDetectionConfig) -> None:
        super().__init__(config)

    def predict(self, image: NDArray, class_threshold: float = 0.9) -> NDArray:
        _, tiles, y_steps, pad_x, pad_y = self._preprocess_image(
            image, patch_size=self._patch_size
        )
        prediction = self._predict(tiles)
        prediction = np.squeeze(prediction, axis=1)
        prediction = sigmoid(prediction)
        prediction = np.where(prediction > class_threshold, 1.0, 0.0)
        merged_image = stitch_predictions(prediction, y_steps=y_steps)
        merged_image = self._crop_prediction(image, merged_image, pad_x, pad_y)
        merged_image = merged_image.astype(np.uint8)
        merged_image *= 255

        return merged_image


class LayoutDetection(Detection):
    def __init__(self, config: LayoutDetectionConfig, debug: bool = False) -> None:
        super().__init__(config)
        self._classes = config.classes
        self._debug = debug

    def _get_contours(
        self, prediction: NDArray, optimize: bool = True, size_tresh: int = 200
    ) -> list:
        prediction = np.where(prediction > 200, 255, 0)
        prediction = prediction.astype(np.uint8)

        if np.sum(prediction) > 0:
            contours, _ = cv2.findContours(
                prediction, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )

            if optimize:
                contours = [optimize_countour(x) for x in contours]
                contours = [x for x in contours if cv2.contourArea(x) > size_tresh]
            return contours
        else:
            return []

    def create_preview_image(
        self,
        image: NDArray,
        prediction: NDArray,
        alpha: float = 0.4,
    ) -> NDArray | None:

        if image is None:
            return None

        image_predictions = self._get_contours(prediction[:, :, 1])
        line_predictions = self._get_contours(prediction[:, :, 2])
        caption_predictions = self._get_contours(prediction[:, :, 3])
        margin_predictions = self._get_contours(prediction[:, :, 4])

        mask = np.zeros(image.shape, dtype=np.uint8)

        if len(image_predictions) > 0:
            color = tuple([int(x) for x in COLOR_DICT["image"].split(",")])

            for idx, _ in enumerate(image_predictions):
                cv2.drawContours(
                    mask, image_predictions, contourIdx=idx, color=color, thickness=-1
                )

        if len(line_predictions) > 0:
            color = tuple([int(x) for x in COLOR_DICT["line"].split(",")])

            for idx, _ in enumerate(line_predictions):
                cv2.drawContours(
                    mask, line_predictions, contourIdx=idx, color=color, thickness=-1
                )

        if len(caption_predictions) > 0:
            color = tuple([int(x) for x in COLOR_DICT["caption"].split(",")])

            for idx, _ in enumerate(caption_predictions):
                cv2.drawContours(
                    mask, caption_predictions, contourIdx=idx, color=color, thickness=-1
                )

        if len(margin_predictions) > 0:
            color = tuple([int(x) for x in COLOR_DICT["margin"].split(",")])

            for idx, _ in enumerate(margin_predictions):
                cv2.drawContours(
                    mask, margin_predictions, contourIdx=idx, color=color, thickness=-1
                )

        cv2.addWeighted(mask, alpha, image, 1 - alpha, 0, image)

        return image

    def predict(self, image: NDArray, class_threshold: float = 0.8) -> NDArray:
        _, tiles, y_steps, pad_x, pad_y = self._preprocess_image(
            image, patch_size=self._patch_size
        )
        prediction = self._predict(tiles)
        prediction = np.transpose(prediction, axes=[0, 2, 3, 1])
        prediction = softmax(prediction, axis=-1)
        prediction = np.where(prediction > class_threshold, 1.0, 0)
        merged_image = stitch_predictions(prediction, y_steps=y_steps)
        merged_image = self._crop_prediction(image, merged_image, pad_x, pad_y)
        merged_image = merged_image.astype(np.uint8)
        merged_image *= 255

        return merged_image


class OCRInference:
    def __init__(self, ocr_config: OCRModelConfig, kenlm_config: KenLMConfig | None):

        self.config = ocr_config
        self._onnx_model_file = ocr_config.model_file
        self._input_width = ocr_config.input_width
        self._input_height = ocr_config.input_height
        self._input_layer = ocr_config.input_layer
        self._output_layer = ocr_config.output_layer
        self._characters = ocr_config.charset
        self._squeeze_channel_dim = ocr_config.squeeze_channel
        self._swap_hw = ocr_config.swap_hw
        self._add_blank = ocr_config.add_blank

        self._execution_providers = get_execution_providers()
        self.ocr_session = ort.InferenceSession(self._onnx_model_file)

        self.ctc_decoder = CTCDecoder(
            self._characters,
            self._add_blank,
            kenlm_config=None
        )
        
        # KenLM-based CTC-Decoder if KenLM model is provided
        self.ctc_decoder_lm = None

        if kenlm_config is not None:

            self.ctc_decoder_lm = CTCDecoder(
            self._characters,
            self._add_blank,
            kenlm_config
        )
        
    def _pad_ocr_line(
        self,
        img: NDArray,
        padding: str = "black",
    ) -> NDArray:

        width_ratio = self._input_width / img.shape[1]
        height_ratio = self._input_height / img.shape[0]

        if width_ratio < height_ratio:
            out_img = pad_to_width(img, self._input_width, self._input_height, padding)

        elif width_ratio > height_ratio:
            out_img = pad_to_height(img, self._input_width, self._input_height, padding)
        else:
            out_img = pad_to_width(img, self._input_width, self._input_height, padding)

        return cv2.resize(
            out_img,
            (self._input_width, self._input_height),
            interpolation=cv2.INTER_LINEAR,
        )

    def _prepare_ocr_line(self, image: NDArray) -> NDArray:
        line_image = self._pad_ocr_line(image)
        line_image = binarize(line_image)

        if len(line_image.shape) == 3:
            line_image = cv2.cvtColor(line_image, cv2.COLOR_RGB2GRAY)

        line_image = line_image.reshape((1, self._input_height, self._input_width))
        line_image = (line_image / 127.5) - 1.0
        line_image = line_image.astype(np.float32)

        return line_image

    def _pre_pad(self, image: NDArray):
        """
        Adds a small white patch of size HxH to the left and right of the line
        """
        h, _, c = image.shape
        patch = np.ones(shape=(h, h, c), dtype=np.uint8)
        patch *= 255
        out_img = np.hstack(tup=[patch, image, patch])
        return out_img

    def _predict(self, image_batch: NDArray) -> NDArray:
        image_batch = image_batch.astype(np.float32)

        ort_batch = ort.OrtValue.ortvalue_from_numpy(image_batch)
        ocr_results = self.ocr_session.run_with_ort_values(
            [self._output_layer], {self._input_layer: ort_batch}
        )

        logits = ocr_results[0].numpy()
        logits = np.squeeze(logits)

        return logits

    def _decode(self, logits: NDArray, use_lm: bool = False) -> str:
        if logits.shape[0] == len(self.ctc_decoder.ctc_vocab):
            logits = np.transpose(
                logits, axes=[1, 0]
            )  # adjust logits to have shape time, vocab

        if not use_lm:
            return self.ctc_decoder.ctc_decode(logits)
        elif self.ctc_decoder_lm is not None:
                return self.ctc_decoder_lm.ctc_decoder(logits)
        else:
            print("Warning: KenLM-based CTC-Decoder is None! Using default CTC-Decoder")
            return self.ctc_decoder.ctc_decode(logits)

    def _decode_beams(self, logits: NDArray, use_lm: bool = False) -> list[OutputBeam]:
        if logits.shape[0] == len(self.ctc_decoder.ctc_vocab):
            logits = np.transpose(
                logits, axes=[1, 0]
            )  # adjust log_decode_beamsits to have shape time, vocab

        if not use_lm:
            return self.ctc_decoder.ctc_beam_decode(logits)
        elif self.ctc_decoder_lm is not None:
            return self.ctc_decoder_lm.ctc_beam_decode(logits)
        else:
            print("Warning: KenLM-based CTC-Decoder is None! Using default CTC-Decoder")
            return self.ctc_decoder.ctc_beam_decode(logits)

    def run_beam_code(
        self, line_image: NDArray, pre_pad: bool = True
    ) -> list[OutputBeam]:
        if pre_pad:
            line_image = self._pre_pad(line_image)
        line_image = self._prepare_ocr_line(line_image)

        if self._swap_hw:
            line_image = np.transpose(line_image, axes=[0, 2, 1])

        if not self._squeeze_channel_dim:
            line_image = np.expand_dims(line_image, axis=1)

        logits = self._predict(line_image)
        return self._decode_beams(logits)

    def run(self, line_image: NDArray, pre_pad: bool = True, use_lm: bool = False) -> str:

        if pre_pad:
            line_image = self._pre_pad(line_image)
        line_image = self._prepare_ocr_line(line_image)

        if self._swap_hw:
            line_image = np.transpose(line_image, axes=[0, 2, 1])

        if not self._squeeze_channel_dim:
            line_image = np.expand_dims(line_image, axis=1)

        logits = self._predict(line_image)
        return self._decode(logits)


class OCRPipeline:
    """
    Note: The handling of line model vs. layout model is kind of provisional here and totally depends on
    the way you want to run this. You could also pass both configs to the pipeline, run both models and merge
    the (partially) overlapping output before extracting the line images to compensate for the strengths/weaknesses
    of either model. So that is basically up to you.
    """

    def __init__(
        self,
        ocr_config: OCRModelConfig,
        line_config: LineDetectionConfig | LayoutDetectionConfig,
        kenlm_config: KenLMConfig | None,
        use_line_prepadding: bool = False,
    ):
        self.ready = False
        self.ocr_model_config = ocr_config
        self.line_config = line_config
        self.encoder = ocr_config.encoder
        self.ocr_inference = OCRInference(self.ocr_model_config, kenlm_config=kenlm_config)
        self.converter = pyewts.pyewts()
        self.use_line_prepadding = use_line_prepadding

        if isinstance(self.line_config, LineDetectionConfig):
            self.line_inference = LineDetection(self.line_config)
            self.ready = True
        elif isinstance(self.line_config, LayoutDetectionConfig):
            self.line_inference = LayoutDetection(self.line_config)
            self.ready = True
        else:
            self.line_inference = None
            self.ready = False

    def update_ocr_model(self, config: OCRModelConfig, kenlm_config: KenLMConfig | None):
        self.ocr_model_config = config
        self.ocr_inference = OCRInference(config, kenlm_config)

    def update_line_detection(
        self, config: Union[LineDetectionConfig, LayoutDetectionConfig]
    ):
        if isinstance(config, LineDetectionConfig) and isinstance(
            self.line_config, LayoutDetectionConfig
        ):
            self.line_inference = LineDetection(config)
        elif isinstance(config, LayoutDetectionConfig) and isinstance(
            self.line_config, LineDetectionConfig
        ):
            self.line_inference = LayoutDetection(config)

        else:
            return

    # ==================== Stage Methods ====================
    # These methods break down the OCR pipeline into discrete stages
    # that can be called individually or composed together.

    def detect_lines(self, image: NDArray) -> tuple[OpStatus, NDArray | str]:
        """Stage 1: Run line/layout detection to get line mask.

        Returns:
            (OpStatus.SUCCESS, line_mask) or (OpStatus.FAILED, error_message)
        """
        if (
            isinstance(self.line_config, LineDetectionConfig)
            and self.line_inference is not None
        ):
            line_mask = self.line_inference.predict(image)
        elif (
            isinstance(self.line_config, LayoutDetectionConfig)
            and self.line_inference is not None
        ):
            layout_mask = self.line_inference.predict(image)
            line_mask = layout_mask[:, :, self.line_config.classes.index("line")]

        return OpStatus.SUCCESS, line_mask

    def build_lines(
        self, image: NDArray, line_mask: NDArray
    ) -> tuple[OpStatus, tuple[NDArray, NDArray, list, list, float] | str]:
        """Stage 2: Build and filter line contours from mask.

        Returns:
            (OpStatus.SUCCESS, (rot_img, rot_mask, raw_contours, filtered_contours, page_angle))
            or (OpStatus.FAILED, error_message)
        """
        rot_img, rot_mask, line_contours, page_angle = build_raw_line_data(
            image, line_mask
        )
        if len(line_contours) == 0:
            return OpStatus.FAILED, "No lines detected"

        filtered_contours = filter_line_contours(rot_mask, line_contours)
        if len(filtered_contours) == 0:
            return OpStatus.FAILED, "No valid lines after filtering"

        return OpStatus.SUCCESS, (
            rot_img,
            rot_mask,
            line_contours,
            filtered_contours,
            page_angle,
        )

    def apply_dewarping(
        self,
        rot_img: NDArray,
        rot_mask: NDArray,
        filtered_contours: list,
        page_angle: float,
        use_tps: bool = False,
        tps_threshold: float = 0.25,
    ) -> tuple[OpStatus, DewarpingResult | str]:
        """Stage 3: Optionally apply TPS dewarping.

        Returns:
            (OpStatus.SUCCESS, DewarpingResult) or (OpStatus.FAILED, error_message)
        """
        if not use_tps:
            return OpStatus.SUCCESS, DewarpingResult(
                work_img=rot_img,
                work_mask=rot_mask,
                filtered_contours=filtered_contours,
                page_angle=page_angle,
                applied=False,
            )

        ratio, tps_line_data = check_for_tps(rot_img, filtered_contours)
        if ratio <= tps_threshold:
            return OpStatus.SUCCESS, DewarpingResult(
                work_img=rot_img,
                work_mask=rot_mask,
                filtered_contours=filtered_contours,
                page_angle=page_angle,
                applied=False,
                tps_ratio=ratio,
            )

        # Apply dewarping
        dewarped_img, dewarped_mask = apply_global_tps(rot_img, rot_mask, tps_line_data)
        if len(dewarped_mask.shape) == 3:
            dewarped_mask = cv2.cvtColor(dewarped_mask, cv2.COLOR_RGB2GRAY)

        # Rebuild line data from dewarped image
        dew_rot_img, dew_rot_mask, line_contours, new_page_angle = build_raw_line_data(
            dewarped_img, dewarped_mask
        )
        new_filtered_contours = filter_line_contours(dew_rot_mask, line_contours)

        return OpStatus.SUCCESS, DewarpingResult(
            work_img=dew_rot_img,
            work_mask=dew_rot_mask,
            filtered_contours=new_filtered_contours,
            page_angle=new_page_angle,
            applied=True,
            tps_ratio=ratio,
            dewarped_img=dewarped_img,
            dewarped_mask=dewarped_mask,
        )

    def extract_lines(
        self,
        work_img: NDArray,
        rot_mask: NDArray,
        filtered_contours: list,
        merge_lines: bool = True,
        k_factor: float = 2.5,
        bbox_tolerance: float = 4.0,
    ) -> tuple[OpStatus, tuple[list, list] | str]:
        """Stage 4: Build line data, sort lines, and extract line images.

        Returns:
            (OpStatus.SUCCESS, (sorted_lines, line_images)) or (OpStatus.FAILED, error_message)
        """
        line_data = [build_line_data(x) for x in filtered_contours]
        sorted_lines, _ = sort_lines_by_threshold2(
            rot_mask, line_data, group_lines=merge_lines
        )
        line_images = extract_line_images(
            work_img, sorted_lines, k_factor, bbox_tolerance
        )

        if not line_images:
            return OpStatus.FAILED, "No valid line images extracted"

        return OpStatus.SUCCESS, (sorted_lines, line_images)

    def run_text_recognition(
        self,
        line_images: list,
        sorted_lines: list,
        target_encoding: Encoding = Encoding.UNICODE,
    ) -> tuple[OpStatus, list[OCRLine] | str]:
        """Stage 5: Run OCR inference on line images.

        Returns:
            (OpStatus.SUCCESS, ocr_lines) or (OpStatus.FAILED, error_message)
        """
        ocr_lines = []
        for line_img, line_info in zip(line_images, sorted_lines):
            if line_img.shape[0] == 0 or line_img.shape[1] == 0:
                continue

            pred = (
                self.ocr_inference.run(line_img, self.use_line_prepadding)
                .strip()
                .replace("§", " ")
            )

            if (
                self.encoder == CharsetEncoder.WYLIE
                and target_encoding == Encoding.UNICODE
            ):
                pred = self.converter.toUnicode(pred)
            elif (
                self.encoder == CharsetEncoder.STACK
                and target_encoding == Encoding.WYLIE
            ):
                pred = self.converter.toWylie(pred)

            ocr_lines.append(
                OCRLine(
                    guid=line_info.guid,
                    text=pred,
                    encoding=(
                        Encoding.WYLIE.name
                        if target_encoding == Encoding.WYLIE.name
                        else Encoding.UNICODE.name
                    ),
                    ctc_conf=None,
                    logits=None,
                    lm_scores=None,
                )
            )

        return OpStatus.SUCCESS, ocr_lines

    def run_text_recognition_eval(
        self,
        line_images: list,
        sorted_lines: list,
        target_encoding: Encoding = Encoding.UNICODE,
        top_k_beams: int = 10,
    ) -> tuple[OpStatus, list[OCRLine]]:
        """Stage 5: Run OCR inference on line images.

        Returns:
            (OpStatus.SUCCESS, ocr_lines) or (OpStatus.FAILED, error_message)
        """
        ocr_lines = []
        for line_img, line_info in zip(line_images, sorted_lines):
            if line_img.shape[0] == 0 or line_img.shape[1] == 0:
                continue

            beams = self.ocr_inference.run_beam_code(line_img, self.use_line_prepadding)

            if not beams:
                continue

            if len(beams) > top_k_beams:
                beams = beams[:top_k_beams]

            pred = beams[0].text.strip().replace(" ", "")  # beams[0] = top-1 pred
            pred = pred.replace("§", " ")

            if (
                self.encoder == CharsetEncoder.WYLIE
                and target_encoding == Encoding.UNICODE
            ):
                pred = self.converter.toUnicode(pred)
            elif (
                self.encoder == CharsetEncoder.STACK
                and target_encoding == Encoding.WYLIE
            ):
                pred = self.converter.toWylie(pred)

            # length-normalized log-probs of top-1 pred
            L = max(len(beams[0].text), 1)
            norm_logp = beams[0].logit_score / L

            ocr_lines.append(
                OCRLine(
                    guid=line_info.guid,
                    text=pred,
                    encoding=(
                        Encoding.WYLIE.name
                        if target_encoding == Encoding.WYLIE._name_
                        else Encoding.UNICODE.name
                    ),
                    ctc_conf=float(math.exp(norm_logp)),
                    logits=[float(x.logit_score) for x in beams],
                    lm_scores=None,
                )
            )

        return OpStatus.SUCCESS, ocr_lines

    # ==================== Main Pipeline Method ====================

    # TODO: Generate specific meaningful error codes that can be returned inbetween the steps
    # TPS Mode is global-only at the moment
    def run_ocr(
        self,
        image: NDArray,
        k_factor: float = 2.5,
        bbox_tolerance: float = 4.0,
        merge_lines: bool = True,
        use_tps: bool = False,
        tps_threshold: float = 0.25,
        target_encoding: Encoding = Encoding.UNICODE,
        eval_mode: bool = False,
    ) -> tuple[OpStatus, list[OCRLine] | list[Line] | str]:
        try:
            if not self.ready:
                return OpStatus.FAILED, "OCR pipeline not ready"
            if image is None:
                return OpStatus.FAILED, "Input image is None"

            # Stage 1: Line detection
            try:
                status, result = self.detect_lines(image)
                if status == OpStatus.FAILED:
                    return status, result
                line_mask = result
            except Exception as e:
                return OpStatus.FAILED, f"Line detection failed: {str(e)}"

            # Stage 2: Build lines
            try:
                status, result = self.build_lines(image, line_mask)
                if status == OpStatus.FAILED:
                    return status, result
                rot_img, rot_mask, _, filtered_contours, page_angle = result
            except Exception as e:
                return OpStatus.FAILED, f"Line data building failed: {str(e)}"

            # Stage 3: Dewarping
            try:
                status, result = self.apply_dewarping(
                    rot_img,
                    rot_mask,
                    filtered_contours,
                    page_angle,
                    use_tps=use_tps,
                    tps_threshold=tps_threshold,
                )
                if status == OpStatus.FAILED:
                    return status, result
                dewarp_result = result
            except Exception as e:
                return OpStatus.FAILED, f"Line processing failed: {str(e)}"

            # Stage 4: Extract lines
            try:
                status, result = self.extract_lines(
                    dewarp_result.work_img,
                    rot_mask,
                    dewarp_result.filtered_contours,
                    merge_lines=merge_lines,
                    k_factor=k_factor,
                    bbox_tolerance=bbox_tolerance,
                )
                if status == OpStatus.FAILED:
                    return status, result
                sorted_lines, line_images = result
            except Exception as e:
                return OpStatus.FAILED, f"Line extraction failed: {str(e)}"

            # Stage 5: OCR inference
            try:

                if eval_mode:
                    status, result = self.run_text_recognition_eval(
                        line_images, sorted_lines, target_encoding=target_encoding
                    )
                    if status == OpStatus.FAILED:
                        return status, result
                else:
                    status, result = self.run_text_recognition(
                        line_images, sorted_lines, target_encoding=target_encoding
                    )
                    if status == OpStatus.FAILED:
                        return status, result
                ocr_lines = result
            except Exception as e:
                return OpStatus.FAILED, f"OCR processing failed: {str(e)}"

            return OpStatus.SUCCESS, [
                rot_mask,
                sorted_lines,
                ocr_lines,
                float(page_angle),
            ]

        except Exception as e:
            return OpStatus.FAILED, f"OCR pipeline failed: {str(e)}"


class ImageInferenceDataset(Dataset):
    def __init__(self, root_dir: str, mode: str = "rgb"):
        self._mode = mode
        self.paths = sorted(
            p
            for p in glob(os.path.join(root_dir, "*"))
            if p.lower().endswith((".jpg", ".png", ".jpeg"))
        )

    def __len__(self):
        return len(self.paths)
    
    def get_item(self, idx):
         return self.__getitem__(idx)

    def __getitem__(self, idx):
        # TODO: check if images are .tif/.tiff and use alternative loading since torchvision doesn't support tif/tiff
        
        if self._mode == "rgb":
            img = read_image(self.paths[idx], ImageReadMode.RGB)
        else:
            img = read_image(self.paths[idx])

        meta = {
            "image_name": os.path.basename(self.paths[idx]),
            "orig_shape": (img.shape[1], img.shape[2]),  # (H, W)
        }

        return img, meta


class ModernBookFormatLayoutDetection:
    def __init__(self, config: LayoutDetectionConfig):
        self.config = config
        self.classes = config.classes
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = load_model(self.config.checkpoint, len(self.classes), self.device)

    
    def extract_json_data(self, meta: dict, predicton: torch.Tensor, filter_classes: list[str] | None, output_dir: str):
        np_predicton = predicton.cpu().detach().numpy()

        found_classes = dict()
        
        if filter_classes is not None and len(filter_classes) > 0:
            for idx, class_name in enumerate(self.classes):
                if class_name in filter_classes:
                    bbox = self.post_process_sample(np_predicton, idx)
                    if bbox is None:
                        continue

                    found_classes[class_name] = bbox
        else:
            for idx, class_name in enumerate(self.classes):
                bbox = self.post_process_sample(np_predicton, idx)

                if bbox is None:
                    continue

                found_classes[class_name] = bbox
            
        file_name = get_filename(meta["image_name"])
        self.save_to_json(file_name, output_dir, found_classes)
    
    def save_to_json(self, image_name: str, output_dir: str, json_record: dict):
        out_file = f"{output_dir}/{image_name}.json"

        with open(out_file, "w", encoding="UTF-8") as f:
            json.dump(json_record, f, ensure_ascii=False, indent=1)

    def post_process_sample(self, prediction: NDArray, class_index: int) -> dict | None:
        class_map = prediction[class_index, :, :]
        contours, _ = cv2.findContours(class_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        if len(contours) == 0:
            return None

        bbox, _ = get_union_bbox(contours)

        if bbox is None:
            return None
        
        return {
            "bbox": {
                "x": bbox.x,
                "y": bbox.y,
                "w": bbox.w,
                "h": bbox.h
            }
        }


    def run(self, directory: str, output_dir: str, filter_classes: list[str] | None, batch_size: int = 4, num_workers: int = 4, class_threshold: float = 0.8):
        
        if filter_classes is not None and len(filter_classes) > 0:
            for f_class in filter_classes:
                if f_class not in self.classes:
                    raise ValueError(f"ERROR: provided filter classes: {filter_classes} are not part of the model's classes!")

        os.makedirs(output_dir, exist_ok=True)

        dataset = ImageInferenceDataset(directory, mode="rgb")

        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=multi_image_collate_fn)

        if self.device == "cuda":
            torch.cuda.synchronize()

        with torch.no_grad():
            for all_tiles, tile_ranges, metas in tqdm(data_loader, total=len(data_loader)):
                all_tiles = all_tiles.to("cuda", non_blocking=True)
                preds = self.model(all_tiles)
   
                soft = torch.softmax(preds, dim=1)

                for (start, end), meta in zip(tile_ranges, metas):
                    pred = soft[start:end]
                    pred = stitch_tiles(pred, meta["x_steps"], meta["y_steps"])
                    pred = crop_padding(pred, meta["pad_x"], meta["pad_y"])
                    
                    orig_height, original_width = meta["orig_shape"]
                    pred = resize_image_gpu(pred, original_width, orig_height)
                    pred = (pred > class_threshold).to(torch.uint8) * 255
                    
                    self.extract_json_data(meta, pred, filter_classes, output_dir)
                
        if self.device == "cuda":
            torch.cuda.synchronize()


class OCREvaluator:
    """
    A simple wrapper class around some inference functions ro run ocr inference and CER calculation
    based on line-image and line-label inputs
    """

    def __init__(
        self,
        config_path: str,
        cer_scorer,
        kenlm_config: KenLMConfig | None = None,
        label_encoding: Encoding = Encoding.UNICODE,
    ):
        assert os.path.isfile(config_path)

        self._config_file = config_path
        self._cer_scorer = cer_scorer
        self._kenlm_config = kenlm_config
        self._label_encoding = label_encoding  # mostly Unicode anyways

        try:
            self._model_config = read_ocr_model_config(self._config_file)
        except BaseException as e:
            print(
                f"Failed to load ocr model config from file: {self._config_file}, {e}"
            )

        # TODO: add StackEncoder
        self._label_encoder = WylieEncoder(self._model_config.charset)

        try:
            self._inference = OCRInference(self._model_config, self._kenlm_config)
        except BaseException as e:
            print(f"Failed to create OCRInference instance: {e}")

    def get_architecture(self) -> str:
        return self._model_config.architecture

    def evaluate(self, image_path: str, label_path: str) -> float:
        img = cv2.imread(image_path)
        label = self._label_encoder.read_label(label_path)

        prediction = self._inference.run(img)
        cer_score = self._cer_scorer.compute(
            predictions=[prediction], references=[label]
        )

        return cer_score

    def evaluate_distribution(
        self, folder_name: str, image_paths: list[str], label_paths: list[str]
    ) -> EvaluationSet:

        cer_scores = dict()

        # TODO: add batched inference
        for image_path, label_path in tqdm(
            zip(image_paths, label_paths), total=len(image_paths)
        ):
            img = cv2.imread(image_path)
            img_name = get_filename(image_path)
            img = binarize(img)
            label = self._label_encoder.read_label(label_path)

            prediction = self._inference.run(img)
            cer_score = self._cer_scorer.compute(
                predictions=[prediction], references=[label]
            )

            cer_scores[img_name] = float(cer_score)

        return EvaluationSet(
            folder_name,
            image_paths,
            label_paths,
            cer_scores
        )
