import cv2
import numpy as np
import numpy.typing as npt
import onnxruntime as ort
import pyewts
from pyctcdecode.decoder import build_ctcdecoder
from scipy.special import softmax

from bdrc.data import (
    CharsetEncoder,
    DewarpingResult,
    Encoding,
    LayoutDetectionConfig,
    LineDetectionConfig,
    LineExtractionError,
    OCRError,
    OCRLine,
    OCRModelConfig,
)
from bdrc.image_dewarping import apply_global_tps, check_for_tps
from bdrc.line_detection import (
    build_line_data,
    build_raw_line_data,
    extract_line_images,
    filter_line_contours,
    optimize_countour,
    sort_lines_by_threshold2,
)
from bdrc.utils import (
    binarize,
    get_execution_providers,
    normalize,
    pad_to_height,
    pad_to_width,
    preprocess_image,
    sigmoid,
    stitch_predictions,
    tile_image,
)
from config import COLOR_DICT

RGB_NDIM = 3  # RGB images have 3 dimensions (height, width, channels)


class CTCDecoder:
    def __init__(self, charset: str | list[str], *, add_blank: bool) -> None:
        if isinstance(charset, str):
            self.charset = list(charset)

        elif isinstance(charset, list):
            self.charset = charset

        self.ctc_vocab = self.charset.copy()

        if add_blank and " " not in self.ctc_vocab:
            self.ctc_vocab.insert(0, " ")

        self.ctc_decoder = build_ctcdecoder(self.ctc_vocab)

    def encode(self, label: str) -> list[int]:
        return [self.charset.index(x) + 1 for x in label]

    def decode(self, inputs: list[int]) -> str:
        return "".join(self.charset[x - 1] for x in inputs)

    def ctc_decode(self, logits: npt.NDArray) -> str:
        return self.ctc_decoder.decode(logits).replace(" ", "")


class Detection:
    def __init__(self, config: LineDetectionConfig | LayoutDetectionConfig) -> None:
        self.config = config
        self._config_file = config
        self._onnx_model_file = config.model_file
        self._patch_size = config.patch_size
        self._execution_providers = get_execution_providers()
        self._inference = ort.InferenceSession(self._onnx_model_file, providers=self._execution_providers)

    def _preprocess_image(
        self, image: npt.NDArray, patch_size: int = 512
    ) -> tuple[npt.NDArray, npt.NDArray, int, int, int]:
        padded_img, pad_x, pad_y = preprocess_image(image, patch_size)
        tiles, y_steps = tile_image(padded_img, patch_size)
        tiles = [binarize(x) for x in tiles]
        tiles = [normalize(x) for x in tiles]
        tiles = np.array(tiles)

        return padded_img, tiles, y_steps, pad_x, pad_y

    def _crop_prediction(self, image: npt.NDArray, prediction: npt.NDArray, x_pad: int, y_pad: int) -> npt.NDArray:
        x_lim = prediction.shape[1] - x_pad
        y_lim = prediction.shape[0] - y_pad

        prediction = prediction[:y_lim, :x_lim]
        return cv2.resize(prediction, dsize=(image.shape[1], image.shape[0]))

    def _predict(self, image_batch: npt.NDArray) -> npt.NDArray:
        image_batch = np.transpose(image_batch, axes=[0, 3, 1, 2])
        ort_batch = ort.OrtValue.ortvalue_from_numpy(image_batch)
        prediction = self._inference.run_with_ort_values(["output"], {"input": ort_batch})
        return prediction[0].numpy()

    def predict(self, image: npt.NDArray, class_threshold: float = 0.8) -> npt.NDArray:
        raise NotImplementedError


class LineDetection(Detection):
    def __init__(self, config: LineDetectionConfig) -> None:
        super().__init__(config)

    def predict(self, image: npt.NDArray, class_threshold: float = 0.9) -> npt.NDArray:
        _, tiles, y_steps, pad_x, pad_y = self._preprocess_image(image, patch_size=self._patch_size)
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
    BIN_THRESHOLD = 200

    def __init__(self, config: LayoutDetectionConfig, *, debug: bool = False) -> None:
        super().__init__(config)
        self._classes = config.classes
        self._debug = debug

    def _get_contours(self, prediction: npt.NDArray, *, optimize: bool = True, size_tresh: int = 200) -> list:
        prediction = np.where(prediction > self.BIN_THRESHOLD, 255, 0)
        prediction = prediction.astype(np.uint8)

        if np.sum(prediction) > 0:
            contours, _ = cv2.findContours(prediction, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            if optimize:
                contours = [optimize_countour(x) for x in contours]
                contours = [x for x in contours if cv2.contourArea(x) > size_tresh]
            return list(contours)
        return []

    def create_preview_image(
        self,
        image: npt.NDArray,
        prediction: npt.NDArray,
        alpha: float = 0.4,
    ) -> npt.NDArray | None:
        if image is None:
            return None

        image_predictions = self._get_contours(prediction[:, :, 1])
        line_predictions = self._get_contours(prediction[:, :, 2])
        caption_predictions = self._get_contours(prediction[:, :, 3])
        margin_predictions = self._get_contours(prediction[:, :, 4])

        mask = np.zeros(image.shape, dtype=np.uint8)

        if len(image_predictions) > 0:
            color = tuple(int(x) for x in COLOR_DICT["image"].split(","))

            for idx, _ in enumerate(image_predictions):
                cv2.drawContours(mask, image_predictions, contourIdx=idx, color=color, thickness=-1)

        if len(line_predictions) > 0:
            color = tuple(int(x) for x in COLOR_DICT["line"].split(","))

            for idx, _ in enumerate(line_predictions):
                cv2.drawContours(mask, line_predictions, contourIdx=idx, color=color, thickness=-1)

        if len(caption_predictions) > 0:
            color = tuple(int(x) for x in COLOR_DICT["caption"].split(","))

            for idx, _ in enumerate(caption_predictions):
                cv2.drawContours(mask, caption_predictions, contourIdx=idx, color=color, thickness=-1)

        if len(margin_predictions) > 0:
            color = tuple(int(x) for x in COLOR_DICT["margin"].split(","))

            for idx, _ in enumerate(margin_predictions):
                cv2.drawContours(mask, margin_predictions, contourIdx=idx, color=color, thickness=-1)

        cv2.addWeighted(mask, alpha, image, 1 - alpha, 0, image)

        return image

    def predict(self, image: npt.NDArray, class_threshold: float = 0.8) -> npt.NDArray:
        _, tiles, y_steps, pad_x, pad_y = self._preprocess_image(image, patch_size=self._patch_size)
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
    def __init__(self, ocr_config: OCRModelConfig) -> None:
        self.config = ocr_config
        self._onnx_model_file = ocr_config.model_file
        self._input_width = ocr_config.input_width
        self._input_height = ocr_config.input_height
        self._input_layer = ocr_config.input_layer
        self._output_layer = ocr_config.output_layer
        self._characters = ocr_config.charset
        self._squeeze_channel_dim = ocr_config.squeeze_channel
        self._swap_hw = ocr_config.swap_hw
        self._execution_providers = get_execution_providers()
        self.ocr_session = ort.InferenceSession(self._onnx_model_file, providers=self._execution_providers)
        self._add_blank = ocr_config.add_blank
        self.decoder = CTCDecoder(self._characters, add_blank=self._add_blank)

    def _pad_ocr_line(
        self,
        img: npt.NDArray,
        padding: str = "black",
    ) -> npt.NDArray:
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

    def _prepare_ocr_line(self, image: npt.NDArray) -> npt.NDArray:
        line_image = self._pad_ocr_line(image)
        line_image = binarize(line_image)

        if len(line_image.shape) == RGB_NDIM:
            line_image = cv2.cvtColor(line_image, cv2.COLOR_RGB2GRAY)

        line_image = line_image.reshape((1, self._input_height, self._input_width))
        line_image = (line_image / 127.5) - 1.0
        return line_image.astype(np.float32)

    def _pre_pad(self, image: npt.NDArray) -> npt.NDArray:
        """
        Adds a small white patch of size HxH to the left and right of the line
        """
        h, _, c = image.shape
        patch = np.ones(shape=(h, h, c), dtype=np.uint8)
        patch *= 255
        return np.hstack(tup=[patch, image, patch])

    def _predict(self, image_batch: npt.NDArray) -> npt.NDArray:
        image_batch = image_batch.astype(np.float32)
        ort_batch = ort.OrtValue.ortvalue_from_numpy(image_batch)
        ocr_results = self.ocr_session.run_with_ort_values([self._output_layer], {self._input_layer: ort_batch})

        logits = ocr_results[0].numpy()
        return np.squeeze(logits)

    def _decode(self, logits: npt.NDArray) -> str:
        if logits.shape[0] == len(self.decoder.ctc_vocab):
            logits = np.transpose(logits, axes=[1, 0])  # adjust logits to have shape time, vocab

        return self.decoder.ctc_decode(logits)

    def run(self, line_image: npt.NDArray, *, pre_pad: bool = True) -> str:
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
        *,
        use_line_prepadding: bool = False,
    ) -> None:
        self.ready = False
        self.ocr_model_config = ocr_config
        self.line_config = line_config
        self.encoder = ocr_config.encoder
        self.ocr_inference = OCRInference(self.ocr_model_config)
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

    def update_ocr_model(self, config: OCRModelConfig) -> None:
        self.ocr_model_config = config
        self.ocr_inference = OCRInference(config)

    def update_line_detection(self, config: LineDetectionConfig | LayoutDetectionConfig) -> None:
        if isinstance(config, LineDetectionConfig) and isinstance(self.line_config, LayoutDetectionConfig):
            self.line_inference = LineDetection(config)
        elif isinstance(config, LayoutDetectionConfig) and isinstance(self.line_config, LineDetectionConfig):
            self.line_inference = LayoutDetection(config)

        else:
            return

    # ==================== Stage Methods ====================
    # These methods break down the OCR pipeline into discrete stages
    # that can be called individually or composed together.

    def detect_lines(self, image: npt.NDArray) -> npt.NDArray:
        """Stage 1: Run line/layout detection to get line mask.

        Returns:
            line_mask: The detected line mask.

        Raises:
            RuntimeError: If line detection model is not initialized.
        """
        if self.line_inference is None:
            raise RuntimeError("Line detection model is not initialized")

        if isinstance(self.line_config, LineDetectionConfig):
            line_mask = self.line_inference.predict(image)
        else:
            layout_mask = self.line_inference.predict(image)
            line_mask = layout_mask[:, :, self.line_config.classes.index("line")]
        return line_mask

    def build_lines(
        self, image: npt.NDArray, line_mask: npt.NDArray
    ) -> tuple[npt.NDArray, npt.NDArray, list, list, float]:
        """Stage 2: Build and filter line contours from mask.

        Returns:
            (rot_img, rot_mask, raw_contours, filtered_contours, page_angle)

        Raises:
            LineExtractionError: If no lines are detected or all lines are filtered out.
        """
        rot_img, rot_mask, line_contours, page_angle = build_raw_line_data(image, line_mask)
        if len(line_contours) == 0:
            raise LineExtractionError("No lines detected")

        filtered_contours = filter_line_contours(rot_mask, line_contours)
        if len(filtered_contours) == 0:
            raise LineExtractionError("No valid lines after filtering")

        return rot_img, rot_mask, line_contours, filtered_contours, page_angle

    def apply_dewarping(
        self,
        rot_img: npt.NDArray,
        rot_mask: npt.NDArray,
        filtered_contours: list,
        page_angle: float,
        *,
        use_tps: bool = False,
        tps_threshold: float = 0.25,
    ) -> DewarpingResult:
        """Stage 3: Optionally apply TPS dewarping.

        Returns:
            DewarpingResult with dewarping information.
        """
        if not use_tps:
            return DewarpingResult(
                work_img=rot_img,
                work_mask=rot_mask,
                filtered_contours=filtered_contours,
                page_angle=page_angle,
                applied=False,
            )

        ratio, tps_line_data = check_for_tps(rot_img, filtered_contours)
        if ratio <= tps_threshold:
            return DewarpingResult(
                work_img=rot_img,
                work_mask=rot_mask,
                filtered_contours=filtered_contours,
                page_angle=page_angle,
                applied=False,
                tps_ratio=ratio,
            )

        # Apply dewarping
        dewarped_img, dewarped_mask = apply_global_tps(rot_img, rot_mask, tps_line_data)
        if len(dewarped_mask.shape) == RGB_NDIM:
            dewarped_mask = cv2.cvtColor(dewarped_mask, cv2.COLOR_RGB2GRAY)

        # Rebuild line data from dewarped image
        dew_rot_img, dew_rot_mask, line_contours, new_page_angle = build_raw_line_data(dewarped_img, dewarped_mask)
        new_filtered_contours = filter_line_contours(dew_rot_mask, line_contours)

        return DewarpingResult(
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
        work_img: npt.NDArray,
        rot_mask: npt.NDArray,
        filtered_contours: list,
        k_factor: float = 2.5,
        bbox_tolerance: float = 4.0,
        *,
        merge_lines: bool = True,
    ) -> tuple[list, list]:
        """Stage 4: Build line data, sort lines, and extract line images.

        Returns:
            (sorted_lines, line_images)

        Raises:
            LineExtractionError: If no valid line images are extracted.
        """
        line_data = [build_line_data(x) for x in filtered_contours]
        sorted_lines, _ = sort_lines_by_threshold2(rot_mask, line_data, group_lines=merge_lines)
        line_images = extract_line_images(work_img, sorted_lines, k_factor, bbox_tolerance)

        if not line_images:
            raise LineExtractionError("No valid line images extracted")

        return sorted_lines, line_images

    def run_text_recognition(
        self, line_images: list, sorted_lines: list, target_encoding: Encoding = Encoding.UNICODE
    ) -> list[OCRLine]:
        """Stage 5: Run OCR inference on line images.

        Returns:
            ocr_lines: List of recognized OCR lines.
        """
        ocr_lines = []
        for line_img, line_info in zip(line_images, sorted_lines, strict=True):
            if line_img.shape[0] == 0 or line_img.shape[1] == 0:
                continue

            pred = self.ocr_inference.run(line_img, pre_pad=self.use_line_prepadding).strip().replace("§", " ")

            if self.encoder == CharsetEncoder.WYLIE and target_encoding == Encoding.UNICODE:
                pred = self.converter.toUnicode(pred)
            elif self.encoder == CharsetEncoder.STACK and target_encoding == Encoding.WYLIE:
                pred = self.converter.toWylie(pred)

            ocr_lines.append(
                OCRLine(
                    guid=line_info.guid,
                    text=pred,
                    encoding=Encoding.WYLIE if target_encoding == Encoding.WYLIE else Encoding.UNICODE,
                )
            )

        return ocr_lines

    # ==================== Main Pipeline Method ====================

    # TPS Mode is global-only at the moment
    def run_ocr(
        self,
        image: npt.NDArray,
        k_factor: float = 2.5,
        bbox_tolerance: float = 4.0,
        target_encoding: Encoding = Encoding.UNICODE,
        *,
        merge_lines: bool = True,
        use_tps: bool = False,
        tps_threshold: float = 0.25,
    ) -> tuple[npt.NDArray, list, list[OCRLine], float]:
        """Run the full OCR pipeline.

        Returns:
            (rot_mask, sorted_lines, ocr_lines, page_angle)

        Raises:
            OCRError: If the pipeline is not ready or image is None.
            LineExtractionError: If line detection/extraction fails.
        """
        if not self.ready:
            raise OCRError("OCR pipeline not ready")
        if image is None:
            raise OCRError("Input image is None")

        # Stage 1: Line detection
        line_mask = self.detect_lines(image)

        # Stage 2: Build lines
        rot_img, rot_mask, _, filtered_contours, page_angle = self.build_lines(image, line_mask)

        # Stage 3: Dewarping
        dewarp_result = self.apply_dewarping(
            rot_img, rot_mask, filtered_contours, page_angle, use_tps=use_tps, tps_threshold=tps_threshold
        )

        # Stage 4: Extract lines
        sorted_lines, line_images = self.extract_lines(
            dewarp_result.work_img,
            rot_mask,
            dewarp_result.filtered_contours,
            merge_lines=merge_lines,
            k_factor=k_factor,
            bbox_tolerance=bbox_tolerance,
        )

        # Stage 5: OCR inference
        ocr_lines = self.run_text_recognition(line_images, sorted_lines, target_encoding=target_encoding)

        return rot_mask, sorted_lines, ocr_lines, page_angle
