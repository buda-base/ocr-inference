# ocr-inference

Scripts to run OCR inference

### TODO: optimization for line detections

##### a bit of data

- will run on EC2 GPU instances (multiple possibilities)
- onnx model is 22M parameters, 85MB
- input shape: N, 3, 512, 512   FLOAT32

##### easy
- (more for error handling) map images with a PIL mode other than RGB, L or 1 (like CMYK, RGBA, etc.) to grayscale
- keep the line mask binary if possible (?), or at least avoid unneccessary conversions
- after line mask is generated, it seems calculating the values rotation_angle and tps don't require the image, only the mask, so remove the image from the arguments
- define minimum angle threshold to trigger the rotation path (0.5°?)
- define minimum points difference threshold to trigger the tps path (?)
- compute filtered contours once and use it to calculate angle (currently non-filtered contours are used but this seems not intentional?), keep them if rotation not triggered
- no rotation calculation after tps dewarp?
- add performance indicators to measure time spent in the different phases (CPU, GPU, etc.) for each image in a volume, or better thread idle time, esp. GPU thread
- parallelize CPU work (esp. i/o)

##### less easy
- most important: insteach of batching per image, have a "GPU" thread handling fixed-size (image tile) batches (8? 16? 32?)
- optimize the opencv conversion of tiles to reduce overhead
- reuse buffers?
- onnx optimizer?
- use FP16 instead of FP32?

### TODO: new API for line detections (will help optimization)

##### get_preprocessed_img(img)

return img in grayscale or binary, resized

##### get_lines_mask(img)

runs model and returns binary line mask

##### get_line_contours(lines_mask, img_shape, min_area=MIN_CONTOUR_AREA, min_w_ratio=0.01, min_height=MIN_LINE_HEIGHT)

returns filtered line contours (maybe height should also be proportional to the image height?)

##### get_tps_data(line_contours)

returns (input_points, output_points)

##### apply_tps(img, input_points, output_points)

returns modified img

##### get_rotation_angle(line_contours, max_angle=5.0, img_shape)

returns angle (float)

##### apply_rotation(img, angle)

returns modified img

##### get_contours_and_params(lines_mask, detect_angle=True, detect_dewarp=True)

returns (contours, angle, input_points, output_points)

##### apply_transform(img, angle, input_points, output_points)

returns modified img