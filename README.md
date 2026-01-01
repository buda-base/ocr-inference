# ocr-inference

Scripts to run OCR inference

### TODO: optimization
- find good max width / height for image and resize images if larger
- map images with a PIL mode other than RGB, L or 1 (like CMYK, RGBA, etc.) to RGB
- map RGB to grayscale?
- keep the line mask binary if possible (?), or at least avoid unneccessary conversions
- after line mask is generated, it seems calculating the values rotation_angle and tps don't require the image, only the mask, so remove the image from the arguments
- avoid calculating contours twice if rotation is 0? (might not be worth it)
- no rotation calculation after tps dewarp?

### TODO: new API (will help optimization)

##### get_preprocessed_img(img)

return img in well known color space, resized

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

##### get_detection(lines_mask, detect_angle=True, detect_dewarp=True)

returns (contours, angle, input_points, output_points)

##### apply_transform(img, angle, input_points, output_points)

returns modified img