---
id: line_detection_pechas_v1
---

### line detection pechas v1

This is the application of the first line detection model to all the pecha format images.

##### Artefacts

image_lines.parquet with the following columns:
- `img_file_name` (string)
- `img_s3_etag` (string)
- `resized_w` (int, width of the resized image)
- `resized_h` (int)
- `rotation_angle` (double)
- `tps_points` (json, list with 2 values: input_points and output_points (themselves lists of floats or ints), or null of not tps needed)
- `nb_lines` (int)
- `lines_contours` (json, list of list of points?)