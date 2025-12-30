# Google Vision OCR Pipeline – File & Manifest Conventions

This document describes the file, manifest, and output conventions used by the
BDRC OCR pipeline based on Google Vision API.

The system operates **strictly at the volume level** (BDRC `W_id` + `I_id`).
Batches are an internal execution detail and never mix volumes.

## 1. Terminology

| Term | Meaning |
|----|----|
| `W_id` | BDRC Work ID |
| `I_id` | BDRC Volume ID (image group), globally unique |
| Image | A single scanned page |
| Volume | All images under one `I_id` |
| Lane | Vision API path (`images` or `files`) |
| Batch | One Vision API async request (internal) |
| Bundle | Final output unit for downstream |

## 2. Source Images (S3 → GCS)

### Staged GCS layout

The S3 structure is preserved verbatim:

```
gs://bec-ocr-in/Works/{md5_2}/{W_id}/images/{W_id}-{I_id}/...
```

## 3. Image Identity

Each image is identified by:

| Field | Description |
|----|----|
| `W_id` | Work ID |
| `I_id` | Volume ID |
| `img_fname` | Image file name (string, as-is) |
| `img_sha256` | SHA-256 checksum of image bytes (**authoritative**) |
| `source_path` | Full S3/GCS path |

Checksum ensures correctness when images are replaced upstream.

## 4. Manifests

### 4.1 Volume Inventory Manifest (JSONL)

One file per volume.

**Path**

```
gs://ocr-in/manifests/{run_id}/volumes/{W_id}-{I_id}/inventory.jsonl
````

**One line per image**

```json
{
  "I_id": "I12345",
  "W_id": "W22084",
  "img_fname": "I123450001.tif",
  "source_gcs_uri": "gs://ocr-in/Works/60/W22084/images/W22084-I12345/I123450001.tif",
  "media_type": "image/tiff",
  "lane": "files",
  "width": 2000,
  "height": 3000,
  "img_size_bytes": 512044,
  "img_sha256": "abc123..."
}
````

This inventory may be built using `dimensions.json`.

### 4.2 Batch Manifest (internal)

Batches never mix volumes.

**Path**

```
gs://ocr-in/manifests/{run_id}/batches/{W_id}-{I_id}/{lane}/{batch_id}.json
```

```json
{
  "run_id": "ocr-20251229-101530Z-v1",
  "I_id": "I12345",
  "W_id": "W22084",
  "lane": "files",
  "batch_id": "files-I12345-0003",
  "items_jsonl": "gs://ocr-in/manifests/.../inventory.jsonl",
  "image_indices": [400, 401, 402],
  "vision_output_prefix": "gs://ocr-out/vision-json/run_id=.../I_id=I12345/lane=files/batch_id=files-I12345-0003/",
  "ocr_profile": {
    "name": "weekly_v1",
    "schema_version": 1,
    "feature": {
      "type": "DOCUMENT_TEXT_DETECTION",
      "model": "builtin/weekly"
    }
  }
}
```

## 5. Vision Raw Outputs (Canonical)

Written directly by Google Vision API.

```
gs://ocr-out/vision-json/
  {run_id}/
    {W_id}-{I_id}/
      {lane}/
        {batch_id}/
          output-0001.json
          output-0002.json
```

These files are **immutable** and preserved for audit/debug.

## 6. Volume Bundles (Downstream Interface)

### Definition

A **volume bundle** contains OCR results for all the images of a single volume.

### Layout

```
gs://ocr-volumes/
  {run_id}/
    {W_id}-{I_id}/
      bundle.parquet
      raw.jsonl.gz
      index.json
      _SUCCESS
```

This layout is mirrored to S3 if required.

### 6.1 `bundle.parquet`

One row per image.

Columns:

* `run_id`
* `W_id`
* `I_id`
* `row_in_bundle`
* `img_fname`
* `img_sha256`
* `img_size_bytes`
* `source_gcs_uri`
* `lane`
* `status` (`OK | ERROR | SKIPPED`)
* `error_code`
* `error_message`
* `text` (plain text output of the OCR, given in the Google Vision json output)
* `confidence_index` (aggregate for the whole image, given in the Google Vision json output, implementation defined)
* `detected_languages` (array, given in the Google Vision json output, can be encoded as a json string)
* `ocr_profile_name`

### 6.2 `raw.jsonl.gz`

One JSON object per image, same order as parquet.

```json
{
  "row_in_bundle": 12,
  "I_id": "I12345",
  "W_id": "W22084",
  "img_fname": "I123450001.tif",
  "image_sha256": "abc123...",
  "vision_response": { ... original Vision per-image response ... }
}
```

This preserves original Vision output at image granularity.

### 6.3 `index.json`

Lightweight control file.

```json
{
  "run_id": "ocr-20251229-101530Z-v1",
  "I_id": "I12345",
  "W_id": "W22084",
  "count": 387,
  "parquet": "bundle.parquet",
  "raw": "raw.jsonl.gz",
  "schema_version": 1,
  "ocr_profile": { ...same as batch... },
}
```

## 7. Idempotency & Retries

* Bundles are committed by writing `_SUCCESS`
* If `_SUCCESS` exists, the bundle is immutable
* Reprocessing a volume produces new `run_id`
* Checksums prevent silent reuse of stale images

## 8. Guarantees

* No batch mixes volumes
* No bundle mixes volumes
* Every image is traceable to:
  * original S3 path
  * checksum
  * Vision raw response
