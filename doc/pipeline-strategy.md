# Multi-Stage Pipeline Architecture Strategy

## Glossary

| Term             | Definition                                                                                                      |
| ---------------- | --------------------------------------------------------------------------------------------------------------- |
| **Job**          | A batch request to process multiple volumes. One job = N volumes.                                               |
| **Task**         | Processing one volume within a job. Tracks progress through all stages.                                         |
| **Stage**        | A processing step (line_detection, line_extraction, ocr, reduce). Each task goes through 4 stages sequentially. |
| **Worker**       | A process that polls one queue and processes tasks for one stage.                                               |
| **Orchestrator** | A process that advances tasks between stages and handles retries.                                               |
| **Volume**       | A BDRC image group (w_id + i_id) containing multiple page images.                                               |

---

## Overview

Refactor from monolithic worker to multi-stage pipeline with specialized
workers.

**Goals:**

- Cost optimization (CPU work on cheap instances, GPU work on expensive ones)
- Selective re-runs (re-run OCR without re-doing line detection)
- Composability (combine outputs from multiple sources via reducers)
- Simpler I/O (Parquet per volume instead of JSON per image)

---

## Stage Types

### Stage 1: Line Detection (GPU)

- **Input:** Original images from S3
- **Output:** `line_detection.parquet` + rotated images per volume
- **Instance:** GPU (g4dn.xlarge or similar)

**Calls:** `detect_lines()`, `build_lines()`, `apply_dewarping()`

**S3 artifacts:**

```
line_detection/
  rotated_images/
    {image_name}.jpg          # rotated/dewarped work image
  line_detection.parquet
```

**Parquet columns:**

- `image_name`: str
- `rotated_image_path`: str (S3 key to rotated image)
- `rot_mask`: bytes (PNG-encoded, needed by Stage 2 for line sorting)
- `page_angle`: float
- `dewarping_applied`: bool
- `tps_ratio`: float | null
- `filtered_contours`: list[list[list[int]]] (contour points as nested lists)

**Note:** Rotated images and rot_mask are persisted for Stage 2. Contours are
serialized as nested lists (JSON-compatible).

### Stage 2: Line Extraction (CPU)

- **Input:** Rotated images + `line_detection.parquet`
- **Output:** Line images + `line_images.parquet`
- **Instance:** CPU-only (c6i.xlarge or similar)

**Calls:** `extract_lines()` (which calls `build_line_data()`,
`sort_lines_by_threshold2()`, `extract_line_images()`)

**S3 artifacts:**

```
line_extraction/
  lines/
    {image_name}_{line_index}.jpg
  line_images.parquet
```

**Parquet columns:**

- `image_name`: str
- `line_index`: int
- `line_image_path`: str (S3 key to line image)
- `line_guid`: str (UUID for tracking)
- `bbox_x`: int
- `bbox_y`: int
- `bbox_w`: int
- `bbox_h`: int
- `center_x`: int
- `center_y`: int
- `contour`: list[list[int]] (contour points for Line object reconstruction)

**Note:** All fields needed to reconstruct `Line` objects for OCR stage are
stored.

### Stage 3: OCR (GPU)

- **Input:** Line images + `line_images.parquet`
- **Output:** `ocr_results.parquet` per volume
- **Instance:** GPU

**Calls:** `run_text_recognition()`

**S3 artifacts:**

```
ocr/
  ocr_results.parquet
```

**Parquet columns:**

- `image_name`: str
- `line_index`: int
- `line_guid`: str (from line_images.parquet)
- `text`: str
- `encoding`: str ('unicode' or 'wylie')

**Note:** Line objects are reconstructed from `line_images.parquet` data (guid,
bbox, center, contour) before calling `run_text_recognition()`.

### Reducer: Metrics & Finalization (CPU)

- **Input:** All parquet files from previous stages
- **Output:** DB updates only (no S3 artifacts)
- **Instance:** CPU-only

**Calls:** Aggregation logic (no pipeline methods)

**Responsibilities:**

- Compute per-volume metrics from stage parquet files
- Update `volume_results` table in DB with metrics
- Set task `current_stage = 'done'`, `status = 'done'`

---

## S3 Artifact Structure

```
s3://bucket/
  jobs/{job_key}/
    volumes/{w_id}-{i_id}/
      line_detection/
        rotated_images/
          {image_name}.jpg
        line_detection.parquet
      line_extraction/
        lines/
          {image_name}_{line_index}.jpg
        line_images.parquet
      ocr/
        ocr_results.parquet
```

**Key points:**

- Each stage writes to its own prefix
- Stage completion status lives in DB (`tasks.current_stage` + `tasks.status`)
- Workers query DB to check if stage already done (skip if exists)
- Parquet files are the only S3 artifacts (no metadata files)

---

## Schema Changes

**Decision:** Add columns to existing tables and simplify task_status enum.

```sql
-- Simplify task_status enum (remove retryable_failed/terminal_failed distinction)
-- Orchestrator handles retry logic based on attempts < max_attempts
ALTER TYPE task_status RENAME TO task_status_old;
CREATE TYPE task_status AS ENUM ('pending', 'running', 'done', 'failed');
ALTER TABLE tasks ALTER COLUMN status TYPE task_status USING status::text::task_status;
DROP TYPE task_status_old;

-- Tasks table: stage tracking
ALTER TABLE tasks ADD COLUMN current_stage text NOT NULL DEFAULT 'pending';
-- Note: last_error column already exists as jsonb

-- Jobs table: config column already exists as jsonb
```

**Config JSONB structure:**

```python
{
    "k_factor": 2.5,
    "bbox_tolerance": 4.0,
    "merge_lines": False,
    "use_tps": False,
    "encoding": "unicode",
    "line_mode": "line",
}
```

Workers fetch job config at start of processing via `job_id` from message.

**Stage order defined in code, not schema:**

```python
STAGES = ['pending', 'line_detection', 'line_extraction', 'ocr', 'reduce', 'done']
# 'pending' = initial state (bootstrapping)
# 'done' = all stages complete
```

**Task lifecycle:**

1. Task created: `current_stage = 'pending'`, `status = 'pending'`
2. Orchestrator sees "pending pending" → publishes to line_detection queue, sets
   `current_stage = 'line_detection'`, `status = 'running'`
3. Worker completes → sets `status = 'done'`
4. Orchestrator sees "line_detection done" → publishes to line_extraction
   queue...
5. (repeat for each stage)
6. Reducer worker completes → sets `current_stage = 'done'`, `status = 'done'`

**Note:** Reducer is special - it sets both `current_stage = 'done'` and
`status = 'done'` because there's no next stage for orchestrator to advance to.

**Stage failure:**

- Worker fails → `status = 'failed'`, `current_stage` unchanged, `attempts++`
- Orchestrator re-publishes same stage (if `attempts < max_attempts`)
- Terminal failure (`attempts >= max_attempts`) → `status = 'failed'`, stays
  there

**Retry logic:**

- Worker increments `attempts` on failure
- Orchestrator checks `attempts < max_attempts` before re-publishing
- `max_attempts` is per-task (applies across all stages, not per-stage)
- Any failure in a stage = entire stage fails (no partial success)

**Benefits:**

- No new table, minimal schema change
- Flexible: add/remove/rename stages without migrations
- One row per volume per job (not N rows for N stages)
- Stage order lives in code, easy to modify per job type
- Clean bootstrapping via 'pending' pseudo-stage

---

## SQS Strategy

**Decision:** Multiple FIFO queues (one per stage).

- `ocr-line-detection.fifo`
- `ocr-line-extraction.fifo`
- `ocr-ocr.fifo`
- `ocr-reduce.fifo`

```python
# Publishing (orchestrator)
sqs.send_message(
    QueueUrl=get_queue_url(stage),
    MessageBody=json.dumps(task_data),
    MessageGroupId=str(volume_id),  # ensures same volume processed in order
    MessageDeduplicationId=f"{task_id}-{stage}-{attempt}",
)
```

**Worker environment variables:**

```
DATABASE_URL=postgresql://...
SQS_QUEUE_URL=https://sqs.../ocr-line-detection.fifo  # worker's own queue
```

Each worker only needs its own queue URL.

**Orchestrator environment variables:**

```
DATABASE_URL=postgresql://...
SQS_LINE_DETECTION_URL=https://sqs...
SQS_LINE_EXTRACTION_URL=https://sqs...
SQS_OCR_URL=https://sqs...
SQS_REDUCE_URL=https://sqs...
```

Orchestrator needs all queue URLs to publish to each stage.

**Orchestrator CLI:**

```bash
python orchestrator_main.py
```

**Message body fields:**

```python
{
    "task_id": int,
    "job_id": int,
    "volume_id": int,
    "bdrc_w_id": str,
    "bdrc_i_id": str,
    "attempt": int,
    "job_key": str,
}
```

---

## Worker Changes

### Current: Single processor

```
sqs_loop.py → processor.py (does everything)
```

### New: Stage-specific processors

```
sqs_loop.py (generic) → stage_processors/
                          ├── line_detection.py
                          ├── line_extraction.py
                          ├── ocr.py
                          └── reduce.py
```

**Worker stage selection via CLI argument:**

```bash
python sqs_worker_main.py --stage line_detection --model /path/to/model \
    --input-bucket bdrc-images --output-bucket ocr-results
python sqs_worker_main.py --stage ocr --model /path/to/model \
    --input-bucket bdrc-images --output-bucket ocr-results
python sqs_worker_main.py --stage line_extraction \
    --input-bucket bdrc-images --output-bucket ocr-results
python sqs_worker_main.py --stage reduce \
    --input-bucket bdrc-images --output-bucket ocr-results
```

**Required CLI arguments:**

- `--stage`: Which stage this worker handles
- `--input-bucket`: S3 bucket for original images (BDRC source)
- `--output-bucket`: S3 bucket for job artifacts
- `--model`: Path to OCR model (GPU stages only)

**Environment variables:** `DATABASE_URL`, `SQS_QUEUE_URL` (worker's own queue)

**Key insight:** `sqs_loop.py` stays mostly the same - just routes to different
processor based on `--stage` argument.

---

## Current Processor → Stage Mapping

The current `bdrc/pipeline.py:run_ocr_with_artifacts()` does everything in one
call. Here's how it maps to the new stages:

| Current Code                             | New Stage           | GPU?             |
| ---------------------------------------- | ------------------- | ---------------- |
| `pipeline.detect_lines(image)`           | **line_detection**  | Yes              |
| `pipeline.build_lines(image, line_mask)` | **line_detection**  | No (but bundled) |
| `pipeline.apply_dewarping(...)`          | **line_detection**  | No (but bundled) |
| `pipeline.extract_lines(...)`            | **line_extraction** | No               |
| `pipeline.run_text_recognition(...)`     | **ocr**             | Yes              |
| Save results, compute metrics            | **reduce**          | No               |

**Stage 1: line_detection** (GPU)

- Input: Original images from S3
- Calls: `detect_lines()`, `build_lines()`, `apply_dewarping()`
- Output: `line_detection.parquet` with mask, contours, dewarp params

**Stage 2: line_extraction** (CPU)

- Input: Rotated images + `line_detection.parquet`
- Calls: `extract_lines()`
- Output: Line image files + `line_images.parquet`

**Stage 3: ocr** (GPU)

- Input: Line image files + `line_images.parquet`
- Calls: `run_text_recognition()`
- Output: `ocr_results.parquet`

**Stage 4: reduce** (CPU)

- Input: All parquet files
- Computes: Aggregate metrics
- Output: DB updates only (no S3 artifacts)

---

## Parquet Library

**Decision:** Use `pyarrow`

- Most mature and widely used
- Native S3 support via `pyarrow.fs.S3FileSystem`
- Good compression (snappy, zstd)
- Handles nested types (lists, dicts) well

---

## Implementation Phases

### Phase 1: Parquet Output (Low risk)

- Keep current monolithic worker
- Change output from JSON files to single Parquet per volume
- Update `processor.py` to write Parquet
- No schema changes, no new queues

### Phase 2: Stage Tracking (Medium risk)

- Add `current_stage` column to `tasks` table
- Simplify `task_status` enum to `('pending', 'running', 'done', 'failed')`
- Modify `create_job` to set initial stage and store processing config
- Update `job_status` to show current stage
- Still single worker, but tracks stages

### Phase 3: Split Workers (Higher risk)

- Create stage-specific processors
- Create multiple SQS queues (one per stage)
- Create separate worker entry points per stage
- Add orchestrator process

### Phase 4: Reducers (Medium risk)

- Add reducer stage
- Add metric columns to `volume_results` table:
  ```sql
  ALTER TABLE volume_results ADD COLUMN total_images integer;
  ALTER TABLE volume_results ADD COLUMN total_lines integer;
  ALTER TABLE volume_results ADD COLUMN total_characters integer;
  ALTER TABLE volume_results ADD COLUMN processing_duration_ms real;
  ALTER TABLE volume_results ADD COLUMN dewarping_applied_count integer;
  ```
- Reducer reads all Parquet files, computes metrics
- Reducer updates DB with final results
- Workers no longer touch DB (except status updates)

---

## What We Keep From Current Implementation

| Component       | Keep?  | Notes                                  |
| --------------- | ------ | -------------------------------------- |
| `sqs_loop.py`   | Yes    | Parameterize by stage/queue            |
| `client.py`     | Yes    | Support multiple queues                |
| `messages.py`   | Yes    | No changes (stage implicit from queue) |
| `queries.py`    | Modify | Add current_stage queries              |
| `connection.py` | Yes    | No changes                             |
| `create_job.py` | Modify | Set initial stage, store config        |
| `job_status.py` | Modify | Show current stage per task            |
| `processor.py`  | Split  | Into stage-specific processors         |

---

## Decisions

### Stage Dependencies: Centralized Orchestrator

A single long-running `orchestrator_main.py` process handles all stage
transitions:

```python
# orchestrator_main.py
# Orchestrator only advances through these transitions (reducer sets 'done' itself)
TRANSITIONS = [
    ('pending', 'line_detection'),
    ('line_detection', 'line_extraction'),
    ('line_extraction', 'ocr'),
    ('ocr', 'reduce'),
]

async def run_orchestrator():
    while True:
        # Advance tasks through stage transitions
        for prev_stage, next_stage in TRANSITIONS:
            tasks = await get_tasks_ready_for_stage(prev_stage)
            for task in tasks:
                await publish_to_queue(next_stage, task)
                await update_task_stage(task.id, next_stage, status='running')
        
        # Retry failed tasks (if attempts < max_attempts)
        failed_tasks = await get_retriable_failed_tasks()
        for task in failed_tasks:
            await publish_to_queue(task.current_stage, task)
            await update_task_status(task.id, status='running')
        
        # Check for job completion
        await check_job_completion()
        
        await asyncio.sleep(30)
```

**SQL queries:**

```sql
-- Get tasks ready for next stage
-- For 'pending' stage: status = 'pending' (newly created)
-- For other stages: status = 'done' (previous stage completed)
SELECT t.*, v.bdrc_w_id, v.bdrc_i_id, j.job_key
FROM tasks t
JOIN volumes v ON t.volume_id = v.id
JOIN jobs j ON t.job_id = j.id
WHERE t.current_stage = $1  -- prev_stage
  AND t.status = CASE WHEN $1 = 'pending' THEN 'pending' ELSE 'done' END
  AND j.status = 'running';

-- Get failed tasks that can be retried
SELECT t.*, v.bdrc_w_id, v.bdrc_i_id, j.job_key
FROM tasks t
JOIN volumes v ON t.volume_id = v.id
JOIN jobs j ON t.job_id = j.id
WHERE t.status = 'failed'
  AND t.attempts < t.max_attempts
  AND j.status = 'running';
```

**Logic:**

1. Loops through all stage transitions
2. For each transition, finds tasks where `current_stage = prev_stage` and
   `status = 'pending'` (for new tasks) or `status = 'done'` (for completed
   stages)
3. Publishes to appropriate stage queue
4. Updates `current_stage` to next stage and `status = 'running'`
5. Also retries failed tasks if `attempts < max_attempts`
6. Sleeps 30 seconds, repeats

**Benefits:**

- Single place for orchestration logic
- Stage workers stay simple (just poll queue, process, update DB)
- Easy to monitor and debug
- Lightweight process (just DB queries + SQS publishes)

### Worker DB Queries

```sql
-- Worker: Fetch job config at start of processing
SELECT config FROM jobs WHERE id = $1;

-- Worker: Mark task as running (on message receive)
UPDATE tasks 
SET status = 'running', started_at = NOW() 
WHERE id = $1;

-- Worker: Mark stage complete (on success) - for stages 1-3
UPDATE tasks 
SET status = 'done', done_at = NOW() 
WHERE id = $1;

-- Reducer: Mark task complete (sets current_stage = 'done')
UPDATE tasks 
SET status = 'done', current_stage = 'done', done_at = NOW() 
WHERE id = $1;

-- Worker: Mark stage failed (on failure)
UPDATE tasks 
SET status = 'failed', attempts = attempts + 1, last_error = $2::jsonb 
WHERE id = $1;
```

### Job Completion Check

Orchestrator checks for job completion after each loop:

```sql
-- Check if all tasks for a job are done
UPDATE jobs 
SET status = 'completed', finished_at = NOW()
WHERE id IN (
    SELECT j.id FROM jobs j
    WHERE j.status = 'running'
    AND NOT EXISTS (
        SELECT 1 FROM tasks t 
        WHERE t.job_id = j.id 
        AND t.current_stage != 'done'
    )
);

-- Check if any job has terminal failures (all tasks failed or done, but some failed)
UPDATE jobs
SET status = 'failed', finished_at = NOW()
WHERE id IN (
    SELECT j.id FROM jobs j
    WHERE j.status = 'running'
    AND NOT EXISTS (
        SELECT 1 FROM tasks t
        WHERE t.job_id = j.id
        AND t.status NOT IN ('done', 'failed')
    )
    AND EXISTS (
        SELECT 1 FROM tasks t
        WHERE t.job_id = j.id
        AND t.status = 'failed'
        AND t.attempts >= t.max_attempts
    )
);
```

### Partial Re-runs

To re-run just Stage 3 (OCR) for a volume:

1. Reset task: `current_stage = 'line_extraction'`, `status = 'done'`
2. Orchestrator picks it up on next iteration and publishes to queue

No S3 changes needed - worker overwrites existing Parquet.

### Reducer Trigger: Per-Volume

Reducer is just Stage 4 in the pipeline. Runs immediately after Stage 3
completes for each volume.

**Benefits:**

- Same pattern as other stages
- No complex "wait for all volumes" coordination
- Results available incrementally
