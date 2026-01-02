# OCR Batch Processing CLI Guide

This document covers the command-line interface for managing OCR batch
processing jobs.

## Glossary

| Term         | Description                                                                                                               |
| ------------ | ------------------------------------------------------------------------------------------------------------------------- |
| **Job**      | A batch processing request containing one or more volumes to OCR. Jobs have a unique key and track overall progress.      |
| **Task**     | A single unit of work within a job, representing one volume to process. Each task is processed independently by a worker. |
| **Volume**   | A BDRC volume identified by a W_id (work) and I_id (image group), e.g., `W22308-I1KG9611`.                                |
| **Job Type** | A configuration template that defines OCR processing parameters (model, encoding, line mode, etc.).                       |
| **Worker**   | A process that polls SQS for tasks and performs OCR processing.                                                           |

## Architecture Overview

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  CLI        │────▶│  PostgreSQL │◀────│  Worker     │
│  Commands   │     │  Database   │     │  Process    │
└─────────────┘     └─────────────┘     └─────────────┘
       │                                       │
       │            ┌─────────────┐            │
       └───────────▶│  SQS Queue  │◀───────────┘
                    └─────────────┘
```

1. **CLI** creates jobs and publishes tasks to SQS
2. **Workers** poll SQS, process tasks, update database
3. **Database** is the source of truth for job/task status

## Environment Setup

Create a `.env` file in the project root:

```bash
DATABASE_URL=postgresql://user:password@host:5432/ocr_batch
SQS_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/123456789/ocr-tasks.fifo
AWS_DEFAULT_REGION=us-east-1
INPUT_BUCKET=archive.tbrc.org
```

## CLI Commands

### List Job Types

View available job type configurations.

```bash
python -m batch.cli.list_job_types
```

**Output:**

```
ID   Name                           Model                Encoding   Line Mode
--------------------------------------------------------------------------------
1    woodblock-stacks-unicode       Woodblock-Stacks     unicode    line
2    woodblock-stacks-wylie         Woodblock-Stacks     wylie      line
```

**Use case:** Before creating a job, check which job types are available and
their configurations.

---

### Create Job

Create a new batch job with one or more volumes.

```bash
python -m batch.cli.create_job --type-id <ID> --volumes <VOLUMES> [--job-key <KEY>]
python -m batch.cli.create_job --type-id <ID> --volume-file <FILE> [--job-key <KEY>]
```

**Arguments:**

| Argument        | Required | Description                                   |
| --------------- | -------- | --------------------------------------------- |
| `--type-id`     | Yes      | Job type ID (from `list_job_types`)           |
| `--volumes`     | *        | Comma-separated volumes in `W_id-I_id` format |
| `--volume-file` | *        | Path to file with volumes (one per line)      |
| `--job-key`     | No       | Custom job key (auto-generated if omitted)    |

\* Either `--volumes` or `--volume-file` is required (mutually exclusive).

**Volume file format:**

```
# Comments start with #
W22308-I1KG9611
W22308-I1KG9612
W22308-I1KG9613  # inline comments also work
```

**Examples:**

```bash
# Single volume
python -m batch.cli.create_job --type-id 1 --volumes "W22308-I1KG9611"

# Multiple volumes (inline)
python -m batch.cli.create_job --type-id 1 --volumes "W22308-I1KG9611,W22308-I1KG9612,W22308-I1KG9613"

# From file
python -m batch.cli.create_job --type-id 1 --volume-file volumes.txt

# With custom job key
python -m batch.cli.create_job --type-id 1 --volume-file volumes.txt --job-key "my-test-job"
```

**Output:**

```
Created job: J20241231_a1b2c3d4 (id=42)
Created 3 tasks for job J20241231_a1b2c3d4
Publishing 3 tasks to SQS
Job 42 status changed to running
Job ID: 42
```

**What happens:**

1. Creates a job record in the database
2. Creates task records for each volume
3. Publishes task messages to SQS queue
4. Updates job status to "running"

---

### Job Status

View status of jobs and their tasks.

```bash
python -m batch.cli.job_status [--job-id <ID>] [--job-key <KEY>] [--include-status <STATUS>] [--exclude-status <STATUSES>] [--limit <N>]
```

**Arguments:**

| Argument           | Required | Description                                                      |
| ------------------ | -------- | ---------------------------------------------------------------- |
| `--job-id`         | No       | Show specific job by ID                                          |
| `--job-key`        | No       | Show specific job by key                                         |
| `--include-status` | No       | Filter by status (created, running, completed, failed, canceled) |
| `--exclude-status` | No       | Exclude statuses (default: canceled, failed)                     |
| `--all`            | No       | Show all jobs without exclusions                                 |
| `--limit`          | No       | Max jobs to show (default: 20)                                   |

**Examples:**

```bash
# List all active jobs (excludes canceled/failed by default)
python -m batch.cli.job_status

# Show specific job
python -m batch.cli.job_status --job-id 42

# Show by job key
python -m batch.cli.job_status --job-key "J20241231_a1b2c3d4"

# Show only running jobs
python -m batch.cli.job_status --include-status running

# Show all jobs including canceled/failed
python -m batch.cli.job_status --all

# Show failed jobs only
python -m batch.cli.job_status --include-status failed
```

**Output:**

```
============================================================
Job: J20241231_a1b2c3d4 (id=42)
============================================================
Status:     running
Type ID:    1
Created:    2024-12-31 10:00:00+00:00
Started:    2024-12-31 10:00:01+00:00

Progress: 15/50 done, 2 failed
  Pending: 30, Running: 3
  Retryable: 2, Terminal: 0

Throughput: 12.5 tasks/hour, ETA: 2.8 hours

Recent Errors:
  Task 123 (W22308-I1KG9611) [retryable]: Connection timeout
  Task 124 (W22308-I1KG9612) [retryable]: S3 download failed
```

---

### Cancel Job

Cancel a running or pending job.

```bash
python -m batch.cli.cancel_job --job-id <ID>
```

**Arguments:**

| Argument   | Required | Description      |
| ---------- | -------- | ---------------- |
| `--job-id` | Yes      | Job ID to cancel |

**Example:**

```bash
python -m batch.cli.cancel_job --job-id 42
```

**Output:**

```
Job 42 canceled
```

**What happens:**

1. Sets job status to "canceled"
2. Workers will skip tasks for canceled jobs
3. In-progress tasks will complete but no new tasks will start

---

## Starting a Worker

Workers are separate processes that poll SQS and process tasks.

```bash
python sqs_worker_main.py \
  --model-dir /path/to/ocr/models \
  [--output-bucket bec.bdrc.io] \
  [--visibility-timeout 600]
```

**Arguments:**

| Argument               | Required | Description                                       |
| ---------------------- | -------- | ------------------------------------------------- |
| `--model-dir`          | Yes      | Directory containing OCR model folders            |
| `--output-bucket`      | No       | S3 bucket for output files (default: bec.bdrc.io) |
| `--visibility-timeout` | No       | SQS visibility timeout in seconds (default: 600)  |

**Output path format:**

Results are written to: `{job_type_name}/{w_id}-{i_id}-{version_name}/`

For example: `woodblock-stacks-unicode/W22308-I1KG9611-a1b2c3/`

**Example:**

```bash
python sqs_worker_main.py \
  --model-dir /home/ocr/models
```

**How it works:**

1. Worker starts and connects to database
2. Polls SQS for task messages
3. For each task:
   - Fetches job type configuration from database
   - Loads appropriate OCR model (cached per job type)
   - Downloads images from S3
   - Runs OCR pipeline
   - Uploads results to S3
   - Updates task status in database
4. Gracefully shuts down on SIGINT/SIGTERM

---

## Job Lifecycle

Jobs can be canceled from `created`, `running`, or `failed` states.

| Status      | Description                                     |
| ----------- | ----------------------------------------------- |
| `created`   | Job created but tasks not yet published         |
| `running`   | Tasks published to SQS, processing in progress  |
| `completed` | All tasks finished successfully                 |
| `failed`    | Job failed (creation error or all tasks failed) |
| `canceled`  | Job was manually canceled                       |

---

## Task Lifecycle

| Status             | Description                                 |
| ------------------ | ------------------------------------------- |
| `pending`          | Task created, waiting to be picked up       |
| `running`          | Task currently being processed by a worker  |
| `done`             | Task completed successfully                 |
| `retryable_failed` | Task failed but can be retried              |
| `terminal_failed`  | Task failed and exceeded max retry attempts |

---

## Common Workflows

### Process a Single Volume

```bash
# 1. Check available job types
python -m batch.cli.list_job_types

# 2. Create job
python -m batch.cli.create_job --type-id 1 --volumes "W22308-I1KG9611"

# 3. Monitor progress
python -m batch.cli.job_status --job-id 42
```

### Batch Process Multiple Volumes

```bash
# Create job with multiple volumes
python -m batch.cli.create_job --type-id 1 \
  --volumes "W22308-I1KG9611,W22308-I1KG9612,W22308-I1KG9613,W22308-I1KG9614"

# Monitor
python -m batch.cli.job_status --job-id 42
```

### Cancel and Restart a Job

```bash
# Cancel problematic job
python -m batch.cli.cancel_job --job-id 42

# Create new job with same volumes
python -m batch.cli.create_job --type-id 1 --volumes "W22308-I1KG9611"
```

### Monitor All Running Jobs

```bash
# Show all running jobs
python -m batch.cli.job_status --status running

# Show all jobs including failed
python -m batch.cli.job_status --exclude ""
```

---

## Troubleshooting

### Job stuck in "running" with no progress

1. Check if workers are running
2. Check SQS queue for messages
3. Check worker logs for errors

### Tasks failing repeatedly

1. Check job status for error messages:
   ```bash
   python -m batch.cli.job_status --job-id 42
   ```
2. Common causes:
   - Invalid volume ID
   - S3 access issues
   - Model not found in worker's model directory

### Worker not picking up tasks

1. Verify `SQS_QUEUE_URL` matches the queue used by CLI
2. Check AWS credentials
3. Verify database connectivity

---

## Database Schema

For reference, the main tables are:

- **job_types** - OCR configuration templates
- **jobs** - Batch job records
- **tasks** - Individual volume processing tasks
- **volumes** - BDRC volume references
- **task_metrics** - Processing metrics per task
- **page_metrics** - Per-page OCR metrics

See `batch/db/schema.sql` for full schema definition.
