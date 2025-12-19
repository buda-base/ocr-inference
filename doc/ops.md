## Operation deployment

The goal of this code is to run different types of batch operations (primarily OCR) on the image archive, store the results on s3 and some metrics in a database.

### The image archive

The image archive contains about 25,000,000 scans of Tibetan texts with two main types of images: binary tiffs of about 40kB (about 30% of the archive), full color jpegs of about 500kB. The total is somewhere around 6TB of data.

It is stored on an s3 bucket. It is organized by collection and volume. Collection IDs start with `W` and we note them `{w_id}`, volume IDs start with `I` and we note them `{i_id}`. There are about 70,000 volumes in the archive. We call volume id the combination `{w_id}-{i-id}`.

### The operations

Two examples of operations are:
- running an OCR model locally
- running the Google OCR API
- cleaning up results of local model
- gathering metrics on the results of an OCR

Operations follow a fan-out (run multiple OCR systems) fan-in (combine them into the best version) pattern.

### The database

The database will contain:
- information about each image (i_id, file name, width, height, mode, etc.)
- metrics about the results of operations at the volume level (not the image level), for instance confidence index, perplexity index, errors, etc.
- possibly some information about ongoing operations

### Vocabulary

- dataset: the full set of images in S3
- job: one run of one operation over some set of volumes (e.g., “OCR with model M on volumes X,Y,Z”)
- task: one volume within a job (e.g., “job J + volume V”)
- worker: an ephemeral process/EC2 instance that executes tasks
- shard: a small file containing many tasks (e.g., 50–200 volumes per shard)
- manifest: the collection of shard files describing what to process
- checkpoint / done marker: a small file written when a task (or shard) finishes
- reducer job: a job that consumes outputs from multiple OCR jobs and creates new outputs (ex: selects the “best OCR”)

### General ops infrastructure

##### Step 1: sharded manifests

In a first iteration, the jobs will be defined by sharded manifests on s3:

```
s3://bec.bdrc.io/manifests/
  jobs/{job_id}/
    job.json                      # job definition: op type, model, params, input dataset, created_at
    shards/
      shard-00000.jsonl            # each line: { "volume_id": "WXXX-IXXX" }
      shard-00001.jsonl
    shard_done/
      shard-00000.done.json        # written by worker when entire shard completed
```

##### Future steps

In a second iteration, the system will use AWS SQS to handle task distribution.

##### Output file structure

The output for each task is in:

```
s3://bec.bdrc.io/output/
  jobs/{job_id}/
    volumes/{volume_id}/
      artifacts/                   # OCR outputs for that volume (layout you prefer)
      metrics.json                 # optional: per-volume metrics computed by worker
      _done.json                   # completion marker for this volume for this job
      _error.json                  # optional: last failure details if terminal failure
```

### Step by step example: “run OCR on volumes X, Y, Z”

##### job creation (one command)

```
create_job --op ocr --model model_A --volumes X,Y,Z
```

- internally creates job id `J123`
- writes to S3
   * `manifests/jobs/J123/job.json` (job definition)
   * `manifests/jobs/J123/shards/shard-00000.jsonl` with 3 lines (X/Y/Z)
- writes to Postgres
   * jobs row: (J123, op=ocr, model=model_A, status=created)
   * tasks rows (one per volume):
      - (J123, X, status=pending, attempts=0)
      - same for Y, Z
   * shards_lease

##### compute start

```
launch_workers --job J123 --count 10 --instance-type ...
```

Starts the instances.

##### Worker loop

Each worker generates a UUID on startup and periodically heartbeats to Postgres. It then loops:

1. Claim a shard

A tiny Postgres “shard lease” table so only one worker claims a shard at a time.

2. Process the shard lines

For each task in the shard JSONL:

- read volume information (W, I) from the jsonl
- updates the task in the db to status = running
- download image list from s3 (dimensions.json)
- download images in batch to avoid sequential small queries to s3, possibly with local disk cache
- run OCR on all images in the volume

Write outputs under:

output/jobs/J123/volumes/{volume_id}/artifacts/...

Write `output/jobs/J123/volumes/{volume_id}/_done.json` when complete, including summary counts, duration, model/version, maybe a checksum of the input manifest.

3. Update Postgres per volume

- on success, tasks.status = done, done_at, stats (counts, durations), output_prefix
- on failure, increment attempts, set status = retryable_failed or terminal_failed, store last_error

4. Shard completion

When the worker finishes all volumes in the shard, writes write: `manifests/jobs/J123/shard_done/shard-00000.done.json`.

Update jobs.progress counters in Postgres.

Shard done means that no more attempts will be made on the tasks in the shard.

5. Human checks progress (one command / dashboard query)

Command

```
job_status J123
```

Reads from Postgres

counts by status (pending/running/done/failed), error samples, throughput.

6. Human stops / scales down (optional)

Command

```
scale_workers --job J123 --count 0 (or reduce ASG desired capacity)
```

terminates instances; state remains in Postgres + S3 outputs.