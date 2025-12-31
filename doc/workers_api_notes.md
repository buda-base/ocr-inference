A worker uses the following API, most function are provided by the BEC ops library, which is shared by all workers.

## Conventions

- a **job** is an action that can be performed on many volumes (ex: "OCR using model X" or "line detection")
- `job_id` is a string representing a job
- a **task** is running a job on a volume (ex: "OCR using model X on volume V")
- a **volume** is identified by the tuple `(w_id, i_id, i_version)`
- an **artefact** is any output the worker produces (ex: transcriptions from OCR)

#### get_next_task(job_id)

returns tuple `(w_id, i_id, i_version)` or `None` if not tasks available

typically `(w_id, i_id)` only from SQS, `i_version` from SQL to get the latest volume version

#### get_output_info(job_id, w_id, i_id, i_version)

returns `(bucket_name, prefix)`. The worker writes its artefacts there, no output format is enforced except it should write a `success.json` once the task is executed and all the other files have been successfully written. The file should contain 

```json
{
	"worker_id": "...",
	"success_timestamp": "..."
}
```

typically (`bec.bdrc.io`, `artefacts/{job_id}/{w_id}-{i_id}-{i_version}/`)

#### is_successfully_done(job_id, w_id, i_id)

returns `True` if the job has already run successfully for a volume, `False` otherwise. Used in two cases:
- the worker uses this function at the beginning of a task, if it returns true the worker ends the task
- if a job is dependent on another job, the worker uses this function to make sure the job has run successfully on the volume

typically checking the presence of `s3://bec.bdrc.io/artefacts/{job_id}/{w_id}-{i_id}/success.json`

#### task_done(job_id, w_id, i_id, i_version, status)

called at the end of a task

typically sends the information to SQL and SQS

## TODO

register() adds to SQL, or if already registered does nothing, get_worker_id()?

heartbeat()

% done (register total number of items in task, call function at end of each item)

no_tasks_left(): sends notification, later: kills the worker?
