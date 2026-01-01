## Jobs descriptions

This folder contains the documentation for the various jobs, as well as the list of volumes they should be run on.

We use the following conventions:
- everything is case-sensitive
- `{job_name}` corresponds exactly to the `job_name` field of the sql database.
- each worker uses the following AWS tags:
   * `Project` : `BEC`
   * `Name`: `BEC_{job_name}` (replacing with the actual value)
- each worker uses two SQS queues:
   * `BEC_{job_name}` for the main queue
   * `BEC_{job_name}_DLQ` for the dead letter queue

#### {job_name}.md

A Markdown file with a yaml frontmatter. The frontmatter can contain any field, we recommend the following:

- `depends_on` (single value or list)
- `new_version_of` (idem)
- `creation_timestamp`

We recommend 

#### {job_name}.csv

A csv file with one line of header and two columns:
- `w_id`
- `i_id`
