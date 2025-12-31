-- Assumes: CREATE DATABASE ocr_batch ENCODING 'UTF8';

-- Types ------------------------------------------------------------------

CREATE TYPE job_status AS ENUM (
  'created',
  'running',
  'completed',
  'failed',
  'canceled'
);

CREATE TYPE task_status AS ENUM (
  'pending',
  'running',
  'done',
  'retryable_failed',
  'terminal_failed'
);

CREATE TYPE volume_result_status AS ENUM (
  'success',
  'partial',
  'failed'
);

CREATE TYPE image_type AS ENUM (
  'jpg',
  'png',
  'single_image_tiff',
  'jp2',
  'jpxl'
);

CREATE TYPE image_mode AS ENUM (
  '1',
  'L',
  'RGB',
  'RGBA',
  'CMYK',
  'P',
  'OTHER'
);

-- Tables -----------------------------------------------------------------

CREATE TABLE job_types (
  id                   serial PRIMARY KEY,
  name                 varchar(50) NOT NULL,
  model_name           varchar(100) NOT NULL,
  encoding             varchar(20) NOT NULL DEFAULT 'unicode',
  line_mode            varchar(20) NOT NULL DEFAULT 'line',
  k_factor             real NOT NULL DEFAULT 2.5,
  bbox_tolerance       real NOT NULL DEFAULT 4.0,
  merge_lines          boolean NOT NULL DEFAULT false,
  use_tps              boolean NOT NULL DEFAULT false,
  artifact_granularity varchar(20) NOT NULL DEFAULT 'standard',
  description          text
);

CREATE UNIQUE INDEX job_types_name_uniq ON job_types (name);

-- Seed default job type
INSERT INTO job_types (name, model_name, encoding, description)
VALUES ('woodblock-stacks-unicode', 'Woodblock-Stacks', 'unicode', 'Woodblock OCR with Unicode output');


CREATE TABLE volumes (
  id        serial PRIMARY KEY,
  bdrc_w_id varchar(32) NOT NULL,
  bdrc_i_id varchar(32) NOT NULL
);

CREATE UNIQUE INDEX volumes_w_i_uniq ON volumes (bdrc_w_id, bdrc_i_id);
CREATE INDEX volumes_bdrc_w_id_idx ON volumes (bdrc_w_id);
CREATE INDEX volumes_bdrc_i_id_idx ON volumes (bdrc_i_id);


CREATE TABLE workers (
  worker_id         uuid PRIMARY KEY,
  instance_id       text,
  hostname          text,
  tags              jsonb,
  started_at        timestamptz NOT NULL DEFAULT now(),
  last_heartbeat_at timestamptz NOT NULL DEFAULT now(),
  stopped_at        timestamptz
);

CREATE INDEX workers_last_heartbeat_idx ON workers (last_heartbeat_at);
CREATE INDEX workers_instance_id_idx ON workers (instance_id);


CREATE TABLE jobs (
  id               bigserial PRIMARY KEY,
  job_key          text NOT NULL,
  type_id          integer NOT NULL REFERENCES job_types(id),
  status           job_status NOT NULL DEFAULT 'created',
  created_at       timestamptz NOT NULL DEFAULT now(),
  started_at       timestamptz,
  finished_at      timestamptz,
  total_tasks      integer NOT NULL DEFAULT 0,
  done_tasks       integer NOT NULL DEFAULT 0,
  failed_tasks     integer NOT NULL DEFAULT 0,
  last_progress_at timestamptz
);

CREATE UNIQUE INDEX jobs_job_key_uniq ON jobs (job_key);
CREATE INDEX jobs_type_id_idx ON jobs (type_id);
CREATE INDEX jobs_status_idx ON jobs (status);
CREATE INDEX jobs_created_at_idx ON jobs (created_at);


CREATE TABLE tasks (
  id                  bigserial PRIMARY KEY,
  job_id              bigint NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  volume_id           integer NOT NULL REFERENCES volumes(id),
  status              task_status NOT NULL DEFAULT 'pending',
  attempts            integer NOT NULL DEFAULT 0,
  max_attempts        integer NOT NULL DEFAULT 3,
  leased_by_worker_id uuid REFERENCES workers(worker_id),
  lease_expires_at    timestamptz,
  started_at          timestamptz,
  done_at             timestamptz,
  output_prefix       text,
  last_error          jsonb
);

CREATE INDEX tasks_job_id_idx ON tasks (job_id);
CREATE INDEX tasks_job_status_idx ON tasks (job_id, status);
CREATE INDEX tasks_status_idx ON tasks (status);
CREATE INDEX tasks_volume_id_idx ON tasks (volume_id);
CREATE INDEX tasks_lease_expires_idx ON tasks (lease_expires_at) WHERE status = 'running';


CREATE TABLE task_metrics (
  id                       bigserial PRIMARY KEY,
  task_id                  bigint NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
  total_pages              integer NOT NULL,
  successful_pages         integer NOT NULL,
  total_duration_ms        real NOT NULL,
  avg_duration_per_page_ms real NOT NULL,
  total_lines_detected     integer NOT NULL,
  created_at               timestamptz NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX task_metrics_task_id_uniq ON task_metrics (task_id);


CREATE TABLE page_metrics (
  id                bigserial PRIMARY KEY,
  task_id           bigint NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
  image_name        text NOT NULL,
  duration_ms       real NOT NULL,
  lines_detected    integer NOT NULL,
  lines_processed   integer NOT NULL,
  dewarping_applied boolean NOT NULL,
  rotation_angle    real NOT NULL
);

CREATE INDEX page_metrics_task_id_idx ON page_metrics (task_id);


CREATE TABLE image_files (
  id         serial PRIMARY KEY,
  sha256     bytea NOT NULL CHECK (octet_length(sha256) = 32),
  size       bigint NOT NULL,
  image_type image_type NOT NULL,
  image_mode image_mode NOT NULL,
  width      integer NOT NULL,
  height     integer NOT NULL,
  quality    smallint,
  bps        smallint NOT NULL
);

CREATE UNIQUE INDEX image_files_sha256_size_uniq ON image_files (sha256, size);


CREATE TABLE image_paths (
  id                     bigserial PRIMARY KEY,
  image_file_id          integer REFERENCES image_files(id),
  volume_id              integer REFERENCES volumes(id),
  image_file_path        varchar(256),
  valid                  boolean DEFAULT true,
  image_number           smallint NOT NULL,
  image_number_corrected smallint
);

CREATE INDEX image_paths_image_file_id_idx ON image_paths (image_file_id);
CREATE INDEX image_paths_volume_id_idx ON image_paths (volume_id);


CREATE TABLE volume_results (
  id               bigserial PRIMARY KEY,
  job_id           bigint NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  volume_id        integer NOT NULL REFERENCES volumes(id),
  status           volume_result_status NOT NULL,
  confidence_score real,
  est_cer          real,
  complex_layout   boolean,
  created_at       timestamptz NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX volume_results_job_volume_uniq ON volume_results (job_id, volume_id);
CREATE INDEX volume_results_volume_id_idx ON volume_results (volume_id);


-- Comments ---------------------------------------------------------------

COMMENT ON TABLE jobs IS 'Batch processing jobs';
COMMENT ON COLUMN jobs.total_tasks IS 'Total number of tasks (volumes) in this job';
COMMENT ON COLUMN jobs.done_tasks IS 'Number of successfully completed tasks';
COMMENT ON COLUMN jobs.failed_tasks IS 'Number of terminally failed tasks';

COMMENT ON TABLE tasks IS 'Individual volume processing tasks within a job';
COMMENT ON COLUMN tasks.max_attempts IS 'Maximum retry attempts before terminal failure';
COMMENT ON COLUMN tasks.output_prefix IS 'S3 prefix where outputs are stored';

COMMENT ON TABLE volumes IS 'BDRC volume identifiers';
COMMENT ON COLUMN volumes.bdrc_w_id IS 'BDRC Work RID (e.g., W22084)';
COMMENT ON COLUMN volumes.bdrc_i_id IS 'BDRC Image Group RID (e.g., I0896)';

COMMENT ON TABLE image_files IS 'Unique image file metadata by content hash';
COMMENT ON COLUMN image_files.size IS 'File size in bytes';
COMMENT ON COLUMN image_files.image_mode IS 'PIL image mode';
COMMENT ON COLUMN image_files.quality IS 'JPEG quality (0-100) or PNG compression (0-9)';
COMMENT ON COLUMN image_files.bps IS 'Bits per sample';
