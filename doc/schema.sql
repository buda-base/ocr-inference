-- Assumes the database itself is created with:
--   CREATE DATABASE your_db ENCODING 'UTF8';

-- Domains / types --------------------------------------------------------

CREATE TYPE job_status AS ENUM (
  'running',
  'completed',
  'failed',
  'canceled',
  'created',
);

CREATE TYPE task_status AS ENUM (
  'pending',
  'running',
  'done',
  'retryable_failed',
  'terminal_failed'
);

CREATE TYPE shard_status AS ENUM (
  'pending',
  'leased',
  'done'
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
  id          serial PRIMARY KEY,
  name        varchar(50)  NOT NULL,
  version     varchar(100) NOT NULL,
  description text
);

CREATE UNIQUE INDEX job_types_name_version_uniq
  ON job_types (name, version);


CREATE TABLE volumes (
  id         serial PRIMARY KEY,
  bdrc_w_id  varchar(32) NOT NULL,
  bdrc_i_id  varchar(32) NOT NULL
);

CREATE UNIQUE INDEX volumes_w_i_uniq
  ON volumes (bdrc_w_id, bdrc_i_id);

CREATE INDEX volumes_bdrc_w_id_idx ON volumes (bdrc_w_id);
CREATE INDEX volumes_bdrc_i_id_idx ON volumes (bdrc_i_id);


CREATE TABLE IF NOT EXISTS jobs (
  id                bigserial PRIMARY KEY,
  -- external/job-id used in S3 paths and CLI output (e.g. "J123")
  job_key           text NOT NULL,
  type_id           integer NOT NULL REFERENCES job_types(id),
  status            job_run_status NOT NULL DEFAULT 'created',
  created_at        timestamptz NOT NULL DEFAULT now(),
  started_at        timestamptz,
  finished_at       timestamptz,
  config            jsonb,
  last_progress_at  timestamptz
);

CREATE UNIQUE INDEX IF NOT EXISTS jobs_job_key_uniq ON jobs(job_key);
CREATE INDEX IF NOT EXISTS jobs_type_id_idx         ON jobs(type_id);
CREATE INDEX IF NOT EXISTS jobs_status_idx          ON jobs(status);
CREATE INDEX IF NOT EXISTS jobs_created_at_idx      ON jobs(created_at);

CREATE TABLE IF NOT EXISTS tasks (
  id               bigserial PRIMARY KEY,
  job_id           bigint NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  volume_id        integer NOT NULL REFERENCES volumes(id),
  status           task_status NOT NULL DEFAULT 'pending',
  attempts         integer NOT NULL DEFAULT 0,
  leased_by_worker_id uuid REFERENCES workers(worker_id),
  lease_expires_at timestamptz,
  started_at       timestamptz,
  done_at          timestamptz,
  -- simple execution stats (store more in stats_json if needed)
  stats_json       jsonb,
  -- last failure payload (string, stack trace, exit code, etc.)
  last_error       jsonb
);

-- workers are ephemeral and entries can be deleted after they are stopped
CREATE TABLE IF NOT EXISTS workers (
  worker_id           uuid PRIMARY KEY,
  instance_id         text,         -- e.g. EC2 i-0123...
  hostname            text,
  tags                jsonb,
  started_at          timestamptz NOT NULL DEFAULT now(),
  last_heartbeat_at   timestamptz NOT NULL DEFAULT now(),
  stopped_at          timestamptz
);

-- quickly find "alive" workers (your app can define alive as heartbeat within N seconds)
CREATE INDEX IF NOT EXISTS workers_last_heartbeat_idx ON workers(last_heartbeat_at);
CREATE INDEX IF NOT EXISTS workers_instance_id_idx     ON workers(instance_id);

CREATE TABLE IF NOT EXISTS shards (
  id               bigserial PRIMARY KEY,
  job_id           bigint NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  shard_name       text NOT NULL,
  status           shard_status NOT NULL DEFAULT 'pending',
  leased_by_worker_id uuid REFERENCES workers(worker_id),
  leased_at        timestamptz,
  lease_expires_at timestamptz,
  done_at          timestamptz,
  total_volumes    integer,
  done_volumes     integer,
  last_error       jsonb
);

CREATE UNIQUE INDEX IF NOT EXISTS shards_job_name_uniq   ON shards(job_id, shard_name);
CREATE INDEX IF NOT EXISTS shards_job_status_idx         ON shards(job_id, status);
CREATE INDEX IF NOT EXISTS shards_lease_exp_idx          ON shards(lease_expires_at);
CREATE INDEX IF NOT EXISTS shards_leased_by_worker_idx   ON shards(leased_by_worker_id);

CREATE TABLE image_files (
  id         serial PRIMARY KEY,
  sha256     bytea      NOT NULL CHECK (octet_length(sha256) = 32),
  size       bigint     NOT NULL,
  image_type image_type NOT NULL,
  image_mode image_mode NOT NULL,
  width      integer    NOT NULL,
  height     integer    NOT NULL,
  quality    smallint,
  bps        smallint   NOT NULL
);

CREATE UNIQUE INDEX image_files_sha256_size_uniq
  ON image_files (sha256, size);


CREATE TABLE volume_results (
  result_id        bigserial PRIMARY KEY,
  job_id           bigint       NOT NULL,
  volume_id        integer      NOT NULL,
  status           image_status NOT NULL,
  confidence_score real,
  est_cer          real,
  complex_layout   boolean
);

CREATE UNIQUE INDEX volume_results_job_volume_uniq
  ON volume_results (job_id, volume_id);

CREATE INDEX volume_results_volume_id_idx
  ON volume_results (volume_id);

CREATE UNIQUE INDEX image_results_job_image_uniq
  ON image_results (job_id, image_id);

CREATE INDEX image_results_job_id_idx
  ON image_results (job_id);


CREATE TABLE image_paths (
  id                     bigserial PRIMARY KEY,
  image_file_id          integer,
  volume_id              integer,
  image_file_path        varchar(256),
  valid                  boolean,
  image_number           smallint NOT NULL,
  image_number_corrected smallint
);

CREATE INDEX image_paths_image_file_id_idx ON image_paths (image_file_id);
CREATE INDEX image_paths_volume_id_idx     ON image_paths (volume_id);
CREATE INDEX image_paths_image_file_path_idx ON image_paths (image_file_path);

-- Constraints (FKs) ------------------------------------------------------

ALTER TABLE jobs
  ADD CONSTRAINT jobs_volume_fk
    FOREIGN KEY (volume_id) REFERENCES volumes (id);

ALTER TABLE jobs
  ADD CONSTRAINT jobs_type_fk
    FOREIGN KEY (type_id) REFERENCES job_types (id);

ALTER TABLE volume_results
  ADD CONSTRAINT volume_results_job_fk
    FOREIGN KEY (job_id) REFERENCES jobs (id);

ALTER TABLE volume_results
  ADD CONSTRAINT volume_results_volume_fk
    FOREIGN KEY (volume_id) REFERENCES volumes (id);

ALTER TABLE image_paths
  ADD CONSTRAINT image_paths_image_file_fk
    FOREIGN KEY (image_file_id) REFERENCES image_files (id);

ALTER TABLE image_paths
  ADD CONSTRAINT image_paths_volume_fk
    FOREIGN KEY (volume_id) REFERENCES volumes (id);

-- Comments ---------------------------------------------------------------

COMMENT ON TABLE jobs IS 'Jobs are at the volume level';

COMMENT ON COLUMN jobs.config IS 'JSON configuration of the model';

COMMENT ON COLUMN image_files.size IS 'the size in bytes';

COMMENT ON COLUMN image_files.image_mode IS 'names are from PIL.mode';

COMMENT ON COLUMN image_files.width IS
  'width of the bitmap (not taking a potential exif rotation into account)';

COMMENT ON COLUMN image_files.height IS
  'height of the bitmap (not taking a potential exif rotation into account)';

COMMENT ON COLUMN image_files.quality IS
  'relevant only for jpg, png and single_image_tiff encoded as jpg: quality of encoding. JPEG is represented between 0 and 100. For PNG this column encodes the compression between 0 and 9.';

COMMENT ON COLUMN image_files.bps IS 'bits per sample';

COMMENT ON COLUMN volumes.bdrc_w_id IS
  'the BDRC RID (ex: W22084), unique persistent identifier, ASCII string no longer than 32 characters';

COMMENT ON COLUMN volumes.bdrc_i_id IS
  'the BDRC RID (ex: I0896), unique persistent identifier, ASCII string no longer than 32 characters';

COMMENT ON TABLE image_paths IS 'Table recording paths of image files';

COMMENT ON COLUMN image_paths.image_file_path IS
  'Unicode string (64 Unicode characters max) representing the (case sensitive) path relative to the image group folder.';

COMMENT ON COLUMN image_paths.valid IS
  'true by default, false when image changed path or is no longer in archive';

COMMENT ON COLUMN image_paths.image_number IS 'derived from the file name';

COMMENT ON COLUMN image_paths.image_number_corrected IS
  'null most of the time, can contain the corrected image number, taken from the pagination information';
