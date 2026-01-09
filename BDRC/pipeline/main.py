import argparse
import asyncio
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import torch

from rich.console import Console
from rich.live import Live
from rich.progress import Progress, BarColumn, TextColumn, TaskProgressColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.table import Table

from .ld_volume_worker import LDVolumeWorker
from .types_common import VolumeTask, ImageTask
from .config import PipelineConfig
from .s3ctx import S3Context

try:
    import boto3  # type: ignore
    import botocore  # type: ignore
except Exception:  # pragma: no cover
    boto3 = None
    botocore = None

SESSION = boto3.Session() if boto3 is not None else None
S3 = SESSION.client("s3") if SESSION is not None else None

console = Console()

# Common image file extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}

## Some helpers (get image list for s3 mode)

def get_s3_folder_prefix(w_id, i_id):
    """
    gives the s3 prefix (~folder) in which the volume will be present.
    inpire from https://github.com/buda-base/buda-iiif-presentation/blob/master/src/main/java/
    io/bdrc/iiif/presentation/ImageInfoListService.java#L73
    Example:
       - w_id=W22084, i_id=I0886
       - result = "Works/60/W22084/images/W22084-0886/
    where:
       - 60 is the first two characters of the md5 of the string W22084
       - 0886 is:
          * the image group ID without the initial "I" if the image group ID is in the form I\\d\\d\\d\\d
          * or else the full image group ID (incuding the "I")
    """
    md5 = hashlib.md5(str.encode(w_id))
    two = md5.hexdigest()[:2]

    pre, rest = i_id[0], i_id[1:]
    if pre == 'I' and rest.isdigit() and len(rest) == 4:
        suffix = rest
    else:
        suffix = i_id

    return 'Works/{two}/{RID}/images/{RID}-{suffix}/'.format(two=two, RID=w_id, suffix=suffix)

def gets3blob(s3Key: str) -> Tuple[Optional[io.BytesIO], Optional[str]]:
    """
    Downloads an S3 object and returns (BytesIO buffer, etag).
    Returns (None, None) if object not found.
    """
    if S3 is None or botocore is None:
        raise RuntimeError(
            "S3 mode requires boto3+botocore. Install them (see requirements.txt) "
            "or use --input-folder / --output-folder file:///... for local mode."
        )
    try:
        # Single request: get_object provides both Body and ETag.
        obj = S3.get_object(Bucket="archive.tbrc.org", Key=s3Key)
        etag = obj.get("ETag", None)
        body_bytes: bytes = obj["Body"].read()
        return io.BytesIO(body_bytes), etag
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            return None, None
        else:
            raise

def get_volume_version(w_id, i_id, s3_etag):
    return s3_etag.replace('"', "")[:6]

def get_image_list_and_version_s3(w_id: str, i_id: str) -> Tuple[Optional[List[ImageTask]], Optional[str]]:
    """
    Gets manifest of files in a volume and returns list of ImageTasks and version.
    Returns (None, None) if manifest not found.
    """
    vol_s3_prefix = get_s3_folder_prefix(w_id, i_id)
    vol_manifest_s3_key = vol_s3_prefix + "dimensions.json"
    blob, etag = gets3blob(vol_manifest_s3_key)
    if blob is None:
        return None, None
    
    i_version = get_volume_version(w_id, i_id, etag or "")
    blob.seek(0)
    b = blob.read()
    ub = gzip.decompress(b)
    s = ub.decode('utf8')
    data = json.loads(s)
    # data is in the form: [ { "filename": "I123.jpg", ... }, ... ]
    
    # Convert to ImageTask list
    image_tasks = []
    for item in data:
        filename = item.get("filename")
        if not filename:
            continue

        # Filter files by extension (images only)
        ext = Path(str(filename)).suffix.lower()
        if ext not in IMAGE_EXTENSIONS:
            continue

        if filename:
            # Build full S3 key by prefixing with volume prefix
            s3_key = vol_s3_prefix + filename
            source_uri = f"s3://archive.tbrc.org/{s3_key}"
            image_tasks.append(ImageTask(
                source_uri=source_uri,
                img_filename=filename
            ))
    
    return image_tasks, i_version


def make_progress_hook(q: asyncio.Queue[Dict[str, Any]]):
    # Called from inside pipeline tasks; must not block.
    def hook(evt: Dict[str, Any]) -> None:
        try:
            q.put_nowait(evt)
        except asyncio.QueueFull:
            # Drop progress events under load; pipeline must win.
            pass
    return hook


def render_queue_table(worker) -> Table:
    rows = [
        ("prefetch→decode", worker.q_prefetcher_to_decoder),
        ("decode→gpu1", worker.q_decoder_to_gpu_pass_1),
        ("gpu1→post", worker.q_gpu_pass_1_to_post_processor),
        ("post→gpu2", worker.q_post_processor_to_gpu_pass_2),
        ("gpu2→post", worker.q_gpu_pass_2_to_post_processor),
        ("post→writer", worker.q_post_processor_to_writer),
    ]

    t = Table(title="Queue fullness", expand=True)
    t.add_column("Queue")
    t.add_column("Size", justify="right")
    t.add_column("Max", justify="right")
    t.add_column("Pressure", justify="right")

    for name, q in rows:
        size = q.qsize()
        mx = q.maxsize or 0
        pct = (100.0 * size / mx) if mx else 0.0
        t.add_row(name, str(size), str(mx), f"{pct:5.1f}%")

    return t


async def ui_loop(
    *,
    events: asyncio.Queue[Dict[str, Any]],
    worker,
    total: Optional[int],
    show_queues: bool,
) -> None:
    progress = Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        transient=False,
        console=console,
    )

    done_task = progress.add_task("Images persisted", total=total)
    err_task = progress.add_task("Errors", total=None)

    flush_state = "idle"
    last_img = ""
    fatal: Optional[str] = None
    fatal_stage: Optional[str] = None
    fatal_hint: Optional[str] = None

    def render():
        # Compose a single live layout (progress + optional queue panel)
        grid = Table.grid(expand=True)
        grid.add_row(progress)

        meta = Table.grid(expand=True)
        total_s = str(total) if total is not None else "?"
        meta_line = f"Total: [bold]{total_s}[/bold]    Flush: [bold]{flush_state}[/bold]    Last: {last_img}"
        if fatal:
            meta_line += f"\n[bold red]FATAL[/bold red] ({fatal_stage or 'unknown'}): {fatal}"
            if fatal_hint:
                meta_line += f"\nHint: {fatal_hint}"
        meta.add_row(meta_line)
        grid.add_row(meta)

        if show_queues:
            grid.add_row(render_queue_table(worker))
        return grid

    with Live(render(), refresh_per_second=10, console=console) as live:
        while True:
            try:
                evt = await asyncio.wait_for(events.get(), timeout=1.0)
            except asyncio.TimeoutError:
                # periodic refresh for queue fullness even if no events
                live.update(render())
                continue

            et = evt.get("type")
            if et == "item":
                last_img = evt.get("img") or ""
                if evt.get("ok"):
                    progress.advance(done_task, 1)
                else:
                    progress.advance(err_task, 1)
            elif et == "flush":
                flush_state = evt.get("state", "unknown")
            elif et == "fatal":
                fatal = evt.get("error") or "unknown error"
                fatal_stage = evt.get("stage")
                fatal_hint = evt.get("hint")
            elif et == "close":
                flush_state = "closed"
                live.update(render())
                return

            live.update(render())


def _normalize_uri(path_or_uri: str) -> str:
    """Convert local path to file:// URI if needed, otherwise return as-is."""
    if path_or_uri.startswith(("s3://", "file://")):
        return path_or_uri.rstrip('/')
    # Convert to absolute path and then to file:// URI
    abs_path = os.path.abspath(path_or_uri)
    # On Windows, handle backslashes
    if os.name == 'nt':
        abs_path = abs_path.replace('\\', '/')
        # Ensure proper format: file:///C:/...
        if abs_path[1] == ':':
            abs_path = '/' + abs_path
    return f"file://{abs_path}"


def _join_uri(base_uri: str, filename: str) -> str:
    """Join a filename to a base URI (s3:// or file://)."""
    base_uri = base_uri.rstrip('/')
    if base_uri.startswith("s3://"):
        return f"{base_uri}/{filename}"
    elif base_uri.startswith("file://"):
        # For file:// URIs, we need to handle path joining properly
        path_part = base_uri[7:]  # Remove "file://"
        if os.name == 'nt' and path_part.startswith('/') and len(path_part) > 1 and path_part[2] == ':':
            # Windows: file:///C:/path -> C:/path
            path_part = path_part[1:]
        elif os.name == 'nt' and not path_part.startswith('/'):
            # Already a Windows path
            pass
        joined = os.path.join(path_part, filename).replace('\\', '/')
        if os.name == 'nt' and joined[1] == ':':
            # Ensure file:///C:/ format for Windows
            return f"file:///{joined}"
        return f"file://{joined}"
    else:
        # Plain path
        joined = os.path.join(base_uri, filename)
        return _normalize_uri(joined)


def _get_local_image_tasks(input_folder: str) -> List[ImageTask]:
    """Scan input_folder for image files and create ImageTask list."""
    input_path = Path(input_folder)
    if not input_path.exists() or not input_path.is_dir():
        raise ValueError(f"Input folder does not exist or is not a directory: {input_folder}")
    
    image_tasks = []
    for file_path in input_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
            source_uri = _normalize_uri(str(file_path))
            image_tasks.append(ImageTask(
                source_uri=source_uri,
                img_filename=file_path.name
            ))
    
    if not image_tasks:
        raise ValueError(f"No image files found in {input_folder}")
    
    return image_tasks

def load_model(
    checkpoint_path: str | Path,
    *,
    classes: int = 1,
) -> torch.nn.Module:
    """
    Load the segmentation model from a checkpoint.

    Assumptions (as per existing training artifacts):
      - checkpoint is a dict with a "state_dict" key.
      - model architecture is DeepLabV3Plus from segmentation_models_pytorch.

    Performance:
      - load weights on CPU (fast, avoids GPU RAM spikes).
      - set eval + disable grads.
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError("Expected checkpoint dict with a 'state_dict' key")

    state_dict: Dict[str, torch.Tensor] = checkpoint["state_dict"]
    # Common training wrappers (DataParallel, Lightning, etc.)
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

    import segmentation_models_pytorch as sm

    # Keep this fully local/offline by default (no encoder pretrained weights download).
    model = sm.DeepLabV3Plus(encoder_name="resnet34", encoder_weights=None, classes=classes)
    model.load_state_dict(state_dict, strict=True)

    model.eval()
    model.requires_grad_(False)
    return model

async def run_one_volume(args):
    model = load_model(args.checkpoint, classes=1)
    
    # Build PipelineConfig with model
    cfg = PipelineConfig(
        s3_bucket="archive.tbrc.org",  # Default for S3 operations
        s3_region="us-east-1",
    )
    cfg.model = model
    
    # Build VolumeTask based on input mode
    volume_task: VolumeTask
    s3ctx: Optional[S3Context] = None
    
    if args.input_folder:
        # Local mode
        input_folder = os.path.abspath(args.input_folder)
        image_tasks = _get_local_image_tasks(input_folder)
        
        # Determine output folder
        if args.output_folder:
            output_base_uri = _normalize_uri(args.output_folder)
        else:
            # Default: output folder next to input folder
            output_base = str(Path(input_folder).parent / f"{Path(input_folder).name}_output")
            output_base_uri = _normalize_uri(output_base)
            # Create directory if local path
            if output_base_uri.startswith("file://"):
                path_part = output_base_uri[7:]
                if os.name == 'nt' and path_part.startswith('/') and len(path_part) > 1 and path_part[2] == ':':
                    path_part = path_part[1:]
                os.makedirs(path_part, exist_ok=True)
        
        parquet_uri = _join_uri(output_base_uri, "results.parquet")
        jsonl_uri = _join_uri(output_base_uri, "errors.jsonl")
        
        # Debug folder (always local)
        debug_folder = str(Path(input_folder).parent / f"{Path(input_folder).name}_debug")
        os.makedirs(debug_folder, exist_ok=True)
        
        volume_task = VolumeTask(
            io_mode="local",
            debug_folder_path=debug_folder,
            output_parquet_uri=parquet_uri,
            output_jsonl_uri=jsonl_uri,
            image_tasks=image_tasks,
        )
        
    elif args.w and args.i:
        # S3 mode
        w_id = args.w
        i_id = args.i
        
        image_tasks, i_version = get_image_list_and_version_s3(w_id, i_id)
        if image_tasks is None or i_version is None:
            raise ValueError(f"Failed to fetch image list for W{w_id} I{i_id}")
        
        # Determine output folder
        if args.output_folder:
            output_base_uri = _normalize_uri(args.output_folder).rstrip('/')
        else:
            output_base_uri = f"s3://bec.bdrc.io/artefacts/line_detection_v1/{w_id}-{i_id}-{i_version}"
        
        parquet_filename = f"{w_id}-{i_id}-{i_version}.parquet"
        jsonl_filename = f"{w_id}-{i_id}-{i_version}.jsonl"
        
        parquet_uri = _join_uri(output_base_uri, parquet_filename)
        jsonl_uri = _join_uri(output_base_uri, jsonl_filename)
        
        volume_task = VolumeTask(
            io_mode="s3",
            debug_folder_path=str(Path.home() / f"debug_{w_id}_{i_id}"),  # Local debug folder
            output_parquet_uri=parquet_uri,
            output_jsonl_uri=jsonl_uri,
            image_tasks=image_tasks,
        )
        
        # Create S3Context for S3 mode
        global_sem = asyncio.Semaphore(cfg.s3_max_inflight_global)
        s3ctx = S3Context(cfg, global_sem)
        
    else:
        raise ValueError("Either --input_folder OR both --w and --i must be provided")

    progress_events: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=10_000)
    hook = make_progress_hook(progress_events) if args.progress else None

    # Use async context manager for proper cleanup
    async with LDVolumeWorker(cfg, volume_task, s3ctx=s3ctx, progress=hook) as worker:
        total = len(volume_task.image_tasks)

        if args.progress:
            ui = asyncio.create_task(
                ui_loop(
                    events=progress_events,
                    worker=worker,
                    total=total,
                    show_queues=args.progress_queues,
                )
            )
            try:
                await worker.run()
            finally:
                # If pipeline dies early, unblock UI.
                try:
                    progress_events.put_nowait({"type": "close"})
                except asyncio.QueueFull:
                    pass
                await ui
        else:
            await worker.run()


def main():
    p = argparse.ArgumentParser(description="Line detection pipeline for OCR inference")
    
    # Input mode
    p.add_argument(
        "--input-folder",
        type=str,
        help="Local folder containing image files (creates local VolumeTask)"
    )
    s3_group = p.add_argument_group("S3 mode (requires both --w and --i)")
    s3_group.add_argument(
        "--w",
        type=str,
        help="Work ID (e.g., W22084) for S3 mode"
    )
    s3_group.add_argument(
        "--i",
        type=str,
        help="Image group ID (e.g., I0886) for S3 mode"
    )
    
    # Output
    p.add_argument(
        "--output-folder",
        type=str,
        help="Output folder path or URI (file:// or s3://). Defaults set based on input mode."
    )
    
    # Model
    p.add_argument(
        "-c", "--checkpoint",
        required=True,
        help="Path to model checkpoint (.pth)",
    )
    
    # UI
    p.add_argument("--progress", action="store_true", help="Show progress UI on stderr.")
    p.add_argument(
        "--progress-queues",
        action="store_true",
        help="Also show queue fullness (updates ~1s). Implies --progress.",
    )
    
    args = p.parse_args()
    
    # Validate input mode arguments
    if args.input_folder:
        if args.w or args.i:
            p.error("--input-folder cannot be used with --w or --i")
    else:
        if not (args.w and args.i):
            p.error("Either --input-folder OR both --w and --i must be provided")
    
    if args.progress_queues:
        args.progress = True

    asyncio.run(run_one_volume(args))


if __name__ == "__main__":
    main()
