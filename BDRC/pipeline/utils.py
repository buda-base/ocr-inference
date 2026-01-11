"""
Utility functions for S3 operations, URI handling, and image task creation.
"""
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import boto3  # type: ignore
    import botocore  # type: ignore
except Exception:  # pragma: no cover
    boto3 = None
    botocore = None

from .types_common import ImageTask

SESSION = boto3.Session() if boto3 is not None else None
S3 = SESSION.client("s3") if SESSION is not None else None

# Common image file extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}


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

