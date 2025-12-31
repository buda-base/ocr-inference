## Images on S3

- About 30M files, including 25M Tibetan images, about 12TB.
- on bucket archive.tbrc.org

#### s3 Path format

```
Works/{md5_2}/{W_id}/images/{W_id}-{I_id}/{I_id}{img_num}.{ext}
```

were:
- `{md5_2}` is the first 2 hex digits of the md5 sum of the W_id (ex: `60` for `W22084`)
- `{W_id}` is the BDRC id of the image instance (a.k.a. scans or scans collection)
- `{I_id}` is the BDRC id of the volume (a.k.a. image group), globally unique
- `{img_num}` is the image number, starting at 1, padded on 4 digits (ex: 0001)
- `{ext}` is ideally only `tiff`, `tif`, `jpeg` or `jpg`, but can very rarely be something else

notes:
- `{img_num}` is enforced 99% of the time but not 100%, some folders skip an image number, others have letter `a` at the end. Very rarely there are two files with the same image number with different extensions

#### Image formats

Images are typically:
- single-page group4 encoded tiffs
- RGB or grayscale jpegs

or much more rarely (ignorable):
- other types of single-page tiffs (jpeg encoded, zip encoded, LZW encoded)
- CMYK jpegs

Image size is typically < 800kb, image largest dimension (width / height) is typically < 2,500, with a few exceptions.

#### Manifests

A manifest named `dimensions.json` is present in the folder of each image group (`Works/{md5_2}/{W_id}/images/{W_id}-{I_id}/dimensions.json`). It is a gzipped-compressed json (not a json file). Its format is

```
[
  {
  	"filename": "I1230001.jpg",
  	"width": 2000,
  	"height": 1000
  },
  ...
]
```

with occasionally a `size` property indicating the size of the image in bytes, when superior to 2MB.

In some extremely rare cases, files referenced in the manifest won't be present on s3, this should be considered a terminal error.

#### Image updates

Very rarely images for a volume can be updated. In that case files are replaced so the same file path can refer to different actual images at different times. In order to make the system strong, we need to add a checksum of the image we're operating on. 

When images are replaced, the old images are preserved, but not on S3 (only on BDRC's internal servers).

BDRC does not maintain explicit human version numbers for volumes. The project will use a deterministic volume_version derived from the `dimensions.json` manifest. The version of a volume is the first 6 hex digits of the ETag of the `dimensions.json` manifest, or 6 random hex digits in case of collision.

#### Checksums

In order to make sure we know the version of the image that has been used for a task, a database will contain:
- the SHA256 checksum of each image
- the S3 ETag (mostly md5 but can be exceptionally different for multi-part uploads)
- the S3 last modification timestamp

Tasks should make sure they record at least the S3 ETags of the images they processed. The information is sent in the http request when GETting a file from the archive bucket.