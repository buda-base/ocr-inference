"""
A simple cli for running line inference using a PyTorch pipeline:

usage:

python infer_lines_cli.py --input-dir D:/Datasets/W2PD17487-v2 --output-dir D:/Datasets/W2PD17487-v2/parquet --checkpoint Models/BDRC/PhotiLines_new_2024-10-7_21-26/2024-10-7_21-26/segmentation_model.pth --batch-size 8 --num-workers 6 --class-threshold 0.9
"""

import os
import argparse

from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader

from BDRC.inference import ImageInferenceDataset
from BDRC.utils import (
    infer_batch,
    load_model,
    multi_image_collate_fn,
    write_result_parquet,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run line segmentation inference using the PyTorch tiling pipeline"
    )

    # I/O
    parser.add_argument(
        "-i", "--input-dir",
        required=True,
        help="Input directory containing images",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="parquet_out",
        help="Output directory for Parquet files",
    )

    # Model
    parser.add_argument(
        "-c", "--checkpoint",
        required=True,
        help="Path to model checkpoint (.pth)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=1,
        help="Number of output classes (default: 1)",
    )

    # Inference params
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=8,
        help="Batch size (number of images per batch)",
    )
    parser.add_argument(
        "-w", "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers",
    )
    parser.add_argument(
        "-t", "--class-threshold",
        type=float,
        default=0.85,
        help="Sigmoid threshold for binary mask",
    )

    # Device
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run inference on",
    )

    # Misc
    parser.add_argument(
        "--no-pin-memory",
        action="store_true",
        help="Disable pin_memory in DataLoader",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    # Dataset
    dataset = ImageInferenceDataset(args.input_dir)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        collate_fn=multi_image_collate_fn,
    )

    # Model
    model = load_model(
        checkpoint_path=args.checkpoint,
        num_classes=args.num_classes,
        device=device,
    )
    model.eval()

    # Timing
    if device == "cuda":
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    results = []

    with torch.no_grad():
        for all_tiles, tile_ranges, metas in tqdm(
            loader,
            total=len(loader),
            desc="Running inference",
        ):
            result = infer_batch(
                model,
                all_tiles,
                tile_ranges,
                metas,
                class_threshold=args.class_threshold,
            )
            results.append(result)

    if device == "cuda":
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end) / 1000.0
        print(f"Elapsed inference time: {elapsed:.2f}s")

    print(f"Processed {len(dataset)} images")

    # Write Parquet
    os.makedirs(args.output_dir, exist_ok=True)

    for res in tqdm(results, desc="Writing Parquet"):
        write_result_parquet(res, out_dir=args.output_dir)


if __name__ == "__main__":
    main()
