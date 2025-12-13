import argparse
import os
import sys

from Config import MODEL_DICT
from BDRC.utils import download_model


def main():
    parser = argparse.ArgumentParser(
        description="Download a model from HuggingFace."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=MODEL_DICT.keys(),
        help="Model key defined in MODEL_DICT",
    )
    parser.add_argument(
        "--output-dir",
        default="Models",
        help="Target directory for downloaded models (default: Models)",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        download_model(MODEL_DICT[args.model])  # ideally pass args.output_dir
    except Exception as e:
        print(f"[ERROR] Failed to download model '{args.model}': {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[OK] Model '{args.model}' downloaded successfully.")


if __name__ == "__main__":
    main()
