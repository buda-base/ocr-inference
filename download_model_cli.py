# ruff: noqa: T201
import argparse
import sys
from pathlib import Path

from bdrc.utils import download_model
from config import MODEL_DICT


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a model from HuggingFace.")
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

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    try:
        download_model(MODEL_DICT[args.model])  # ideally pass args.output_dir
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] Failed to download model '{args.model}': {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[OK] Model '{args.model}' downloaded successfully.")


if __name__ == "__main__":
    main()
