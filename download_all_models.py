# ruff: noqa: T201
from bdrc.utils import download_model
from config import MODEL_DICT


def download_all_models() -> None:
    print(f"Available Models: {list(MODEL_DICT.keys())}")

    for v in MODEL_DICT.values():
        try:
            print(download_model(v))
        except Exception:  # noqa: BLE001, PERF203
            print(f"Failed to download model: {v}")


if __name__ == "__main__":
    download_all_models()
