"""
Model loading utilities for the line detection pipeline.
"""

from pathlib import Path
from typing import Dict

import torch


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

