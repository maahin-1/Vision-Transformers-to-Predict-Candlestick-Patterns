#!/usr/bin/env python3
"""Download pre-trained model checkpoint from GitHub releases."""

import os
import sys
import urllib.request
from pathlib import Path


CHECKPOINT_URL = "https://github.com/maahin-1/Vision-Transformers-to-Predict-Candlestick-Patterns/releases/download/v1.0.0/25_model.pt"
CHECKPOINT_PATH = "checkpoints/25_model.pt"


def download_model():
    """Download the model checkpoint if it doesn't exist."""
    checkpoint_path = Path(CHECKPOINT_PATH)

    # Create directory if needed
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Skip if already exists
    if checkpoint_path.exists():
        print(f"✓ Model already exists at {CHECKPOINT_PATH}")
        return

    print(f"Downloading model from {CHECKPOINT_URL}...")
    try:
        urllib.request.urlretrieve(CHECKPOINT_URL, str(checkpoint_path))
        print(f"✓ Model downloaded to {CHECKPOINT_PATH}")
    except Exception as e:
        print(f"✗ Failed to download model: {e}")
        sys.exit(1)


if __name__ == "__main__":
    download_model()
