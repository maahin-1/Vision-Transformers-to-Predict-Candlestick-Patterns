"""ViT inference on a 700x500 chart image (same preprocessing as src/data.py)."""
import os

import albumentations as A
import numpy as np
import torch

from model import ViT
from rules import CLASSES  # noqa: F401  (re-exported for callers)

CHECKPOINT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints', '25_model.pt')
CROP = dict(x_min=130, y_min=43, x_max=202, y_max=147)  # last eight candles, in 224x224 space

_model = None
_transform = A.Compose([
    A.Resize(224, 224),
    A.Crop(**CROP),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    A.ToTensorV2(),
])


def load_model():
    global _model
    if _model is None:
        if not os.path.exists(CHECKPOINT):
            raise FileNotFoundError(f'{CHECKPOINT} not found. Run: uv run download_model.py')
        torch.set_num_threads(2)  # keep CPU inference from saturating every core
        model = ViT()
        model.load_state_dict(torch.load(CHECKPOINT, weights_only=True, map_location='cpu'))
        model.eval()
        _model = model
    return _model


def model_view(image):
    """The 104x72 crop the model actually sees, as a uint8 RGB array."""
    return A.Compose([A.Resize(224, 224), A.Crop(**CROP)])(image=image)['image']


def predict(image):
    """image: (500, 700, 3) uint8 RGB -> list of 5 class probabilities."""
    model = load_model()
    tensor = _transform(image=np.ascontiguousarray(image))['image']
    with torch.inference_mode():
        return torch.softmax(model(tensor.unsqueeze(0)), dim=1)[0].tolist()
