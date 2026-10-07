"""Attention heatmap: which parts of the 104x72 crop the class token relies on.

Uses attention rollout (Abnar & Zuidema, 2020): average the heads, add the residual connection,
renormalise each layer, then multiply the layers together. The class token's row of the result
says how much each 8x8 patch contributed to the prediction. The rollout and overlay are plain
numpy/PIL; only `layer_attentions` needs torch.
"""
import numpy as np
from PIL import Image

GRID_ROWS, GRID_COLS = 13, 9  # 104x72 crop in 8x8 patches
RECENT_COLS = 3  # right-most patch columns, about the last three candles
_STOPS = np.array([0.0, 0.35, 0.7, 1.0])
_COLORS = np.array([[20, 10, 60], [150, 40, 130], [240, 120, 40], [255, 235, 90]], dtype=float)


def layer_attentions(model, tensor):
    """Head-averaged attention of every encoder layer, as a list of (tokens, tokens) numpy arrays.

    The model calls attention with need_weights=False, so the weights are recomputed from each
    layer's input with a forward pre-hook; the model itself is not modified.
    """
    import torch

    captured = []
    attentions = [layer.self_attention for layer in model.encoder.layers]
    hooks = [attn.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
             for attn in attentions]
    try:
        with torch.inference_mode():
            model(tensor.unsqueeze(0))
            weights = [attn(q, q, q, need_weights=True, average_attn_weights=True)[1][0].numpy()
                       for attn, q in zip(attentions, captured)]
    finally:
        for hook in hooks:
            hook.remove()
    return weights


def rollout_heat(attentions):
    """Attention rollout -> (13, 9) array in [0, 1] of how much each patch fed the class token."""
    n = attentions[0].shape[-1]
    identity = np.eye(n)
    result = identity
    for attn in attentions:
        layer = 0.5 * attn + 0.5 * identity  # residual connection
        layer = layer / layer.sum(axis=-1, keepdims=True)
        result = layer @ result
    heat = result[0, 1:]  # class token -> patches
    if heat.size != GRID_ROWS * GRID_COLS:
        raise ValueError(f'expected {GRID_ROWS * GRID_COLS} patches, got {heat.size}')
    heat = heat.reshape(GRID_ROWS, GRID_COLS)
    span = heat.max() - heat.min()
    return (heat - heat.min()) / span if span > 0 else np.zeros_like(heat)


def recent_share(heat, cols=RECENT_COLS):
    """Fraction of the total attention that lands on the right-most `cols` patch columns."""
    total = float(heat.sum())
    return float(heat[:, -cols:].sum()) / total if total > 0 else 0.0


def upscale(image, scale=3):
    """Nearest-neighbour enlargement of a (H, W, 3) uint8 image."""
    h, w = image.shape[:2]
    return np.asarray(Image.fromarray(image).resize((w * scale, h * scale), Image.Resampling.NEAREST))


def overlay(image, heat, scale=3, alpha=0.55):
    """Blend the heatmap over a (H, W, 3) uint8 crop; returns an upscaled uint8 image."""
    h, w = image.shape[:2]
    base = Image.fromarray(image).resize((w * scale, h * scale), Image.Resampling.NEAREST)
    heat_img = Image.fromarray((heat * 255).astype(np.uint8)).resize(base.size, Image.Resampling.BICUBIC)
    values = np.asarray(heat_img, dtype=float) / 255
    color = np.stack([np.interp(values, _STOPS, _COLORS[:, c]) for c in range(3)], axis=-1)
    strength = (alpha * values)[..., None]  # cold regions stay close to the original picture
    blended = np.asarray(base, dtype=float) * (1 - strength) + color * strength
    return np.clip(blended, 0, 255).astype(np.uint8)
