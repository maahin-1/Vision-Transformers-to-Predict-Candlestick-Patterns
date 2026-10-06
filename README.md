# Vision Transformers to Predict Candlestick Patterns

A Vision Transformer (ViT) that recognises candlestick patterns on a live trading chart. The predictor captures your screen, crops the chart area, classifies the most recent candles, and shows the class probabilities in a small preview window.

**Patterns detected:** doji, bullish engulfing, bearish engulfing, morning star, evening star.

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
pip install uv
git clone https://github.com/maahinosahan/Vision-Transformers-to-Predict-Candlestick-Patterns.git
cd Vision-Transformers-to-Predict-Candlestick-Patterns
uv sync
uv run download_model.py
```

`download_model.py` fetches the trained checkpoint (`checkpoints/25_model.pt`, about 98 MB) from this repository's GitHub Releases. If you would rather train your own, skip it and see **Training**.

## Running the live predictor

1. In one terminal, start the chart (Yahoo Finance 1-minute data replayed in a Dash app at http://127.0.0.1:8050):
   ```bash
   uv run src/utils/candlesticks.py
   ```
2. Open that page in a browser, then in a second terminal run:
   ```bash
   uv run candlestick_prediction.py
   ```

Options:

| Flag | Default | Meaning |
|---|---|---|
| `--monitor N` | `1` | Monitor to capture (1 = primary) |
| `--fps N` | `5` | Maximum frames per second |
| `--full-overlay` | off | Draw the original full-screen overlay (for screen recordings) |

Press `q` or `Esc` in the preview window, or `Ctrl+C` in the terminal, to stop.

The crop assumes the chart sits where it does in the Dash app. If your layout differs, adjust the `A.Crop(...)` calls in `candlestick_prediction.py`.

## Training

```bash
mkdir checkpoints
uv run src/train.py
```

Training reads `data/train_data` and `data/test_data` (images plus `labels.csv`) and saves a checkpoint every 5 epochs to `checkpoints/`. Evaluate one with `uv run src/test.py`, which writes `results.png`.

## Project layout

```
candlestick_prediction.py   live predictor
download_model.py           fetch the trained checkpoint
src/model.py                ViT architecture
src/data.py                 dataset and augmentations
src/train.py, src/test.py   training and evaluation
src/utils/                  chart server, labelling and data generation tools
data/                       labelled training and test images
```

## Model

Input is the cropped chart region (72 x 104 px, eight candles), split into 8 x 8 patches. A class token and sinusoidal positional embeddings feed 3 transformer encoder layers (1024-dim, 8 heads), followed by a 5-way classification head.

## Troubleshooting

- **System slows down or the predictor lags:** the model runs on CPU. Close heavy apps, lower `--fps`, and make sure the preview window is not covering the chart.
- **Wrong predictions:** the model was trained on 4K screenshots of one chart style. Different resolutions, themes or layouts need a different crop, or retraining.
- **Download fails:** get `25_model.pt` from the Releases page and put it in `checkpoints/`.

## Credits and license

Maintained by Maahin Bir Singh Osahan ([@maahinosahan](https://github.com/maahinosahan)).

Based on [nicknochnack/ViTCandlesticks](https://github.com/nicknochnack/ViTCandlesticks) by Nicholas Renotte. Released under the MIT License; see [LICENSE](LICENSE).
