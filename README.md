# Vision Transformers to Predict Candlestick Patterns

A Vision Transformer (ViT) that recognises candlestick patterns on a live stock chart. Type a ticker, and the app pulls 1-minute candles from Yahoo Finance, draws the last 20 as a chart, and classifies the most recent eight candles. A plain-OHLC rule check runs beside the model so you can see when they disagree.

**Patterns detected:** doji, bullish engulfing, bearish engulfing, morning star, evening star.

![The app tracking AMZN: candlestick chart, ViT prediction, rule check, model input and OHLCV table](images/app.png)

*Here the model says `bearish_engulfing` at 93.9% while the rule check finds no pattern. That kind of disagreement is exactly what the rule check is for; see **Caveats**.*

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
pip install uv
git clone https://github.com/maahin-1/Vision-Transformers-to-Predict-Candlestick-Patterns.git
cd Vision-Transformers-to-Predict-Candlestick-Patterns
uv sync
uv run download_model.py
```

`download_model.py` fetches the trained checkpoint (`checkpoints/25_model.pt`, about 98 MB) from this repository's GitHub Releases. If you would rather train your own, skip it and see **Training**.

## Running the app

```bash
uv run src/app.py
```

Open http://127.0.0.1:8050, type a ticker (`AAPL`, `TSLA`, `RELIANCE.NS`, ...), pick a timeframe and press **Track**. Each browser tab has its own state, so you can watch different tickers or modes side by side.

| Mode | What it does |
|---|---|
| **Live** | Polls Yahoo Finance every 30 s and predicts on the newest *closed* candles. Yahoo data can be delayed by minutes depending on the exchange. When the market is closed it shows the last session, labelled "MARKET CLOSED". |
| **Replay** | Replays the loaded candles one per second. Use it after hours or to test. |

**Timeframes:** `1m`, `5m`, `15m`, `1h`, `1d`. Yahoo keeps 1-minute data for about a week, 5-minute to 1-hour data for weeks to months, and daily data for years. The model was trained on 1-minute charts, but it only sees the autoscaled shape of the last 20 candles, so the picture is drawn identically for every timeframe. Daily candles count as closed once their day has ended.

The page shows:

- the candlestick chart, with the eight candles the model reads highlighted;
- the model's probability for each class (a "weak" tag appears below 50%);
- a rule check, using textbook OHLC definitions, saying whether it agrees with the model;
- "What the model sees", the exact 72 x 104 px crop fed to the network, with an optional **attention heatmap** (warmer = the patch mattered more to the class token) and a note on how much attention falls on the newest candles;
- trend and volume context: the move over the 10 candles before the pattern (up/down/sideways, measured in average candle ranges), the last candle's volume against its 20-candle average, and whether the pattern fits that trend (a bullish reversal after a downtrend fits; after an uptrend it goes against it);
- the OHLCV table of those eight candles, so you can compare against Yahoo or your broker.
- a **watchlist scanner**: enter up to 20 tickers (`AAPL, MSFT, TSLA`) and press **Scan** to see each one's current pattern, confidence, rule agreement and last candle, sorted by confidence. Rows where the model and the rule check agree are tinted green, and **Open** loads a ticker into the main view. It scans on the timeframe selected above, shows each ticker's trend, volume and context, and *Only signals that fit the trend* hides the rest. Tick *Auto-refresh* to rescan every 30 s. A bad or unknown ticker only affects its own row;
- an **Alerts** card: set a minimum confidence, and whether the rules must agree and the signal must fit the trend. Matching signals appear in an alert feed and can also raise a browser notification or a Telegram message. Alerts fire only for *live* signals (never Replay or a closed market), once per candle, and only while the page is open (see **Alerts setup**);
- a detection history (newest 10 rows) with a **Download CSV** button. Every new candle's prediction is appended to `logs/detections.csv` (time, ticker, mode, timeframe, close, model class and confidence, rule patterns, whether they agree), so you can review how the model behaved later. The `logs/` folder is git-ignored.

Press `Ctrl+C` in the terminal to stop.

### Attention heatmap

Tick *Show attention heatmap* under "What the model sees". It uses attention rollout (`src/explain.py`): the attention heads are averaged, the residual connections added, and the three encoder layers multiplied together, giving the class token's weight on each 8 x 8 patch. It is computed on demand and the model is not modified. On generated patterns about 60-80% of the attention lands on the right-most third of the crop, i.e. the newest candles, which is where these patterns are defined. Treat it as an indication of where the model looks, not proof of why it decided.

### Alerts setup

The feed and browser notifications work out of the box (tick *browser notification* and allow the permission prompt). For Telegram, create a bot with [@BotFather](https://t.me/BotFather), send it a message, find your chat id (for example via `https://api.telegram.org/bot<TOKEN>/getUpdates`), and set both variables before starting the app:

```powershell
$env:TELEGRAM_BOT_TOKEN = "123456:ABC..."
$env:TELEGRAM_CHAT_ID = "123456789"
uv run src/app.py
```

The token is read from the environment only and is never written to disk or the logs. Then tick *Telegram* in the Alerts card.

### How the prediction works

The model was trained on screenshots of a Plotly chart showing 20 candles. The app draws the same kind of picture from the candle data (`src/render.py`) and runs it through the same resize and crop as training, so it does not read your screen and does not depend on window position. On 150 freshly generated pattern charts the rendered images were classified 100% correctly; the model scores 99.5% on its own labelled test screenshots.

### Caveats

- The training data is synthetic (randomly generated candles shaped into each pattern), so real markets can give confident but wrong answers. Check the rule row and the table.
- The model has no "no pattern" class. It always picks one of the five.
- Educational project, not financial advice.

## Optional: screen-capture predictor

`candlestick_prediction.py` is the original approach: it captures a monitor, crops a fixed region and shows a small preview window. It only works when a chart in the training layout fills that region (for example the Plotly chart in a maximised browser at 100% zoom).

```bash
uv run candlestick_prediction.py [--monitor N] [--fps N] [--full-overlay]
```

Press `q` or `Esc` in the preview window, or `Ctrl+C`, to stop.

## Training

```bash
mkdir checkpoints
uv run src/train.py
```

Training reads `data/train_data` and `data/test_data` (images plus `labels.csv`) and saves a checkpoint every 5 epochs to `checkpoints/`. Evaluate one with `uv run src/test.py`, which writes `results.png`. `src/utils/datageneration_*.py` generate the synthetic charts and `src/utils/annotator.py` helps label them.

## Tests

```bash
uv run pytest
```

## Project layout

```
src/app.py                  Dash app: ticker input, chart, prediction, rule check
src/market.py               Yahoo Finance data and market status
src/history.py              CSV log of every prediction
src/context.py              trend and volume context for a signal
src/alerts.py               alert rules, feed and Telegram delivery
src/explain.py              attention-rollout heatmap
src/sessions.py             per-browser-tab state
src/scanner.py              watchlist scan (parallel fetch, sequential inference)
src/render.py               draws the 20-candle chart image the model expects
src/predictor.py            loads the ViT and runs inference
src/rules.py                OHLC definitions of the five patterns
src/model.py                ViT architecture
src/data.py                 dataset and augmentations
src/train.py, src/test.py   training and evaluation
src/utils/                  data generation and labelling tools
candlestick_prediction.py   optional screen-capture predictor
download_model.py           fetch the trained checkpoint
data/                       labelled training and test images
tests/                      unit tests
```

## Model

Input is the cropped chart region (72 x 104 px, eight candles), split into 8 x 8 patches. A class token and sinusoidal positional embeddings feed 3 transformer encoder layers (1024-dim, 8 heads), followed by a 5-way classification head.

## Troubleshooting

- **"No 1-minute data":** check the symbol. Non-US markets need a suffix (`.NS`, `.L`, `.TO`). Yahoo only serves 1-minute data for roughly the last week.
- **Live mode shows an old session:** the market is closed or the exchange feed is delayed. Switch to Replay.
- **Download fails:** get `25_model.pt` from the Releases page and put it in `checkpoints/`.
- **System slows down:** inference runs on CPU. Close heavy apps; the app itself uses a few hundred MB.

## Credits and license

Maintained by Maahin Bir Singh Osahan ([@maahin-1](https://github.com/maahin-1)).

Released under the MIT License; see [LICENSE](LICENSE).
