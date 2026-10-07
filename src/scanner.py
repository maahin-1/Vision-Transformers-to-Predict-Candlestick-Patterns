"""Watchlist scanner: fetch several tickers, run the ViT on each, cross-check with the rules."""
import re
from concurrent.futures import ThreadPoolExecutor

import context
import market
import render
import rules

MAX_TICKERS = 20
FETCH_WORKERS = 4


def parse_tickers(text):
    """-> (tickers, rejected, truncated). Upper-cased, de-duplicated, capped at MAX_TICKERS."""
    tickers, rejected = [], []
    for part in re.split(r'[,;\s]+', text or ''):
        if not part:
            continue
        try:
            ticker = market.normalize_ticker(part)
        except market.MarketDataError:
            rejected.append(part)
            continue
        if ticker not in tickers:
            tickers.append(ticker)
    return tickers[:MAX_TICKERS], rejected, len(tickers) > MAX_TICKERS


def analyze(ticker, df, predict_fn, timeframe=market.DEFAULT_TIMEFRAME):
    """Prediction row for one ticker from its candle DataFrame."""
    seconds = market.timeframe_seconds(timeframe)
    closed = market.closed_only(df, seconds)
    if len(closed) < render.N_CANDLES:
        raise market.MarketDataError(f'only {len(closed)} closed candles')
    data = closed.iloc[-render.N_CANDLES:]
    image = render.render_chart(data['Open'], data['High'], data['Low'], data['Close'], render.training_style_times())
    probs = predict_fn(image)
    top = max(range(len(probs)), key=probs.__getitem__)
    found = rules.detect_frame(closed)
    state, age = market.market_status(df, seconds)
    ctx = context.describe(closed, rules.CLASSES[top])
    return {'ticker': ticker, 'error': '', 'timeframe': timeframe, 'candle_time': closed['Datetime'].iloc[-1],
            'close': float(closed['Close'].iloc[-1]), 'prediction': rules.CLASSES[top], 'confidence': probs[top],
            'rules': found, 'agrees': rules.CLASSES[top] in found, 'market': state, 'age_min': age,
            'trend': ctx['trend'], 'volume': ctx['volume'], 'volume_ratio': ctx['volume_ratio'], 'fit': ctx['fit']}


def _fetch(ticker, fetch, timeframe):
    try:
        return ticker, fetch(ticker, timeframe)
    except Exception as exc:  # one bad ticker must not sink the scan
        return ticker, exc


def scan(tickers, predict_fn, fetch=market.fetch_candles, timeframe=market.DEFAULT_TIMEFRAME):
    """Rows for every ticker: successful ones sorted by confidence, then failures."""
    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
        fetched = list(pool.map(lambda t: _fetch(t, fetch, timeframe), tickers))
    ok, failed = [], []
    for ticker, result in fetched:
        try:
            if isinstance(result, Exception):
                raise result
            ok.append(analyze(ticker, result, predict_fn, timeframe))
        except Exception as exc:
            failed.append({'ticker': ticker, 'error': str(exc) or type(exc).__name__})
    ok.sort(key=lambda r: r['confidence'], reverse=True)
    return ok + failed
