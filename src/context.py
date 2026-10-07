"""Trend and volume context around a signal.

A reversal pattern means more after a move in the opposite direction, and on unusual volume.
These helpers describe that context in plain terms; they do not change the model's output.
"""
import numpy as np

TREND_LOOKBACK = 10  # candles before the 3-candle pattern window
TREND_THRESHOLD = 1.5  # net move, in average candle ranges, that counts as a trend
VOLUME_LOOKBACK = 20
HIGH_VOLUME, LOW_VOLUME = 1.5, 0.5  # multiples of the average volume

BULLISH = ('bullish_engulfing', 'morning_star')
BEARISH = ('bearish_engulfing', 'evening_star')


def prior_trend(df):
    """('up' | 'down' | 'sideways' | 'n/a', net move in average candle ranges) before the last 3 candles."""
    needed = TREND_LOOKBACK + 3
    if len(df) < needed:
        return 'n/a', 0.0
    window = df.iloc[-needed:-3]
    avg_range = float((window['High'] - window['Low']).mean())
    net = float(window['Close'].iloc[-1] - window['Close'].iloc[0])
    if avg_range <= 0:
        return 'sideways', 0.0
    score = net / avg_range
    if score >= TREND_THRESHOLD:
        return 'up', score
    if score <= -TREND_THRESHOLD:
        return 'down', score
    return 'sideways', score


def volume_state(df):
    """('high' | 'normal' | 'low' | 'n/a', last volume / average of the previous 20)."""
    if len(df) < 2 or 'Volume' not in df:
        return 'n/a', 0.0
    previous = df['Volume'].iloc[-1 - VOLUME_LOOKBACK:-1]
    avg = float(np.nan_to_num(previous.mean()))
    last = float(np.nan_to_num(df['Volume'].iloc[-1]))
    if avg <= 0:
        return 'n/a', 0.0
    ratio = last / avg
    return ('high' if ratio >= HIGH_VOLUME else 'low' if ratio <= LOW_VOLUME else 'normal'), ratio


def context_fit(pattern, trend):
    """'fits' | 'against' | 'neutral' for a pattern given the trend before it."""
    if trend in ('n/a', 'sideways'):
        return 'neutral'
    if pattern in BULLISH:
        return 'fits' if trend == 'down' else 'against'
    if pattern in BEARISH:
        return 'fits' if trend == 'up' else 'against'
    return 'fits'  # doji: indecision after any strong move


def describe(df, pattern):
    """Everything the UI shows about context for `pattern` at the end of `df`."""
    trend, score = prior_trend(df)
    volume, ratio = volume_state(df)
    return {'trend': trend, 'trend_score': score, 'volume': volume, 'volume_ratio': ratio,
            'fit': context_fit(pattern, trend)}
