import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd

import context


def frame(closes, volumes=None, spread=1.0):
    n = len(closes)
    return pd.DataFrame({'Open': closes, 'High': [c + spread / 2 for c in closes],
                         'Low': [c - spread / 2 for c in closes], 'Close': closes,
                         'Volume': volumes or [1000] * n})


def test_prior_trend_down_up_sideways():
    assert context.prior_trend(frame([100 - i for i in range(30)]))[0] == 'down'
    assert context.prior_trend(frame([100 + i for i in range(30)]))[0] == 'up'
    assert context.prior_trend(frame([100 + (i % 2) * 0.1 for i in range(30)]))[0] == 'sideways'


def test_prior_trend_needs_enough_candles():
    assert context.prior_trend(frame([100] * 5)) == ('n/a', 0.0)


def test_volume_state():
    assert context.volume_state(frame([1] * 21, [100] * 20 + [300]))[0] == 'high'
    assert context.volume_state(frame([1] * 21, [100] * 20 + [30]))[0] == 'low'
    assert context.volume_state(frame([1] * 21, [100] * 21))[0] == 'normal'
    assert context.volume_state(frame([1] * 21, [0] * 21))[0] == 'n/a'


def test_context_fit():
    assert context.context_fit('bullish_engulfing', 'down') == 'fits'
    assert context.context_fit('bullish_engulfing', 'up') == 'against'
    assert context.context_fit('evening_star', 'up') == 'fits'
    assert context.context_fit('morning_star', 'sideways') == 'neutral'
    assert context.context_fit('doji', 'down') == 'fits'
    assert context.context_fit('doji', 'n/a') == 'neutral'
