import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

import render
import rules


def c(o, h, l, cl):
    return (o, h, l, cl)


def test_doji():
    assert rules.detect([c(10.0, 11.0, 9.0, 10.05)]) == ['doji']
    assert rules.detect([c(10.0, 11.0, 9.0, 10.9)]) == []


def test_bullish_engulfing():
    assert 'bullish_engulfing' in rules.detect([c(10, 10.2, 9.0, 9.2), c(9.1, 10.6, 9.0, 10.5)])


def test_bearish_engulfing():
    assert 'bearish_engulfing' in rules.detect([c(9.2, 10.2, 9.1, 10.0), c(10.1, 10.2, 8.8, 9.0)])


def test_morning_star():
    found = rules.detect([c(12, 12.1, 9.9, 10.0), c(9.7, 9.9, 9.4, 9.8), c(9.9, 11.8, 9.8, 11.5)])
    assert 'morning_star' in found


def test_evening_star():
    found = rules.detect([c(10, 12.1, 9.9, 12.0), c(12.3, 12.6, 12.2, 12.4), c(12.2, 12.3, 10.3, 10.5)])
    assert 'evening_star' in found


def test_flat_market_has_no_patterns():
    assert rules.detect([c(10, 10, 10, 10)] * 3) == []


def test_render_shape_and_flat_prices():
    flat = np.full(render.N_CANDLES, 100.0)
    img = render.render_chart(flat, flat, flat, flat)
    assert img.shape == (500, 700, 3) and img.dtype == np.uint8
