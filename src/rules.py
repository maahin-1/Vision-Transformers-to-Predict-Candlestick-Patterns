"""Plain-OHLC definitions of the five patterns, used to cross-check the ViT.

Each function takes the most recent candles as (open, high, low, close) tuples, oldest first.
These are common textbook thresholds, not the labelling used for training, so a mismatch is
a prompt to look at the chart, not proof the model is wrong.
"""

DOJI_BODY_RATIO = 0.1  # body <= 10% of the candle's high-low range
SMALL_BODY_RATIO = 0.5  # star middle body <= 50% of the first candle's body


def _body(c):
    return abs(c[3] - c[0])


def _range(c):
    return c[1] - c[2]


def _bull(c):
    return c[3] > c[0]


def _bear(c):
    return c[3] < c[0]


def is_doji(c):
    return _range(c) > 0 and _body(c) <= DOJI_BODY_RATIO * _range(c)


def is_bullish_engulfing(prev, cur):
    return _bear(prev) and _bull(cur) and cur[0] <= prev[3] and cur[3] >= prev[0] and _body(cur) > _body(prev)


def is_bearish_engulfing(prev, cur):
    return _bull(prev) and _bear(cur) and cur[0] >= prev[3] and cur[3] <= prev[0] and _body(cur) > _body(prev)


def is_morning_star(a, b, c):
    mid = (a[0] + a[3]) / 2
    return (_bear(a) and _body(b) <= SMALL_BODY_RATIO * _body(a) and max(b[0], b[3]) <= a[3]
            and _bull(c) and c[3] > mid)


def is_evening_star(a, b, c):
    mid = (a[0] + a[3]) / 2
    return (_bull(a) and _body(b) <= SMALL_BODY_RATIO * _body(a) and min(b[0], b[3]) >= a[3]
            and _bear(c) and c[3] < mid)


def detect(candles):
    """Names of every pattern the newest candles satisfy, in the model's class names."""
    found = []
    if len(candles) >= 1 and is_doji(candles[-1]):
        found.append('doji')
    if len(candles) >= 2:
        if is_bullish_engulfing(*candles[-2:]):
            found.append('bullish_engulfing')
        if is_bearish_engulfing(*candles[-2:]):
            found.append('bearish_engulfing')
    if len(candles) >= 3:
        if is_morning_star(*candles[-3:]):
            found.append('morning_star')
        if is_evening_star(*candles[-3:]):
            found.append('evening_star')
    return found
