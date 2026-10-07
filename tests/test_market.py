import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import pytest

import market


def df_ending(ts):
    times = pd.date_range(end=ts, periods=25, freq='min', tz='UTC')
    return pd.DataFrame({'Datetime': times, 'Open': 1.0, 'High': 1.0, 'Low': 1.0, 'Close': 1.0, 'Volume': 1})


def test_closed_only_drops_forming_candle_per_timeframe():
    last = pd.Timestamp('2026-01-05 15:00', tz='UTC')
    df = df_ending(last)
    assert len(market.closed_only(df, 60, now=datetime(2026, 1, 5, 15, 0, 30, tzinfo=timezone.utc))) == 24
    assert len(market.closed_only(df, 60, now=datetime(2026, 1, 5, 15, 1, 5, tzinfo=timezone.utc))) == 25
    # a 5-minute candle that started at 15:00 is still forming at 15:03
    assert len(market.closed_only(df, 300, now=datetime(2026, 1, 5, 15, 3, tzinfo=timezone.utc))) == 24


def test_market_status_scales_with_timeframe():
    last = pd.Timestamp('2026-01-05 15:00', tz='UTC')
    df = df_ending(last)
    now = datetime(2026, 1, 5, 15, 40, tzinfo=timezone.utc)
    assert market.market_status(df, 60, now=now)[0] == 'closed'
    assert market.market_status(df, 3600, now=now)[0] == 'live'  # 40 min is inside an hourly bar's tolerance


def test_every_timeframe_has_a_config():
    for name, cfg in market.TIMEFRAMES.items():
        assert cfg['seconds'] > 0 and cfg['interval'] and cfg['period'], name


def test_unknown_timeframe_is_a_clear_error():
    with pytest.raises(market.MarketDataError, match='Unknown timeframe'):
        market.fetch_candles('AAPL', '3m')


def test_format_time():
    ts = pd.Timestamp('2026-01-05 15:07')
    assert market.format_time(ts, '1m') == '15:07'
    assert market.format_time(ts, '1h') == '01-05 15:07'
    assert market.format_time(ts, '1d') == '2026-01-05'
