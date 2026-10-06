import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd

import market
import scanner


def fake_df(n=25):
    times = pd.date_range('2020-01-02 09:30', periods=n, freq='min', tz='America/New_York')
    base = 100 + np.sin(np.arange(n))
    return pd.DataFrame({'Datetime': times, 'Open': base, 'High': base + 0.5, 'Low': base - 0.5,
                         'Close': base + 0.1, 'Volume': 1000})


def fake_predict(image):
    assert image.shape == (500, 700, 3)
    return [0.1, 0.6, 0.1, 0.1, 0.1]


def test_parse_tickers_cleans_input():
    tickers, rejected, truncated = scanner.parse_tickers(' aapl, MSFT;aapl  tsla  bad$$ ')
    assert tickers == ['AAPL', 'MSFT', 'TSLA']
    assert rejected == ['bad$$']
    assert not truncated


def test_parse_tickers_caps_the_list():
    names = [f'T{i}' for i in range(scanner.MAX_TICKERS + 5)]
    tickers, _, truncated = scanner.parse_tickers(','.join(names))
    assert len(tickers) == scanner.MAX_TICKERS and truncated


def test_analyze_builds_a_row():
    row = scanner.analyze('AAPL', fake_df(), fake_predict)
    assert row['prediction'] == 'bullish_engulfing' and row['confidence'] == 0.6
    assert row['market'] == 'closed'  # candles are from 2020
    assert isinstance(row['agrees'], bool)


def test_scan_sorts_by_confidence_and_isolates_failures():
    def fetch(ticker):
        if ticker == 'BAD':
            raise market.MarketDataError('No 1-minute data for "BAD".')
        return fake_df()

    confidences = iter([0.4, 0.9])

    def predict(image):
        c = next(confidences)
        return [c, (1 - c) / 4, (1 - c) / 4, (1 - c) / 4, (1 - c) / 4]

    rows = scanner.scan(['AAA', 'BAD', 'BBB'], predict, fetch=fetch)
    assert [r['ticker'] for r in rows] == ['BBB', 'AAA', 'BAD']
    assert rows[0]['confidence'] > rows[1]['confidence']
    assert 'No 1-minute data' in rows[2]['error']


def test_scan_reports_too_few_candles():
    rows = scanner.scan(['TINY'], fake_predict, fetch=lambda t: fake_df(10))
    assert 'closed candles' in rows[0]['error']
