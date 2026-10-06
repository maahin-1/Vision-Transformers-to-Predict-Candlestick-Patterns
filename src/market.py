"""Yahoo Finance access: ticker validation, 1-minute candles, and market status."""
import re
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

TICKER_RE = re.compile(r'^[A-Za-z0-9.\-=^]{1,15}$')
MIN_CANDLES = 20  # the model needs 20 candles on the chart
LIVE_MAX_AGE_MIN = 5  # newest closed candle older than this means the market is not live


class MarketDataError(Exception):
    """Raised with a user-readable message when data cannot be loaded."""


def normalize_ticker(text):
    ticker = (text or '').strip().upper()
    if not TICKER_RE.match(ticker):
        raise MarketDataError('Enter a ticker like AAPL, AMZN, TSLA or RELIANCE.NS')
    return ticker


def fetch_candles(ticker):
    """Latest trading session of 1-minute candles. Columns: Datetime, Open, High, Low, Close, Volume."""
    try:
        raw = yf.Ticker(ticker).history(period='5d', interval='1m')
    except Exception as exc:  # network errors, rate limits, delisted symbols
        raise MarketDataError(f'Could not reach Yahoo Finance for {ticker}: {exc}') from exc
    if raw is None or raw.empty:
        raise MarketDataError(f'No 1-minute data for "{ticker}". Check the symbol (suffix like .NS or .L for non-US markets).')

    df = raw.reset_index().rename(columns={raw.index.name or 'index': 'Datetime'})
    df = df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']].dropna(subset=['Open', 'High', 'Low', 'Close'])
    last_day = df['Datetime'].iloc[-1].date()
    df = df[df['Datetime'].dt.date == last_day].reset_index(drop=True)
    if len(df) < MIN_CANDLES:
        raise MarketDataError(f'Only {len(df)} candles available for {ticker} in the latest session; need {MIN_CANDLES}.')
    return df


def closed_only(df, now=None):
    """Drop the newest candle if its minute has not finished yet."""
    now = now or datetime.now(timezone.utc)
    last_start = df['Datetime'].iloc[-1]
    if (now - last_start.to_pydatetime()).total_seconds() < 60:
        return df.iloc[:-1].reset_index(drop=True)
    return df


def market_status(df, now=None):
    """('live' | 'closed', minutes since the newest candle)."""
    now = now or datetime.now(timezone.utc)
    age = (now - df['Datetime'].iloc[-1].to_pydatetime()).total_seconds() / 60
    return ('live' if age <= LIVE_MAX_AGE_MIN else 'closed'), age
