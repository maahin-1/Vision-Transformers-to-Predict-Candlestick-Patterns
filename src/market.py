"""Yahoo Finance access: ticker validation, candles for several timeframes, and market status."""
import re
from datetime import datetime, timezone

import yfinance as yf

TICKER_RE = re.compile(r'^[A-Za-z0-9.\-=^]{1,15}$')
MIN_CANDLES = 20  # the model needs 20 candles on the chart

# interval/period are Yahoo's. session=True keeps only the latest trading day (intraday charts);
# otherwise the most recent `rows` candles are kept. Yahoo limits: 1m ~7 days, 5m-1h ~60-730 days.
TIMEFRAMES = {
    '1m': dict(interval='1m', period='5d', seconds=60, session=True, rows=None),
    '5m': dict(interval='5m', period='5d', seconds=300, session=True, rows=None),
    '15m': dict(interval='15m', period='1mo', seconds=900, session=False, rows=150),
    '1h': dict(interval='1h', period='6mo', seconds=3600, session=False, rows=150),
    '1d': dict(interval='1d', period='2y', seconds=86400, session=False, rows=150),
}
DEFAULT_TIMEFRAME = '1m'
MIN_LIVE_AGE_MIN = 5  # a feed this fresh counts as live whatever the timeframe


class MarketDataError(Exception):
    """Raised with a user-readable message when data cannot be loaded."""


def normalize_ticker(text):
    ticker = (text or '').strip().upper()
    if not TICKER_RE.match(ticker):
        raise MarketDataError('Enter a ticker like AAPL, AMZN, TSLA or RELIANCE.NS')
    return ticker


def timeframe_seconds(timeframe):
    return TIMEFRAMES[timeframe]['seconds']


def fetch_candles(ticker, timeframe=DEFAULT_TIMEFRAME):
    """Recent candles. Columns: Datetime (tz-aware), Open, High, Low, Close, Volume."""
    if timeframe not in TIMEFRAMES:
        raise MarketDataError(f'Unknown timeframe "{timeframe}". Use one of: {", ".join(TIMEFRAMES)}')
    cfg = TIMEFRAMES[timeframe]
    try:
        raw = yf.Ticker(ticker).history(period=cfg['period'], interval=cfg['interval'])
    except Exception as exc:  # network errors, rate limits, delisted symbols
        raise MarketDataError(f'Could not reach Yahoo Finance for {ticker}: {exc}') from exc
    if raw is None or raw.empty:
        raise MarketDataError(f'No {timeframe} data for "{ticker}". Check the symbol '
                              f'(suffix like .NS or .L for non-US markets).')

    df = raw.reset_index().rename(columns={raw.index.name or 'index': 'Datetime'})
    df = df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']].dropna(subset=['Open', 'High', 'Low', 'Close'])
    if df['Datetime'].dt.tz is None:
        df['Datetime'] = df['Datetime'].dt.tz_localize('UTC')
    if cfg['session']:
        last_day = df['Datetime'].iloc[-1].date()
        df = df[df['Datetime'].dt.date == last_day]
    else:
        df = df.tail(cfg['rows'])
    df = df.reset_index(drop=True)
    if len(df) < MIN_CANDLES:
        raise MarketDataError(f'Only {len(df)} {timeframe} candles available for {ticker}; need {MIN_CANDLES}.')
    return df


def closed_only(df, seconds=60, now=None):
    """Drop the newest candle if its period has not finished yet."""
    now = now or datetime.now(timezone.utc)
    last_start = df['Datetime'].iloc[-1]
    if (now - last_start.to_pydatetime()).total_seconds() < seconds:
        return df.iloc[:-1].reset_index(drop=True)
    return df


def market_status(df, seconds=60, now=None):
    """('live' | 'closed', minutes since the newest candle started)."""
    now = now or datetime.now(timezone.utc)
    age = (now - df['Datetime'].iloc[-1].to_pydatetime()).total_seconds() / 60
    return ('live' if age <= max(MIN_LIVE_AGE_MIN, 1.5 * seconds / 60) else 'closed'), age


def format_time(ts, timeframe):
    """Compact candle label: time for 1m/5m, date+time for 15m/1h, date for daily."""
    if timeframe == '1d':
        return f'{ts:%Y-%m-%d}'
    if timeframe in ('15m', '1h'):
        return f'{ts:%m-%d %H:%M}'
    return f'{ts:%H:%M}'
