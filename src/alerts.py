"""Alert rules, an in-memory feed with de-duplication, and optional Telegram delivery.

Alerts fire only for rows from a live market, so replayed or stale data never notifies you.
They are evaluated whenever the app refreshes (main view in Live mode, or a scanner run),
so the page has to stay open.
"""
import json
import os
import threading
import urllib.parse
import urllib.request
from collections import deque
from dataclasses import dataclass
from datetime import datetime

TELEGRAM_TIMEOUT_S = 5


@dataclass(frozen=True)
class Settings:
    min_confidence: float = 0.85
    require_agree: bool = True
    require_fit: bool = False


def matches(row, settings):
    """True if a scanner-style row deserves an alert under `settings`."""
    if row.get('error') or row.get('market') != 'live':
        return False
    if row['confidence'] < settings.min_confidence:
        return False
    if settings.require_agree and not row['agrees']:
        return False
    if settings.require_fit and row['fit'] != 'fits':
        return False
    return True


def format_message(row):
    rules = ', '.join(row['rules']) or 'none'
    return (f"{row['ticker']} {row['timeframe']}: {row['prediction']} {row['confidence']:.0%} "
            f"(rules: {rules}; trend {row['trend']}, {row['fit']}) @ {row['close']:,.2f}")


class AlertFeed:
    """Newest-first list of triggered alerts; each ticker/timeframe/candle/pattern alerts once."""

    def __init__(self, keep=50):
        self._seen = set()
        self._items = deque(maxlen=keep)
        self.total = 0
        self._lock = threading.Lock()

    def add(self, row):
        """Record the alert; returns its message, or None if this candle already alerted."""
        key = (row['ticker'], row['timeframe'], str(row['candle_time']), row['prediction'])
        with self._lock:
            if key in self._seen:
                return None
            self._seen.add(key)
            message = format_message(row)
            self._items.appendleft({'time': datetime.now().strftime('%H:%M:%S'), 'message': message})
            self.total += 1
            return message

    def recent(self, n=10):
        with self._lock:
            return list(self._items)[:n]


def telegram_credentials(env=os.environ):
    """(token, chat_id) from the environment, or None if either is missing."""
    token, chat = env.get('TELEGRAM_BOT_TOKEN', '').strip(), env.get('TELEGRAM_CHAT_ID', '').strip()
    return (token, chat) if token and chat else None


def send_telegram(text, token, chat_id, opener=urllib.request.urlopen):
    """-> (ok, error). Never raises: a failed notification must not break the app."""
    url = f'https://api.telegram.org/bot{token}/sendMessage'
    data = urllib.parse.urlencode({'chat_id': chat_id, 'text': text}).encode()
    try:
        with opener(urllib.request.Request(url, data=data), timeout=TELEGRAM_TIMEOUT_S) as response:
            body = json.loads(response.read().decode() or '{}')
    except Exception as exc:
        return False, (type(exc).__name__ + (f': {exc}' if str(exc) else '')).replace(token, '***')
    return (True, '') if body.get('ok', True) else (False, body.get('description', 'Telegram rejected the message'))
