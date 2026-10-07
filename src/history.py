"""Append-only CSV log of every prediction the app makes (one row per ticker/mode/candle)."""
import csv
import os
from collections import deque
from datetime import datetime, timezone

FIELDS = ['logged_at', 'ticker', 'mode', 'timeframe', 'candle_time', 'close', 'prediction', 'confidence', 'rule_patterns', 'agrees']
DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs', 'detections.csv')


class HistoryLog:
    def __init__(self, path=DEFAULT_PATH, keep=50):
        self.path = path
        self._seen = set()
        self._recent = deque(maxlen=keep)
        self._loaded = False

    def _load(self):
        """Pick up rows from earlier runs so the recent list survives a restart."""
        self._loaded = True
        if not os.path.exists(self.path):
            return
        with open(self.path, newline='', encoding='utf-8') as f:
            header = next(csv.reader(f), [])
        if header != FIELDS:  # written by an older version: keep it aside rather than mix column layouts
            os.replace(self.path, self.path.replace('.csv', f'.legacy-{datetime.now():%Y%m%d%H%M%S}.csv'))
            return
        with open(self.path, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                self._recent.append(row)

    def log(self, ticker, mode, candle_time, close, prediction, confidence, rule_patterns, timeframe='1m'):
        """Write one row unless this ticker/mode/candle was already logged. Returns True if written."""
        if not self._loaded:
            self._load()
        key = (ticker, mode, timeframe, str(candle_time))
        if key in self._seen:
            return False
        self._seen.add(key)
        row = {
            'logged_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'ticker': ticker, 'mode': mode, 'timeframe': timeframe, 'candle_time': str(candle_time), 'close': f'{close:.4f}',
            'prediction': prediction, 'confidence': f'{confidence:.4f}',
            'rule_patterns': '|'.join(rule_patterns), 'agrees': str(prediction in rule_patterns),
        }
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        new_file = not os.path.exists(self.path)
        with open(self.path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow(row)
        self._recent.append(row)
        return True

    def recent(self, n=10):
        """Newest first."""
        if not self._loaded:
            self._load()
        return list(self._recent)[-n:][::-1]

    def exists(self):
        return os.path.exists(self.path)
