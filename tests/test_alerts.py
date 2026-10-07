import io
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import alerts


def row(**kw):
    base = {'error': '', 'ticker': 'AAPL', 'timeframe': '1m', 'candle_time': '2026-01-05 15:00', 'close': 190.5,
            'prediction': 'bullish_engulfing', 'confidence': 0.93, 'rules': ['bullish_engulfing'], 'agrees': True,
            'market': 'live', 'trend': 'down', 'fit': 'fits'}
    base.update(kw)
    return base


def test_matches_defaults():
    s = alerts.Settings()
    assert alerts.matches(row(), s)
    assert not alerts.matches(row(confidence=0.80), s)
    assert not alerts.matches(row(agrees=False, rules=[]), s)
    assert not alerts.matches(row(market='closed'), s)  # stale or replayed data never alerts
    assert not alerts.matches({'error': 'boom'}, s)


def test_matches_optional_filters():
    assert alerts.matches(row(agrees=False), alerts.Settings(require_agree=False))
    assert not alerts.matches(row(fit='against'), alerts.Settings(require_fit=True))
    assert alerts.matches(row(fit='fits'), alerts.Settings(require_fit=True))
    assert alerts.matches(row(confidence=0.5), alerts.Settings(min_confidence=0.5))


def test_feed_alerts_once_per_candle():
    feed = alerts.AlertFeed()
    assert 'AAPL 1m: bullish_engulfing 93%' in feed.add(row())
    assert feed.add(row()) is None
    assert feed.add(row(candle_time='2026-01-05 15:01')) is not None
    assert feed.add(row(timeframe='5m')) is not None
    assert feed.total == 3
    assert feed.recent(1)[0]['message'].startswith('AAPL 5m')  # newest first


def test_telegram_credentials():
    assert alerts.telegram_credentials({}) is None
    assert alerts.telegram_credentials({'TELEGRAM_BOT_TOKEN': 't'}) is None
    assert alerts.telegram_credentials({'TELEGRAM_BOT_TOKEN': ' t ', 'TELEGRAM_CHAT_ID': '42'}) == ('t', '42')


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_send_telegram_success_and_request_shape():
    seen = {}

    def opener(request, timeout):
        seen.update(url=request.full_url, data=request.data.decode(), timeout=timeout)
        return FakeResponse(json.dumps({'ok': True}).encode())

    assert alerts.send_telegram('hello there', 'TOKEN', '42', opener) == (True, '')
    assert seen['url'] == 'https://api.telegram.org/botTOKEN/sendMessage'
    assert 'chat_id=42' in seen['data'] and 'hello+there' in seen['data']
    assert seen['timeout'] == alerts.TELEGRAM_TIMEOUT_S


def test_send_telegram_failures_never_raise_or_leak_the_token():
    def broken(request, timeout):
        raise OSError('cannot reach https://api.telegram.org/botSECRET/sendMessage')

    ok, error = alerts.send_telegram('x', 'SECRET', '1', broken)
    assert not ok and 'SECRET' not in error

    def rejected(request, timeout):
        return FakeResponse(json.dumps({'ok': False, 'description': 'chat not found'}).encode())

    assert alerts.send_telegram('x', 'T', '1', rejected) == (False, 'chat not found')
