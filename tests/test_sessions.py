import os
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sessions import SessionState


def make(max_sessions=20):
    return SessionState(lambda: {'ticker': None, 'used': 0.0}, max_sessions)


def test_tabs_do_not_share_state():
    s = make()
    s.use('a')
    s['ticker'] = 'AAPL'
    s.use('b')
    assert s['ticker'] is None
    s['ticker'] = 'TSLA'
    s.use('a')
    assert s['ticker'] == 'AAPL'


def test_threads_see_their_own_session():
    s = make()
    seen = {}

    def worker(sid, ticker):
        s.use(sid)
        s['ticker'] = ticker
        threading.Event().wait(0.05)
        seen[sid] = s['ticker']

    threads = [threading.Thread(target=worker, args=(f't{i}', f'T{i}')) for i in range(5)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert seen == {f't{i}': f'T{i}' for i in range(5)}


def test_old_sessions_are_evicted():
    s = make(max_sessions=3)
    for i in range(6):
        s.use(f's{i}')
    assert len(s.sessions) == 3 and 's5' in s.sessions and 's0' not in s.sessions


def test_update_and_missing_sid():
    s = make()
    s.use(None)
    s.update(ticker='X')
    s.use('')
    assert s['ticker'] == 'X'  # both map to the shared default session
