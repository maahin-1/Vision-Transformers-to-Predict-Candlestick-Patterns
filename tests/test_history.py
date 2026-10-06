import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from history import FIELDS, HistoryLog


def test_logs_once_per_candle(tmp_path):
    log = HistoryLog(str(tmp_path / 'logs' / 'd.csv'))
    assert log.log('AAPL', 'live', '2026-10-06 15:59', 256.33, 'doji', 0.9, ['doji'])
    assert not log.log('AAPL', 'live', '2026-10-06 15:59', 256.33, 'doji', 0.9, ['doji'])
    assert log.log('AAPL', 'replay', '2026-10-06 15:59', 256.33, 'doji', 0.9, [])  # other mode is a new row
    with open(log.path, newline='') as f:
        rows = list(csv.DictReader(f))
    assert [r['mode'] for r in rows] == ['live', 'replay']
    assert list(rows[0]) == FIELDS
    assert rows[0]['agrees'] == 'True' and rows[1]['agrees'] == 'False'


def test_recent_is_newest_first_and_survives_restart(tmp_path):
    path = str(tmp_path / 'd.csv')
    first = HistoryLog(path)
    for i in range(3):
        first.log('TSLA', 'live', f'c{i}', 1.0, 'doji', 0.5, [])
    assert [r['candle_time'] for r in first.recent(2)] == ['c2', 'c1']
    second = HistoryLog(path)  # new process, same file
    assert [r['candle_time'] for r in second.recent(3)] == ['c2', 'c1', 'c0']
