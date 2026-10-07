"""Per-browser-tab state. S[...] resolves to the tab whose callback is currently running."""
import threading
import time


class SessionState:
    def __init__(self, new_state, max_sessions=20):
        self._new_state = new_state
        self._max = max_sessions
        self.sessions = {}
        self._local = threading.local()

    def use(self, sid):
        """Select (creating if needed) the state for `sid` for the calling thread."""
        state = self.sessions.setdefault(sid or 'default', self._new_state())
        state['used'] = time.monotonic()
        self._local.state = state
        for stale in sorted(self.sessions, key=lambda k: self.sessions[k]['used'])[:-self._max]:
            del self.sessions[stale]

    def __getitem__(self, key):
        return self._local.state[key]

    def __setitem__(self, key, value):
        self._local.state[key] = value

    def update(self, **kwargs):
        self._local.state.update(kwargs)
