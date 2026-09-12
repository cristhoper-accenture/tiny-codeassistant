"""Background token counter — tracks cumulative user/LLM token usage per session.

Tokens are approximated at ~4 chars each, which is accurate enough for
display purposes without requiring an external tokenizer dependency.
"""

import queue
import threading


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class TokenCounter:
    """Thread-safe token accumulator driven by a background worker queue.

    Call record(role, text) from any thread; counting happens asynchronously.
    Accumulated counts persist across start/stop cycles (session totals).
    stats_text() is safe to call from any thread at any time.

    Usage:
        counter = TokenCounter().start()
        counter.record("user", "explain quicksort")
        counter.record("llm", "<full LLM response>")
        print(counter.stats_text())  # "usr 5  llm 312  tot 317 tok"
        counter.stop()
    """

    def __init__(self) -> None:
        self._user: int = 0
        self._llm: int = 0
        self._lock = threading.Lock()
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self) -> "TokenCounter":
        """Start the background worker thread. Safe to call multiple times."""
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True, name="token-counter")
        self._thread.start()
        return self

    def stop(self) -> None:
        """Signal the worker to stop after draining the queue."""
        if self._thread is None:
            return
        self._stop.set()
        self._queue.put(None)  # wake the worker so it exits immediately
        self._thread.join(timeout=1)
        self._thread = None

    def __enter__(self) -> "TokenCounter":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    # ── Public API ─────────────────────────────────────────────────────────────

    def record(self, role: str, text: str) -> None:
        """Queue text for counting. role must be 'user' or 'llm'."""
        if text:
            self._queue.put((role, text))

    def stats_text(self) -> str:
        """Compact one-liner suitable for status bars."""
        with self._lock:
            return f"usr {self._user:,}  llm {self._llm:,}  tot {self._user + self._llm:,} tok"

    @property
    def user_tokens(self) -> int:
        with self._lock:
            return self._user

    @property
    def llm_tokens(self) -> int:
        with self._lock:
            return self._llm

    @property
    def total_tokens(self) -> int:
        with self._lock:
            return self._user + self._llm

    # ── Worker ─────────────────────────────────────────────────────────────────

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                # Drain any items that arrived before the stop signal
                while True:
                    try:
                        item2 = self._queue.get_nowait()
                        if item2 is not None:
                            self._accumulate(*item2)
                    except queue.Empty:
                        break
                break
            self._accumulate(*item)

    def _accumulate(self, role: str, text: str) -> None:
        n = _approx_tokens(text)
        with self._lock:
            if role == "user":
                self._user += n
            else:
                self._llm += n
