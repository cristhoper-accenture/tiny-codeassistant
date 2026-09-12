"""Background token counter — tracks cumulative user/LLM token usage per session.

Tokens are approximated at ~4 chars each, which is accurate enough for
display purposes without requiring an external tokenizer dependency.

A single TokenCounter instance is shared across the agent hierarchy so that
delegated sub-agent tokens roll up into the session total automatically.
"""

import queue
import threading


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class TokenCounter:
    """Thread-safe token accumulator driven by a background worker queue.

    Call record(role, text, agent) from any thread; counting happens
    asynchronously. Accumulated counts persist across start/stop cycles
    (session totals). All stats methods are safe to call from any thread.

    Usage:
        counter = TokenCounter().start()
        counter.record("user", "explain quicksort", agent="orchestrator")
        counter.record("llm", "<response>", agent="coder")
        print(counter.stats_text())               # session total
        print(counter.agent_stats_text("coder"))  # per-agent line
        counter.stop()
    """

    def __init__(self) -> None:
        self._user: int = 0
        self._llm: int = 0
        # {agent_name: {"user": int, "llm": int}}
        self._per_agent: dict[str, dict[str, int]] = {}
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

    def record(self, role: str, text: str, agent: str = "") -> None:
        """Queue text for counting. role must be 'user' or 'llm'."""
        if text:
            self._queue.put((role, text, agent))

    def stats_text(self) -> str:
        """Session-total one-liner: 'usr X  llm Y  tot Z tok'."""
        with self._lock:
            return f"usr {self._user:,}  llm {self._llm:,}  tot {self._user + self._llm:,} tok"

    def agent_stats_text(self, agent: str) -> str:
        """Per-agent one-liner: 'coder usr 5  llm 48  tot 53 tok'."""
        with self._lock:
            counts = self._per_agent.get(agent, {"user": 0, "llm": 0})
            u, l = counts["user"], counts["llm"]
            return f"{agent} usr {u:,}  llm {l:,}  tot {u + l:,} tok"

    def agent_names(self) -> list[str]:
        """Return agent names seen so far, in insertion order."""
        with self._lock:
            return list(self._per_agent.keys())

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

    def _accumulate(self, role: str, text: str, agent: str) -> None:
        n = _approx_tokens(text)
        with self._lock:
            if role == "user":
                self._user += n
            else:
                self._llm += n
            if agent:
                bucket = self._per_agent.setdefault(agent, {"user": 0, "llm": 0})
                bucket["user" if role == "user" else "llm"] += n
