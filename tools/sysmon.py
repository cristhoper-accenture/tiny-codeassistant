"""Near-realtime system resource monitor — CPU% and RAM usage."""

import threading

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False


class SystemMonitor:
    """Background daemon thread that polls CPU and RAM every *interval* seconds.

    Usage:
        mon = SystemMonitor().start()
        ...
        print(mon.stats_text())  # "RAM 4.2/16 GB  CPU 23%"
        mon.stop()

    If psutil is not installed, all methods are no-ops and stats_text() returns "".
    """

    def __init__(self, interval: float = 0.5):
        self._interval = interval
        self._cpu: float = 0.0
        self._ram_used: float = 0.0
        self._ram_total: float = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self) -> "SystemMonitor":
        if not _PSUTIL:
            return self
        # Prime cpu_percent so the first real reading isn't 0.0
        psutil.cpu_percent(interval=None)
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True, name="sysmon")
        self._thread.start()
        return self

    def stop(self) -> None:
        if not _PSUTIL or self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=1)
        self._thread = None

    def __enter__(self) -> "SystemMonitor":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    # ── Poll loop ──────────────────────────────────────────────────────────────

    def _poll(self) -> None:
        while not self._stop.wait(self._interval):
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            with self._lock:
                self._cpu = cpu
                self._ram_used = mem.used / (1024 ** 3)
                self._ram_total = mem.total / (1024 ** 3)

    # ── Public API ─────────────────────────────────────────────────────────────

    def stats_text(self) -> str:
        """Return a compact stats string, or '' if psutil is unavailable."""
        if not _PSUTIL:
            return ""
        with self._lock:
            return f"RAM {self._ram_used:.1f}/{self._ram_total:.0f} GB  CPU {self._cpu:.0f}%"
