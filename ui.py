#!/usr/bin/env python3
"""PyQt6 GUI entry point for the local AI code assistant.

Usage:
  .venv/bin/python ui.py
  .venv/bin/python ui.py --cwd /some/project
"""

import sys
import json
import os
import html
import argparse

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QTextBrowser, QTextEdit, QPushButton, QLabel,
    QComboBox, QLineEdit, QCheckBox, QGroupBox, QStatusBar,
    QScrollBar, QFileDialog,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QTextCursor, QFont, QKeySequence, QShortcut

from agents import REGISTRY, DEFAULT_AGENT
from config import AGENT_MODELS, DEFAULT_MODEL, LLM_PROVIDER
import llm


# ── Stylesheet ──────────────────────────────────────────────────────────────

_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-size: 13px;
}
QTextBrowser, QTextEdit {
    background-color: #181825;
    border: 1px solid #313244;
    border-radius: 6px;
    padding: 6px;
    color: #cdd6f4;
}
QTextEdit:focus { border-color: #89b4fa; }
QPushButton {
    background-color: #89b4fa;
    color: #1e1e2e;
    border: none;
    border-radius: 6px;
    padding: 8px 18px;
    font-weight: bold;
}
QPushButton:hover  { background-color: #b4befe; }
QPushButton:disabled { background-color: #45475a; color: #6c7086; }
QPushButton[flat="true"] {
    background-color: #313244;
    color: #cdd6f4;
    font-weight: normal;
}
QPushButton[flat="true"]:hover { background-color: #45475a; }
QComboBox, QLineEdit {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 4px 8px;
    color: #cdd6f4;
}
QComboBox::drop-down { border: none; }
QLabel    { color: #a6adc8; }
QCheckBox { color: #cdd6f4; }
QGroupBox {
    border: 1px solid #313244;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 10px;
    font-weight: bold;
    color: #89b4fa;
}
QScrollBar:vertical {
    background: #181825;
    width: 8px;
    border-radius: 4px;
}
QScrollBar::handle:vertical {
    background: #45475a;
    border-radius: 4px;
    min-height: 20px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QStatusBar { background-color: #181825; color: #6c7086; font-size: 12px; }
QSplitter::handle { background-color: #313244; width: 2px; }
"""

# ── Event styles (kind → border-color, bg-color) ────────────────────────────

_STYLES: dict[str, tuple[str, str]] = {
    "user":         ("#89b4fa", "#12213d"),
    "thought":      ("#6c7086", "#1a1a2e"),
    "tool_call":    ("#f9e2af", "#2e2008"),
    "tool_result":  ("#94e2d5", "#082624"),
    "answer":       ("#a6e3a1", "#082010"),
    "delegation":   ("#cba6f7", "#1a0828"),
    "clarification":("#f9e2af", "#2e2008"),
    "error":        ("#f38ba8", "#2a0812"),
    "warning":      ("#fab387", "#2a1208"),
}


def _event_html(kind: str, label: str, body: str, sub: str = "") -> str:
    color, bg = _STYLES.get(kind, ("#cdd6f4", "#181825"))
    body_esc = html.escape(body).replace("\n", "<br>")
    sub_html = (
        f'<div style="color:#6c7086;font-size:11px;margin-top:3px">{html.escape(sub)}</div>'
        if sub else ""
    )
    return (
        f'<div style="background:{bg};border-left:3px solid {color};'
        f'border-radius:4px;padding:8px 12px;margin:4px 0;">'
        f'<div style="color:{color};font-weight:bold;font-size:11px;'
        f'margin-bottom:4px;">{html.escape(label)}</div>'
        f'<div style="color:#cdd6f4;font-family:monospace">{body_esc}</div>'
        f'{sub_html}</div>'
    )


# ── Agent worker thread ──────────────────────────────────────────────────────

class AgentWorker(QThread):
    sig_thought      = pyqtSignal(str)
    sig_tool_call    = pyqtSignal(str, str)   # action, params_json
    sig_tool_result  = pyqtSignal(str, str)   # preview, cwd_display
    sig_answer       = pyqtSignal(str, str)   # label, text
    sig_chunk        = pyqtSignal(str)        # raw streaming chunk (unused in display)
    sig_delegation   = pyqtSignal(str, str)   # agent_name, task_preview
    sig_clarification = pyqtSignal(str, str)  # label, question
    sig_format_warn  = pyqtSignal(str)
    sig_finished     = pyqtSignal(str)        # new cwd
    sig_error        = pyqtSignal(str)

    def __init__(self, agent_name: str, model_override, cwd: str,
                 user_input: str, streaming: bool) -> None:
        super().__init__()
        self._agent_name = agent_name
        self._model_override = model_override
        self._cwd = cwd
        self._user_input = user_input
        self._streaming = streaming

    def run(self) -> None:
        try:
            agent_cls = REGISTRY[self._agent_name]
            agent = agent_cls(
                model=self._model_override,
                cwd=self._cwd,
                streaming=self._streaming,
            )
            self._patch(agent)
            _, new_cwd = agent.run(self._user_input)
            self.sig_finished.emit(new_cwd)
        except Exception as exc:
            self.sig_error.emit(str(exc))
            self.sig_finished.emit(self._cwd)

    def _patch(self, agent) -> None:
        w = self

        def _print_thought(thought):
            w.sig_thought.emit(thought)

        def _print_tool_call(action, params):
            w.sig_tool_call.emit(action, json.dumps(params, indent=2))

        def _print_tool_result(result):
            preview = result[:500] + ("…" if len(result) > 500 else "")
            w.sig_tool_result.emit(preview, agent._cwd_display())

        def _print_answer(text):
            w.sig_answer.emit(agent.label, text)

        def _print_delegation(agent_name, task):
            preview = task[:120] + ("…" if len(task) > 120 else "")
            w.sig_delegation.emit(agent_name, preview)

        def _print_clarification(question):
            w.sig_clarification.emit(agent.label, question)

        def _print_format_warning(raw):
            w.sig_format_warn.emit(raw[:120] + ("…" if len(raw) > 120 else ""))

        def _print_banner():
            pass

        def _call_llm_inner(messages):
            if not agent._streaming:
                return llm.chat(messages, model=agent.model)
            parts: list[str] = []
            def _on_chunk(chunk: str) -> None:
                parts.append(chunk)
                w.sig_chunk.emit(chunk)
            llm.chat(messages, model=agent.model, on_chunk=_on_chunk)
            return "".join(parts)

        agent._print_thought       = _print_thought
        agent._print_tool_call     = _print_tool_call
        agent._print_tool_result   = _print_tool_result
        agent._print_answer        = _print_answer
        agent._print_delegation    = _print_delegation
        agent._print_clarification = _print_clarification
        agent._print_format_warning = _print_format_warning
        agent._print_banner        = _print_banner
        agent._call_llm_inner      = _call_llm_inner

        # OrchestratorAgent creates sub-agents directly via _make_agent / _run_agent —
        # not through delegate_to_agent — so patch those methods to auto-patch sub-agents.
        if hasattr(agent, '_make_agent'):
            _orig_make_agent = agent._make_agent
            def _make_agent(name):
                sub = _orig_make_agent(name)
                w._patch(sub)
                return sub
            agent._make_agent = _make_agent

        if hasattr(agent, '_print_routing'):
            def _print_routing(agent_name, _user_input=""):
                w.sig_thought.emit(f"Routing to {agent_name} agent…")
            agent._print_routing = _print_routing

        if hasattr(agent, '_print_pipeline_start'):
            def _print_pipeline_start():
                w.sig_thought.emit("Running implementation pipeline: planner → breakdown → coder")
            agent._print_pipeline_start = _print_pipeline_start


# ── Chat display ─────────────────────────────────────────────────────────────

class ChatDisplay(QTextBrowser):
    def __init__(self) -> None:
        super().__init__()
        self.setOpenLinks(False)
        self.document().setDefaultStyleSheet(
            "body { font-family: monospace; font-size: 13px; }"
        )

    def add(self, kind: str, label: str, body: str, sub: str = "") -> None:
        cursor = QTextCursor(self.document())
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if not cursor.atStart():
            cursor.insertBlock()
        cursor.insertHtml(_event_html(kind, label, body, sub))
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())


# ── Settings panel ────────────────────────────────────────────────────────────

class SettingsPanel(QWidget):
    def __init__(self, start_cwd: str, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        # Agent
        ag_box = QGroupBox("Agent")
        ag_lay = QVBoxLayout(ag_box)
        self.agent_combo = QComboBox()
        for name, cls in REGISTRY.items():
            self.agent_combo.addItem(f"{name}  —  {cls.label}", name)
            if name == DEFAULT_AGENT:
                self.agent_combo.setCurrentIndex(self.agent_combo.count() - 1)
        ag_lay.addWidget(self.agent_combo)
        layout.addWidget(ag_box)

        # Model override
        mo_box = QGroupBox("Model override")
        mo_lay = QVBoxLayout(mo_box)
        mo_row = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.lineEdit().setPlaceholderText("per-agent default")
        mo_row.addWidget(self.model_combo)
        self._refresh_btn = QPushButton("↻")
        self._refresh_btn.setFixedWidth(32)
        self._refresh_btn.setToolTip("Refresh model list from Ollama")
        self._refresh_btn.setProperty("flat", "true")
        self._refresh_btn.clicked.connect(self._refresh_models)
        mo_row.addWidget(self._refresh_btn)
        mo_lay.addLayout(mo_row)
        layout.addWidget(mo_box)
        self._refresh_models()

        # CWD
        cwd_box = QGroupBox("Working directory")
        cwd_lay = QVBoxLayout(cwd_box)
        cwd_row = QHBoxLayout()
        self.cwd_edit = QLineEdit(start_cwd)
        self.cwd_edit.setReadOnly(True)
        cwd_row.addWidget(self.cwd_edit)
        self._browse_btn = QPushButton("…")
        self._browse_btn.setFixedWidth(32)
        self._browse_btn.setToolTip("Choose directory")
        self._browse_btn.setProperty("flat", "true")
        self._browse_btn.clicked.connect(self._browse_dir)
        cwd_row.addWidget(self._browse_btn)
        cwd_lay.addLayout(cwd_row)
        layout.addWidget(cwd_box)

        # Options
        self.stream_chk = QCheckBox("Streaming")
        self.stream_chk.setChecked(True)
        layout.addWidget(self.stream_chk)

        layout.addStretch()

        self.preload_btn = QPushButton("Preload models")
        layout.addWidget(self.preload_btn)

    @property
    def agent_name(self) -> str:
        return self.agent_combo.currentData()

    @property
    def model_override(self):
        v = self.model_combo.currentText().strip()
        return v if v else None

    def _refresh_models(self) -> None:
        self._refresh_btn.setEnabled(False)
        worker = _ModelListWorker()
        worker.done.connect(self._on_models_loaded)
        worker.start()
        self._model_list_worker = worker

    def _on_models_loaded(self, models: list) -> None:
        current = self.model_combo.currentText()
        self.model_combo.clear()
        self.model_combo.addItem("")          # blank = per-agent default
        for m in models:
            self.model_combo.addItem(m)
        idx = self.model_combo.findText(current)
        self.model_combo.setCurrentIndex(max(idx, 0))
        self._refresh_btn.setEnabled(True)

    @property
    def cwd(self) -> str:
        return self.cwd_edit.text().strip() or os.getcwd()

    @property
    def streaming(self) -> bool:
        return self.stream_chk.isChecked()

    def _browse_dir(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, "Select working directory", self.cwd_edit.text(),
            QFileDialog.Option.DontUseNativeDialog,
        )
        if chosen:
            self.cwd_edit.setText(chosen)

    def set_cwd(self, cwd: str) -> None:
        self.cwd_edit.setText(cwd)

    def lock(self, locked: bool) -> None:
        self.agent_combo.setEnabled(not locked)
        self.preload_btn.setEnabled(not locked)


# ── Input bar ─────────────────────────────────────────────────────────────────

class InputBar(QWidget):
    submitted = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("Type a message…  (Enter = send,  Shift+Enter = newline)")
        self.text_edit.setFixedHeight(66)
        self.text_edit.installEventFilter(self)
        lay.addWidget(self.text_edit)

        self.send_btn = QPushButton("Send")
        self.send_btn.setFixedSize(80, 66)
        self.send_btn.clicked.connect(self._submit)
        lay.addWidget(self.send_btn)

        from PyQt6.QtWidgets import QSizePolicy
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def eventFilter(self, obj, event):
        from PyQt6.QtCore import QEvent
        if obj is self.text_edit and event.type() == QEvent.Type.KeyPress:
            if (event.key() == Qt.Key.Key_Return
                    and not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier)):
                self._submit()
                return True
        return super().eventFilter(obj, event)

    def _submit(self) -> None:
        text = self.text_edit.toPlainText().strip()
        if text:
            self.text_edit.clear()
            self.submitted.emit(text)

    def set_enabled(self, enabled: bool) -> None:
        self.text_edit.setEnabled(enabled)
        self.send_btn.setEnabled(enabled)
        if enabled:
            self.text_edit.setFocus()


# ── Model list worker ────────────────────────────────────────────────────────

class _ModelListWorker(QThread):
    done = pyqtSignal(list)

    def run(self) -> None:
        try:
            models = llm.list_models()
        except Exception:
            models = []
        self.done.emit(models)


# ── Preload worker ────────────────────────────────────────────────────────────

class PreloadWorker(QThread):
    done = pyqtSignal(str)

    def __init__(self, agent_name: str, model_override) -> None:
        super().__init__()
        self._agent_name = agent_name
        self._model_override = model_override

    def run(self) -> None:
        if self._model_override:
            models = [self._model_override]
        elif self._agent_name == "orchestrator":
            models = list(dict.fromkeys(AGENT_MODELS.values()))
        else:
            models = [AGENT_MODELS.get(self._agent_name, DEFAULT_MODEL)]

        results = []
        for m in models:
            try:
                llm.preload(m)
                results.append(f"{m} ready")
            except Exception as exc:
                results.append(f"{m} failed: {exc}")
        self.done.emit(", ".join(results))


# ── Main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self, start_cwd: str) -> None:
        super().__init__()
        self.setWindowTitle("Code Assistant")
        self.resize(1280, 820)

        self._worker: AgentWorker | None = None
        self._preload_worker: PreloadWorker | None = None

        # Layout
        root = QWidget()
        self.setCentralWidget(root)
        root_lay = QHBoxLayout(root)
        root_lay.setContentsMargins(8, 8, 8, 8)
        root_lay.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root_lay.addWidget(splitter)

        # Left: settings
        self.settings = SettingsPanel(start_cwd)
        self.settings.setFixedWidth(260)
        self.settings.preload_btn.clicked.connect(self._preload)
        splitter.addWidget(self.settings)

        # Right: chat + input
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(8, 0, 0, 0)
        right_lay.setSpacing(8)

        self.chat = ChatDisplay()
        right_lay.addWidget(self.chat, 1)

        self.input_bar = InputBar()
        self.input_bar.submitted.connect(self._on_submit)
        right_lay.addWidget(self.input_bar, 0)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # Status bar
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready")

        self.input_bar.text_edit.setFocus()

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_submit(self, text: str) -> None:
        if self._worker and self._worker.isRunning():
            return

        self.chat.add("user", "You", text)
        self.input_bar.set_enabled(False)
        self.settings.lock(True)
        self._status.showMessage("Processing…")

        self._worker = AgentWorker(
            agent_name=self.settings.agent_name,
            model_override=self.settings.model_override,
            cwd=self.settings.cwd,
            user_input=text,
            streaming=self.settings.streaming,
        )
        w = self._worker
        w.sig_thought.connect(lambda t: self.chat.add("thought", "💭 Thinking", t))
        w.sig_tool_call.connect(lambda a, p: self.chat.add("tool_call", f"🔧 {a}", p))
        w.sig_tool_result.connect(lambda r, c: self.chat.add("tool_result", "📋 Result", r, sub=f"cwd: {c}"))
        w.sig_answer.connect(lambda lbl, txt: self.chat.add("answer", f"✅ {lbl}", txt))
        w.sig_delegation.connect(lambda n, t: self.chat.add("delegation", f"→ {n}", t))
        w.sig_clarification.connect(lambda lbl, q: self.chat.add("clarification", f"❓ {lbl}", q))
        w.sig_format_warn.connect(lambda p: self.chat.add("warning", "⚡ Format warning", p))
        w.sig_error.connect(lambda e: self.chat.add("error", "⚠ Error", e))
        w.sig_finished.connect(self._on_finished)
        w.sig_chunk.connect(self._on_chunk)
        self._chunk_count = 0
        w.start()

    def _on_chunk(self, _chunk: str) -> None:
        self._chunk_count += 1
        self._status.showMessage(f"Generating… ({self._chunk_count} tokens)")

    def _on_finished(self, new_cwd: str) -> None:
        self.settings.set_cwd(new_cwd)
        self.input_bar.set_enabled(True)
        self.settings.lock(False)
        self._status.showMessage("Ready")

    def _preload(self) -> None:
        if LLM_PROVIDER != "ollama":
            self._status.showMessage("Preload skipped — cloud provider active")
            return
        self._status.showMessage("Preloading models…")
        self.settings.preload_btn.setEnabled(False)
        self._preload_worker = PreloadWorker(
            agent_name=self.settings.agent_name,
            model_override=self.settings.model_override,
        )
        self._preload_worker.done.connect(self._on_preload_done)
        self._preload_worker.start()

    def _on_preload_done(self, msg: str) -> None:
        self._status.showMessage(f"Preload: {msg}")
        self.settings.preload_btn.setEnabled(True)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Code assistant — PyQt6 GUI")
    parser.add_argument("--cwd", default=None, help="Starting working directory")
    args = parser.parse_args()

    start_cwd = os.path.abspath(args.cwd) if args.cwd else os.getcwd()

    app = QApplication(sys.argv)
    app.setApplicationName("Code Assistant")
    app.setStyleSheet(_STYLESHEET)

    win = MainWindow(start_cwd)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
