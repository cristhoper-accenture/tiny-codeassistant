"""
Regression tests for the ReAct JSON action extractor (agents/base.py).

These cover the small-model failure mode where a `write_file` / `edit_file`
action carries a long free-text field (markdown plan, code) containing real
newlines and unescaped quotes — which plain JSON parsing cannot handle. Before
the salvage path was added, such actions silently failed to parse, so the
planner's plan_*.md file was never written.

Run from project root: .venv/bin/python smoke-test/test_json_extract.py
"""

import sys
import os
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agents.base import BaseAgent  # noqa: E402
from agents.planner import PlannerAgent  # noqa: E402

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
_results = []


def check(name, fn):
    try:
        fn()
        print(f"  {PASS}  {name}")
        _results.append((name, True, None))
    except Exception as e:
        print(f"  {FAIL}  {name}: {e}")
        _results.append((name, False, e))


# ── Extractor: well-formed actions still parse (no regression) ──────────────────

def test_plain_action():
    obj = BaseAgent._extract_action(
        '{"thought": "look", "action": "read_file", "action_input": {"path": "a.py"}}'
    )
    assert obj and obj["action"] == "read_file"
    assert obj["action_input"]["path"] == "a.py"


def test_final_answer():
    obj = BaseAgent._extract_action(
        '{"thought": "done", "action": "final_answer", "action_input": {"response": "ok"}}'
    )
    assert obj and obj["action"] == "final_answer"
    assert obj["action_input"]["response"] == "ok"


def test_escaped_content_fenced():
    raw = ('```json\n{"thought": "x", "action": "write_file", "action_input": '
           '{"path": "plan_foo.md", "content": "# Plan\\n\\n1. do \\"it\\"\\n"}}\n```')
    obj = BaseAgent._extract_action(raw)
    assert obj and obj["action"] == "write_file"
    assert obj["action_input"]["content"] == '# Plan\n\n1. do "it"\n'


# ── Extractor: salvage of messy long-text actions (the reported bug) ────────────

def test_write_file_real_newlines_and_quotes():
    raw = '''```json
{"thought": "save plan", "action": "write_file", "action_input": {"path": "plan_foo.md", "content": "# Plan

## Steps
1. Build `foo.py`

```python
def foo():
    return "ok"
```

Done.
"}}
```'''
    obj = BaseAgent._extract_action(raw)
    assert obj is not None, "salvage returned None"
    assert obj["action"] == "write_file"
    assert obj["action_input"]["path"] == "plan_foo.md"
    content = obj["action_input"]["content"]
    assert "```python" in content, "embedded code fence lost"
    assert 'return "ok"' in content, "interior quotes lost"
    assert "## Steps" in content


def test_edit_file_multiline_newtext():
    raw = ('{"thought": "edit", "action": "edit_file", "action_input": '
           '{"path": "a.py", "old_text": "x = 1", "new_text": "x = "hi"\ny = 2"}}')
    obj = BaseAgent._extract_action(raw)
    assert obj is not None
    assert obj["action"] == "edit_file"
    assert obj["action_input"]["path"] == "a.py"
    assert "y = 2" in obj["action_input"]["new_text"]


# ── Full loop: planner actually writes the file to disk ─────────────────────────

def test_planner_writes_file_to_disk():
    tmp = tempfile.mkdtemp()
    agent = PlannerAgent(cwd=tmp, streaming=False)
    responses = iter([
        '''{"thought": "save", "action": "write_file", "action_input": {"path": "plan_demo.md", "content": "# Plan

```python
def demo():
    print("hi")
```
"}}''',
        '{"thought":"done","action":"final_answer","action_input":{"response":"saved"}}',
    ])
    # Silence display and stub the LLM.
    agent._print_thought = lambda *a, **k: None
    agent._print_tool_call = lambda *a, **k: None
    agent._print_tool_result = lambda *a, **k: None
    agent._print_answer = lambda *a, **k: None
    agent._call_llm = lambda messages: next(responses)

    agent.run("plan demo")
    path = os.path.join(tmp, "plan_demo.md")
    assert os.path.exists(path), "plan file was not written"
    body = open(path).read()
    assert 'print("hi")' in body, "content not preserved on disk"


if __name__ == "__main__":
    print("JSON action extractor tests\n")
    check("plain action parses", test_plain_action)
    check("final_answer parses", test_final_answer)
    check("escaped fenced content parses", test_escaped_content_fenced)
    check("salvage: write_file w/ real newlines + quotes", test_write_file_real_newlines_and_quotes)
    check("salvage: edit_file multiline new_text", test_edit_file_multiline_newtext)
    check("full loop: planner writes plan_*.md to disk", test_planner_writes_file_to_disk)

    failed = [n for n, ok, _ in _results if not ok]
    print(f"\n{len(_results) - len(failed)}/{len(_results)} passed")
    sys.exit(1 if failed else 0)
