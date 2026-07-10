import json

from fastapi.testclient import TestClient

import parametic_studio.api as api
from parametic_studio.kernel.humaneval import (
    build_program,
    check_correctness,
    estimate_pass_at_k,
    truncate_completion,
)

# A real, tiny HumanEvalPack-shaped problem (bigcode columns). The check() runs actual python.
ADD_ROW = {
    "task_id": "Test/0",
    "prompt": "def add(a, b):\n    \"\"\"Return the sum of a and b.\"\"\"\n",
    "canonical_solution": "    return a + b\n",
    "entry_point": "add",
    "import": "",
    "test_setup": "",
    "test": (
        "def check(candidate):\n"
        "    assert candidate(1, 2) == 3\n"
        "    assert candidate(0, 0) == 0\n"
    ),
}


def test_canonical_solution_passes():
    program = build_program(ADD_ROW, ADD_ROW["canonical_solution"])
    passed, detail = check_correctness(program, timeout=5.0)
    assert passed is True, detail


def test_wrong_completion_fails():
    passed, detail = check_correctness(build_program(ADD_ROW, "    return None\n"), timeout=5.0)
    assert passed is False and detail


def test_syntax_error_completion_fails():
    passed, _ = check_correctness(build_program(ADD_ROW, "    return a +\n"), timeout=5.0)
    assert passed is False


def test_import_and_setup_are_included():
    row = {
        "task_id": "Test/1",
        "prompt": "def use_pi():\n    \"\"\"Return math.pi.\"\"\"\n",
        "canonical_solution": "    return math.pi\n",
        "entry_point": "use_pi",
        "import": "import math",
        "test_setup": "EXPECTED = 3.141592653589793",
        "test": "def check(candidate):\n    assert candidate() == EXPECTED\n",
    }
    passed, detail = check_correctness(build_program(row, row["canonical_solution"]), timeout=5.0)
    assert passed is True, detail


def test_truncate_completion_cuts_at_stop():
    raw = "    return a + b\ndef other():\n    pass\n"
    assert truncate_completion(raw) == "    return a + b"


def test_estimate_pass_at_k():
    assert estimate_pass_at_k(1, 1, 1) == 1.0
    assert estimate_pass_at_k(1, 0, 1) == 0.0
    assert estimate_pass_at_k(5, 0, 1) == 0.0
    assert estimate_pass_at_k(5, 5, 1) == 1.0
    assert abs(estimate_pass_at_k(2, 1, 1) - 0.5) < 1e-9


class _FakeSession:
    """Stands in for ModelSession: returns a fixed completion regardless of prompt."""
    def __init__(self, completion):
        self._completion = completion

    def complete_code(self, prompt, max_tokens=512, temperature=0.0, stops=None):
        return self._completion


def _write_dataset(tmp_path, rows):
    d = tmp_path / "datasets"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "he.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return "he.jsonl"


def test_ws_eval_code_streams_progress_then_result(tmp_path, monkeypatch):
    monkeypatch.setenv("PARAMETIC_STUDIO_HOME", str(tmp_path))
    name = _write_dataset(tmp_path, [ADD_ROW, ADD_ROW])  # 2 problems
    api.SESSION = _FakeSession("    return a + b\n")      # correct completion → both pass
    api.SESSIONS.clear()
    with TestClient(api.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "eval_code", "dataset": name})
        progress, final = [], None
        while True:
            msg = ws.receive_json()
            if msg["type"] == "eval_result":
                final = msg
                break
            progress.append(msg)
    assert [p["type"] for p in progress] == ["eval_progress"] * 2
    assert [p["i"] for p in progress] == [1, 2]
    assert all(p["total"] == 2 for p in progress)
    assert progress[-1]["passed"] == 2
    assert final["dataset"] == name and final["passed"] == 2 and final["total"] == 2
    assert final["pass_at_1"] == 1.0


def test_ws_eval_code_wrong_completion_zero_pass(tmp_path, monkeypatch):
    monkeypatch.setenv("PARAMETIC_STUDIO_HOME", str(tmp_path))
    name = _write_dataset(tmp_path, [ADD_ROW])
    api.SESSION = _FakeSession("    return None\n")
    api.SESSIONS.clear()
    with TestClient(api.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "eval_code", "dataset": name})
        while (msg := ws.receive_json())["type"] != "eval_result":
            pass
    assert msg["passed"] == 0 and msg["total"] == 1 and msg["pass_at_1"] == 0.0


def test_ws_eval_code_limit_caps_problems(tmp_path, monkeypatch):
    monkeypatch.setenv("PARAMETIC_STUDIO_HOME", str(tmp_path))
    name = _write_dataset(tmp_path, [ADD_ROW, ADD_ROW, ADD_ROW])
    api.SESSION = _FakeSession("    return a + b\n")
    api.SESSIONS.clear()
    with TestClient(api.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "eval_code", "dataset": name, "limit": 1})
        while (msg := ws.receive_json())["type"] != "eval_result":
            pass
    assert msg["total"] == 1


def test_ws_eval_code_missing_dataset_errors_and_survives(tmp_path, monkeypatch):
    monkeypatch.setenv("PARAMETIC_STUDIO_HOME", str(tmp_path))
    api.SESSION = _FakeSession("    return a + b\n")
    api.SESSIONS.clear()
    with TestClient(api.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "eval_code", "dataset": "nope.jsonl"})
        err = ws.receive_json()
        assert err["type"] == "error" and err["op"] == "eval_code" and err["reason"]
        ws.send_json({"type": "datasets"})  # connection still serves ops after the error
        assert ws.receive_json()["type"] == "datasets"
