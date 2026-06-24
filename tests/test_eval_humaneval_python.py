"""Unit tests for the Python-HumanEval (lm-eval) driver's pure pass@1 extractor (no GPU/lm-eval)."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "eval_humaneval_python",
    Path(__file__).resolve().parents[1] / "scripts" / "eval_humaneval_python.py",
)
ehp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ehp)


def test_extracts_pass_at_1_instruct_shape():
    # lm-eval results: results.humaneval_instruct["pass@1,<filter>"]
    rj = {"results": {"humaneval_instruct": {"pass@1,create_test": 0.42, "pass@1_stderr,create_test": 0.01}}}
    assert ehp.extract_pass_at_1(rj) == 0.42


def test_falls_back_to_plain_humaneval_key():
    rj = {"results": {"humaneval": {"pass@1,none": 0.3}}}
    assert ehp.extract_pass_at_1(rj) == 0.3


def test_skips_stderr_key():
    rj = {"results": {"humaneval_instruct": {"pass@1_stderr,none": 0.01, "pass@1,none": 0.5}}}
    assert ehp.extract_pass_at_1(rj) == 0.5


def test_returns_none_when_absent():
    assert ehp.extract_pass_at_1({"results": {"humaneval_instruct": {"acc": 0.9}}}) is None
    assert ehp.extract_pass_at_1({"results": {}}) is None
