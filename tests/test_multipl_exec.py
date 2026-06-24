"""Unit tests for the multi-language MultiPL-E executor (scripts/transplant/multipl_exec.py).

We can actually run py/cpp/js on the host (interpreters/compilers present); java/go are
skipped when their toolchain is absent. The point is to lock the pass/fail contract per
language: a program that exits 0 == passed; a failing assert / nonzero exit == not passed;
and that the per-language file naming + run command matches the bigcode MultiPL-E runners.
"""
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "transplant"))
import multipl_exec as mx  # noqa: E402


# (lang, toolchain-probe, passing-source, failing-source)
CASES = {
    "py": (
        lambda: shutil.which("python3"),
        "assert 1 + 1 == 2\n",
        "assert 1 + 1 == 3\n",
    ),
    "cpp": (
        lambda: shutil.which("g++"),
        "#include<cassert>\nint main(){ assert(1+1==2); return 0; }\n",
        "#include<cassert>\nint main(){ assert(1+1==3); return 0; }\n",
    ),
    "js": (
        lambda: shutil.which("node"),
        "const assert=require('node:assert');\nassert.equal(1+1,2);\n",
        "const assert=require('node:assert');\nassert.equal(1+1,3);\n",
    ),
    "java": (
        lambda: shutil.which("javac") and shutil.which("java"),
        "public class Problem { public static void main(String[] a){ assert 1+1==2; } }\n",
        "public class Problem { public static void main(String[] a){ assert 1+1==3; if(true) throw new RuntimeException(); } }\n",
    ),
    "go": (
        lambda: shutil.which("go"),
        "package p_test\nimport \"testing\"\nfunc TestX(t *testing.T){ if 1+1!=2 { t.Errorf(\"bad\") } }\n",
        "package p_test\nimport \"testing\"\nfunc TestX(t *testing.T){ if 1+1!=3 { t.Errorf(\"bad\") } }\n",
    ),
}


@pytest.mark.parametrize("lang", list(CASES))
def test_pass_and_fail(lang):
    probe, ok_src, bad_src = CASES[lang]
    if not probe():
        pytest.skip(f"{lang} toolchain not on host")
    ok = mx.run_one(ok_src, lang, timeout=30)
    bad = mx.run_one(bad_src, lang, timeout=30)
    assert ok["passed"] is True, f"{lang} passing program should pass: {ok}"
    assert bad["passed"] is False, f"{lang} failing program should fail: {bad}"
    assert ok["status"] == "passed"


def test_timeout_result_is_json_serializable():
    """A program that times out must yield a JSON-serializable row (TimeoutExpired.stdout
    can be bytes even with text=True — that previously crashed json.dumps on the ceiling model)."""
    import json
    if not shutil.which("python3"):
        pytest.skip("python3 not on host")
    out = mx.run_one("while True:\n    pass\n", "py", timeout=2)
    assert out["status"] == "timeout"
    assert out["passed"] is False
    json.dumps(out)  # must not raise
    for k in ("compile_stdout", "compile_stderr", "run_stdout", "run_stderr"):
        assert isinstance(out[k], str)


def test_unknown_lang_raises():
    with pytest.raises(Exception):
        mx.run_one("x", "ruby", timeout=5)


def test_langs_registry_has_five():
    assert set(mx.LANGS) >= {"py", "java", "cpp", "js", "go"}
