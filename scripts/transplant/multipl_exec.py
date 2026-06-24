#!/usr/bin/env python3
"""Multi-language executor for MultiPL-E completion eval — per-problem pass/fail.

Run commands replicate the authoritative bigcode `multiple_metrics/eval_*.py` runners so
results match the standard MultiPL-E harness:
  py   : python3 prog.py                        (exit 0 = pass)
  java : javac Problem.java && java -ea Problem  (both exit 0 = pass)
  cpp  : g++ -std=c++17 prog.cpp -o prog && ./prog
  js   : node prog.js
  go   : go test prog.go  (filename NOT *_test.go; package is X_test; pass iff rc==0 and no FAIL)

The assembled program for every language is  prompt + completion + tests  (MultiPL-E
convention). java additionally strips the org.javatuples import (no jar shipped), matching
the prior single-language cowork harness so the java floor stays comparable.
"""
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

JAVATUPLES_IMPORT = "import org.javatuples.*;\n"

# lang -> (source filename, list of (cmd-template) stages). Filled implicitly in run_one.
LANGS = ("py", "java", "cpp", "js", "go")
_ALIASES = {"python": "py", "javascript": "js", "golang": "go", "c++": "cpp"}


def _norm(lang):
    lang = lang.lower()
    return _ALIASES.get(lang, lang)


def preprocess_source(source, lang):
    if lang == "java":
        return source.replace(JAVATUPLES_IMPORT, "")
    return source


def _s(x):
    """Coerce subprocess output to str — TimeoutExpired.stdout/stderr can be bytes even
    under text=True, which then breaks json.dumps of the per-problem row."""
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8", "ignore")
    return x or ""


def _missing(status):
    return {"passed": False, "status": status, "compile_stdout": "", "compile_stderr": status,
            "run_stdout": "", "run_stderr": "", "elapsed_sec": 0.0}


def _result(passed, status, comp=None, run=None, t0=0.0):
    comp = comp or subprocess.CompletedProcess([], 0, "", "")
    run = run or subprocess.CompletedProcess([], 0, "", "")
    return {"passed": bool(passed), "status": status,
            "compile_stdout": _s(comp.stdout), "compile_stderr": _s(comp.stderr),
            "run_stdout": _s(run.stdout), "run_stderr": _s(run.stderr),
            "elapsed_sec": time.time() - t0}


def run_one(source, lang, timeout=15):
    """Compile+run one assembled program; return per-problem outcome dict."""
    lang = _norm(lang)
    if lang not in LANGS:
        raise ValueError(f"unsupported language: {lang}")
    source = preprocess_source(source, lang)
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        env = dict(os.environ)
        try:
            if lang == "py":
                if not shutil.which("python3"):
                    return _missing("missing_toolchain")
                f = tmp / "prog.py"; f.write_text(source, encoding="utf-8")
                r = subprocess.run(["python3", "prog.py"], cwd=tmp, text=True, capture_output=True, timeout=timeout)
                return _result(r.returncode == 0, "passed" if r.returncode == 0 else "runtime_error", run=r, t0=t0)

            if lang == "js":
                if not shutil.which("node"):
                    return _missing("missing_toolchain")
                f = tmp / "prog.js"; f.write_text(source, encoding="utf-8")
                r = subprocess.run(["node", "prog.js"], cwd=tmp, text=True, capture_output=True, timeout=timeout)
                return _result(r.returncode == 0, "passed" if r.returncode == 0 else "runtime_error", run=r, t0=t0)

            if lang == "cpp":
                if not shutil.which("g++"):
                    return _missing("missing_toolchain")
                f = tmp / "prog.cpp"; f.write_text(source, encoding="utf-8")
                c = subprocess.run(["g++", "-std=c++17", "prog.cpp", "-o", "prog"], cwd=tmp,
                                   text=True, capture_output=True, timeout=timeout)
                if c.returncode != 0:
                    return _result(False, "compile_error", comp=c, t0=t0)
                r = subprocess.run(["./prog"], cwd=tmp, text=True, capture_output=True, timeout=timeout)
                return _result(r.returncode == 0, "passed" if r.returncode == 0 else "runtime_error", comp=c, run=r, t0=t0)

            if lang == "java":
                if not (shutil.which("javac") and shutil.which("java")):
                    return _missing("missing_toolchain")
                f = tmp / "Problem.java"; f.write_text(source, encoding="utf-8")
                c = subprocess.run(["javac", "Problem.java"], cwd=tmp, text=True, capture_output=True, timeout=timeout)
                if c.returncode != 0:
                    return _result(False, "compile_error", comp=c, t0=t0)
                r = subprocess.run(["java", "-ea", "Problem"], cwd=tmp, text=True, capture_output=True, timeout=timeout)
                return _result(r.returncode == 0, "passed" if r.returncode == 0 else "runtime_error", comp=c, run=r, t0=t0)

            if lang == "go":
                if not shutil.which("go"):
                    return _missing("missing_toolchain")
                # filename must NOT end in _test.go (that triggers external-test-package rules);
                # MultiPL-E go files are `package X_test` regular files run via `go test <file>`.
                f = tmp / "prog.go"; f.write_text(source, encoding="utf-8")
                env["GO111MODULE"] = "off"                 # loose single-file test, no go.mod
                env.setdefault("GOCACHE", str(tmp / ".gocache"))
                env.setdefault("GOPATH", str(tmp / ".gopath"))
                r = subprocess.run(["go", "test", "prog.go"], cwd=tmp, text=True, capture_output=True,
                                   timeout=timeout, env=env)
                bad = ("FAIL" in r.stdout) or ("[build failed]" in r.stdout) or ("[setup failed]" in r.stdout)
                passed = (r.returncode == 0) and not bad
                status = "passed" if passed else ("compile_error" if "build failed" in r.stdout else "runtime_error")
                return _result(passed, status, run=r, t0=t0)
        except subprocess.TimeoutExpired as exc:
            return {"passed": False, "status": "timeout",
                    "compile_stdout": "", "compile_stderr": str(exc),
                    "run_stdout": _s(getattr(exc, "stdout", "")), "run_stderr": _s(getattr(exc, "stderr", "")),
                    "elapsed_sec": time.time() - t0}
    raise ValueError(f"unsupported language: {lang}")  # pragma: no cover
