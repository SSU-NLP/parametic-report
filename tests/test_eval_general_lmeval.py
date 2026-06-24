"""Unit tests for lm-eval result parsing (pure, no GPU/lm-eval install)."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "eval_general_lmeval",
    Path(__file__).resolve().parents[1] / "scripts" / "eval_general_lmeval.py",
)
egl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(egl)


# lm-eval results json shape: {"results": {task: {"<metric>,<filter>": value, ...}}}
SAMPLE = {
    "results": {
        "winogrande": {"acc,none": 0.62, "acc_stderr,none": 0.013},
        "gsm8k": {
            "exact_match,strict-match": 0.11,
            "exact_match,flexible-extract": 0.12,
            "exact_match_stderr,strict-match": 0.008,
        },
        "hellaswag": {"acc,none": 0.40, "acc_norm,none": 0.55, "acc_norm_stderr,none": 0.01},
        "mmlu": {"acc,none": 0.43, "acc_stderr,none": 0.004},
        "truthfulqa_mc2": {"acc,none": 0.41, "acc_stderr,none": 0.014},
    }
}


def test_extracts_primary_metric_per_task():
    out = egl.extract_lmeval_scores(SAMPLE, egl.PRIMARY_METRIC)
    assert out["winogrande"] == 0.62
    assert out["hellaswag"] == 0.55   # acc_norm, not acc
    assert out["gsm8k"] == 0.11       # exact_match (first / strict-match)
    assert out["mmlu"] == 0.43
    assert out["truthfulqa_mc2"] == 0.41


def test_ignores_stderr_keys():
    out = egl.extract_lmeval_scores({"results": {"hellaswag": {"acc_norm_stderr,none": 0.01, "acc_norm,none": 0.5}}},
                                    {"hellaswag": "acc_norm"})
    assert out["hellaswag"] == 0.5


def test_missing_task_skipped():
    out = egl.extract_lmeval_scores({"results": {"gsm8k": {"exact_match,strict-match": 0.1}}}, egl.PRIMARY_METRIC)
    assert out == {"gsm8k": 0.1}


def test_fewshot_map_has_five_paper_tasks():
    assert set(egl.FEWSHOT) == {"gsm8k", "hellaswag", "mmlu", "truthfulqa_mc2", "winogrande"}
    assert egl.FEWSHOT["hellaswag"] == 10 and egl.FEWSHOT["gsm8k"] == 5 and egl.FEWSHOT["truthfulqa_mc2"] == 0
