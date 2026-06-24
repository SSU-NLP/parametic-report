"""Unit tests for tiny-codes language alias handling (C#/C++ -> filesystem-safe)."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "create_code_dataset",
    Path(__file__).resolve().parents[1] / "data_preprocess" / "create_code_dataset.py",
)
ccd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ccd)


def test_safe_label_aliases_special_chars():
    assert ccd.safe_lang_label("C#") == "csharp"
    assert ccd.safe_lang_label("C++") == "cpp"


def test_safe_label_passthrough():
    for lang in ["Java", "Go", "Rust", "TypeScript", "Bash"]:
        assert ccd.safe_lang_label(lang) == lang.lower()


def test_normalize_accepts_alias_and_maps_to_dataset_label():
    # alias inputs (csharp/cpp) must filter on the dataset's real labels (c#/c++)
    assert ccd.normalize_languages(["csharp", "cpp"]) == {"c#", "c++"}


def test_normalize_accepts_raw_dataset_labels():
    assert ccd.normalize_languages(["C#", "Java"]) == {"c#", "java"}


def test_normalize_string_and_none():
    assert ccd.normalize_languages("java,go") == {"java", "go"}
    assert ccd.normalize_languages(None) is None
