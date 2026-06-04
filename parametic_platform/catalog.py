from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    id: str
    display_name: str
    hf_model_id: str
    config_path: str
    tokenizer_path: str
    model_output_name: str
    expected_tensors: int
    revision: str = "main"


@dataclass(frozen=True)
class AreaSpec:
    id: str
    display_name: str
    language: str
    dataset_name: str
    hf_dataset_name: str
    total_examples: int
    train_ratio: float
    revision: str = "streaming-main"


@dataclass(frozen=True)
class ModeSpec:
    id: str
    display_name: str
    sample_size: int
    seeds: tuple[int, ...]
    k: float
    random_seeds: tuple[int, ...]
    ppl_samples: int
    tile_size: int
    full_run: bool = False
    public: bool = True


MODEL_CATALOG: dict[str, ModelSpec] = {
    "llama-3.2-3b": ModelSpec(
        id="llama-3.2-3b",
        display_name="Llama 3.2 3B Instruct",
        hf_model_id="meta-llama/Llama-3.2-3B-Instruct",
        config_path="config.json",
        tokenizer_path="data_preprocess/tokenizers/llama-3.2",
        model_output_name="Llama-3.2-3B-Instruct",
        expected_tensors=254,
    ),
    "qwen3-8b": ModelSpec(
        id="qwen3-8b",
        display_name="Qwen3 8B",
        hf_model_id="Qwen/Qwen3-8B",
        config_path="config.qwen3-8b.json",
        tokenizer_path="Qwen/Qwen3-8B",
        model_output_name="Qwen3-8B",
        expected_tensors=399,
    ),
}


AREA_CATALOG: dict[str, AreaSpec] = {
    "java-code": AreaSpec(
        id="java-code",
        display_name="Java code spot",
        language="java",
        dataset_name="tiny-codes-java-full",
        hf_dataset_name="nampdn-ai/tiny-codes",
        total_examples=1_630_000,
        train_ratio=0.8,
    ),
}


MODE_CATALOG: dict[str, ModeSpec] = {
    "approx-smoke": ModeSpec(
        id="approx-smoke",
        display_name="Approx MRI smoke test",
        sample_size=8,
        seeds=(1234, 5678),
        k=0.01,
        random_seeds=(1,),
        ppl_samples=2,
        tile_size=16,
        public=False,
    ),
    "approx-1024": ModeSpec(
        id="approx-1024",
        display_name="Approx MRI, 1024 samples",
        sample_size=1024,
        seeds=(1234, 5678),
        k=0.01,
        random_seeds=(1, 2, 3),
        ppl_samples=128,
        tile_size=16,
    ),
    "approx-2048": ModeSpec(
        id="approx-2048",
        display_name="Approx MRI, 2048 samples",
        sample_size=2048,
        seeds=(1234, 5678),
        k=0.01,
        random_seeds=(1, 2, 3),
        ppl_samples=128,
        tile_size=16,
        public=False,
    ),
}


def get_model(model_id: str) -> ModelSpec:
    try:
        return MODEL_CATALOG[model_id]
    except KeyError as exc:
        raise ValueError(f"Unsupported model_id: {model_id}") from exc


def get_area(area_id: str) -> AreaSpec:
    try:
        return AREA_CATALOG[area_id]
    except KeyError as exc:
        raise ValueError(f"Unsupported area_id: {area_id}") from exc


def get_mode(mode_id: str) -> ModeSpec:
    try:
        return MODE_CATALOG[mode_id]
    except KeyError as exc:
        raise ValueError(f"Unsupported mode: {mode_id}") from exc


def public_models() -> list[dict[str, Any]]:
    return [asdict(item) for item in MODEL_CATALOG.values()]


def public_areas() -> list[dict[str, Any]]:
    return [asdict(item) for item in AREA_CATALOG.values()]


def public_modes() -> list[dict[str, Any]]:
    return [asdict(item) for item in MODE_CATALOG.values() if item.public]
