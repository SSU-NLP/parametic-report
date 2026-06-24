#!/usr/bin/env python3
"""Emit a runner job_spec.json for a transplant calibration run.

The transplant experiment needs each model's java grad·param scores. Rather than
reimplement dataset→preprocess→accumulate, we reuse the *validated* platform runner:
this builds the exact job_spec.json the runner consumes (catalog-synced area/mode), for
an arbitrary HF model passed on the CLI (the transplant models are not in MODEL_CATALOG).

The calibration driver uploads this spec, runs `python -m parametic_platform.runner`,
then copies scratch_root/calibration/<model_output_name> to a durable /shared location.
"""
import argparse
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from parametic_platform import catalog  # noqa: E402
from parametic_platform.catalog import ModelSpec  # noqa: E402
from parametic_platform.spec import build_analysis_spec, cache_key  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", required=True)
    p.add_argument("--hf-model", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--model-output-name", required=True)
    p.add_argument("--expected-tensors", type=int, required=True)
    p.add_argument("--config-path", default="config.json")
    p.add_argument("--area", default="java-code-smoke")
    p.add_argument("--mode", default="approx-1024")
    p.add_argument("--k", type=float, default=0.01)
    p.add_argument("--repo-root", required=True)
    p.add_argument("--artifact-root", required=True)
    p.add_argument("--scratch-root", required=True)
    p.add_argument("--seeds", type=int, nargs="*", default=None,
                   help="override the mode's calibration seeds (e.g. single seed 1234 for paper-repro)")
    a = p.parse_args()

    model = ModelSpec(
        id=a.model_id,
        display_name=a.model_id,
        hf_model_id=a.hf_model,
        config_path=a.config_path,
        tokenizer_path=a.tokenizer,
        model_output_name=a.model_output_name,
        expected_tensors=a.expected_tensors,
    )
    area = catalog.AREA_CATALOG[a.area]
    mode = catalog.MODE_CATALOG[a.mode]
    if a.seeds:
        from dataclasses import replace
        mode = replace(mode, seeds=tuple(a.seeds))
    analysis = build_analysis_spec(model=model, area=area, mode=mode, k=a.k, pipeline_version="approx-mri-v1")

    spec = {
        "request_id": str(uuid.uuid4()),
        "job_id": str(uuid.uuid4()),
        "cache_key": cache_key(analysis),
        "analysis": analysis,
        "paths": {
            "repo_root": a.repo_root,
            "artifact_root": a.artifact_root,
            "scratch_root": a.scratch_root,
        },
    }
    print(json.dumps(spec, indent=2))


if __name__ == "__main__":
    main()
