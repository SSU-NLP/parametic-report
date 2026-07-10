from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

from sqlalchemy import select

from .config import load_settings
from .db import AnalysisRequest, Job, init_db, make_session_factory

TERMINAL_STATES = {"succeeded", "failed", "terminated"}


def claim_job(session) -> Job | None:
    stmt = select(Job).where(Job.status == "queued").order_by(Job.created_at.asc()).limit(1)
    job = session.execute(stmt).scalar_one_or_none()
    if job is None:
        return None
    job.status = "running"
    job.stage = "claimed"
    job.attempts += 1
    job.started_at = datetime.utcnow()
    job.request.status = "running"
    session.commit()
    return job


def host_artifact_root(settings, request: AnalysisRequest) -> Path:
    """Host-local dir the finished results are downloaded to and the API serves from."""
    return Path(request.artifact_root or settings.artifact_root / request.cache_key)


def vessl_container_paths(settings, request: AnalysisRequest, job: Job) -> dict[str, str]:
    """Container-side paths inside the VESSL job (clean mounts — no host aliasing).

    Code is synced to the fast cluster volume (/work); results live on the S3-backed
    object volume (/shared) so they survive job teardown and are downloadable. The HF
    cache must stay on /work too: HuggingFace stores files as blobs symlinked into
    snapshots/, and the S3-backed /shared mount does NOT support symlinks (the snapshot
    entries land as dead 0-byte files), so a model cached on /shared is unreadable.
    /work is a real SSD filesystem (symlinks work) and is betelgeuse-local, so the
    cache still persists across jobs on the cluster we actually run on.
    """
    ns = settings.vessl_ns
    obj = settings.vessl_object_mnt
    work = settings.vessl_cluster_mnt
    return {
        "code": f"{work}/{ns}/code",
        "artifact_root": f"{obj}/{ns}/results/{request.cache_key}",
        "scratch_root": f"{work}/{ns}/scratch/{job.id}",
        "hf_cache": f"{work}/{ns}/hf-cache",
        "job_spec": f"{obj}/{ns}/jobs/{job.id}/job_spec.json",
        "spec_remote_prefix": f"{ns}/jobs/{job.id}",
        "results_remote_prefix": f"{ns}/results/{request.cache_key}",
    }


def write_job_spec(settings, request: AnalysisRequest, job: Job) -> tuple[Path, dict[str, str], Path]:
    """Stage job_spec.json locally and compute the VESSL container paths.

    Returns (host_spec_path, container_paths, worker_log_path). The runner reads the
    same job_spec.json schema as before — only the path *values* are VESSL-side now.
    """
    artifact_root = host_artifact_root(settings, request)
    staging = settings.scratch_root / job.id
    log_dir = artifact_root / "logs"
    artifact_root.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    cp = vessl_container_paths(settings, request, job)
    spec = {
        "request_id": request.id,
        "job_id": job.id,
        "cache_key": request.cache_key,
        "analysis": request.spec,
        "paths": {
            "repo_root": cp["code"],
            "artifact_root": cp["artifact_root"],
            "scratch_root": cp["scratch_root"],
        },
    }
    host_spec_path = staging / "job_spec.json"
    host_spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return host_spec_path, cp, log_dir / "worker.log"


def _run_logged(cmd: list[str], worker_log: Path, *, check: bool = True) -> str:
    """Run a command, append combined output to worker.log, return stdout text."""
    with worker_log.open("ab") as log:
        log.write(("\n$ " + " ".join(cmd) + "\n").encode("utf-8"))
    result = subprocess.run(cmd, capture_output=True, text=True)
    combined = (result.stdout or "") + (result.stderr or "")
    with worker_log.open("ab") as log:
        log.write(combined.encode("utf-8", errors="replace"))
    if check and result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(cmd[:3])} …")
    return result.stdout or ""


def vessl_submit(settings, request: AnalysisRequest, job: Job, host_spec_path: Path,
                 cp: dict[str, str], worker_log: Path) -> str:
    """Upload the per-job spec to the object volume, submit via scripts/vessl/submit.sh,
    and return the VESSL job slug (job-...)."""
    _run_logged(
        ["vesslctl", "volume", "upload", settings.vessl_object_vol, str(host_spec_path),
         "--remote-prefix", cp["spec_remote_prefix"], "--overwrite"],
        worker_log,
    )

    preamble = (
        f"export HF_HOME={cp['hf_cache']}; "
        f"export PARAMETIC_RUNNER_PYTHON=python; "
        f"export PARAMETIC_IGNORE_REPO_VENV=1; "
        f"export PYTHONUNBUFFERED=1; "
        f"mkdir -p {cp['hf_cache']}; "
    )
    runner_cmd = f"python -m parametic_platform.runner --job-spec {cp['job_spec']}"
    full_cmd = preamble + runner_cmd

    submit = settings.repo_root / "scripts" / "vessl" / "submit.sh"
    args = [
        "bash", str(submit),
        "--name", f"parametic-{job.id[:12]}",
        "--gpus", str(settings.vessl_gpu_count),
        "--image", settings.vessl_image,
        "--pip", "-r requirements-runner.txt",
        "--tag", "parametic",
    ]
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if hf_token:
        args += ["--env", f"HF_TOKEN={hf_token}"]
    args += ["--cmd", full_cmd]

    out = _run_logged(args, worker_log)
    # submit.sh echoes the full --cmd, which contains `--job-spec`; a bare
    # `job-[a-z0-9]+` search would greedily match "job-spec". Anchor on the
    # authoritative `slug: job-...` line submit.sh prints last instead.
    matches = re.findall(r"slug:\s*(job-[a-z0-9]+)", out)
    if not matches:
        raise RuntimeError("could not parse VESSL job slug from submit output")
    return matches[-1]


def vessl_wait(settings, slug: str, worker_log: Path) -> str:
    """Poll the job to a terminal state via scripts/vessl/watch.sh; return the final state."""
    watch = settings.repo_root / "scripts" / "vessl" / "watch.sh"
    out = _run_logged(["bash", str(watch), slug, str(settings.vessl_poll_max)], worker_log, check=False)
    match = re.search(r"=== FINAL: (\S+) ===", out)
    return (match.group(1) if match else "unknown").strip().lower()


def _locate_result_root(base: Path) -> Path | None:
    """Find the dir containing manifest.json or error.json under a downloaded tree."""
    for name in ("manifest.json", "error.json"):
        direct = base / name
        if direct.is_file():
            return base
    for marker in ("manifest.json", "error.json"):
        for found in base.rglob(marker):
            return found.parent
    return None


def vessl_fetch_artifacts(settings, request: AnalysisRequest, worker_log: Path) -> None:
    """Download the job's results from the object volume into the host-local artifact_root.

    Downloads into a temp dir first (vesslctl may nest under the remote prefix), locates
    the result root by its manifest/error marker, then materializes it at the exact path
    the API serves. Retries briefly to tolerate S3 flush lag after job completion."""
    dest = host_artifact_root(settings, request)
    remote = f"{settings.vessl_ns}/results/{request.cache_key}"
    tmp = dest.parent / f".dl-{request.cache_key}"

    root: Path | None = None
    for _ in range(3):
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True, exist_ok=True)
        _run_logged(
            ["vesslctl", "volume", "download", settings.vessl_object_vol, str(tmp),
             "--remote-prefix", remote, "--overwrite"],
            worker_log, check=False,
        )
        root = _locate_result_root(tmp)
        if root is not None:
            break
        time.sleep(2)

    if root is not None:
        dest.mkdir(parents=True, exist_ok=True)
        for item in root.iterdir():
            target = dest / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)
    shutil.rmtree(tmp, ignore_errors=True)


def read_failure_detail(artifact_root: Path, reason: str) -> str:
    """Surface which runner stage failed, from the error.json breadcrumb."""
    fallback = f"vessl job did not succeed (state: {reason})"
    error_path = artifact_root / "error.json"
    if not error_path.exists():
        return fallback
    try:
        data = json.loads(error_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return fallback
    error = data.get("error") or {}
    stage = error.get("stage")
    message = error.get("message")
    if stage:
        detail = f"stage '{stage}' failed"
        if message:
            detail += f": {message.strip().splitlines()[-1][:300]}"
        return f"{detail} (job state: {reason})"
    return fallback


def run_job(settings, session, job: Job) -> None:
    request = session.get(AnalysisRequest, job.request_id)
    if request is None:
        raise RuntimeError(f"request not found: {job.request_id}")
    host_spec_path, cp, worker_log = write_job_spec(settings, request, job)
    job.log_path = str(worker_log)
    session.commit()

    state = "failed"
    try:
        job.stage = "vessl_submit"
        session.commit()
        slug = vessl_submit(settings, request, job, host_spec_path, cp, worker_log)
        job.container_name = slug
        session.commit()

        job.stage = "vessl_wait"
        session.commit()
        state = vessl_wait(settings, slug, worker_log)

        job.stage = "vessl_fetch"
        session.commit()
        vessl_fetch_artifacts(settings, request, worker_log)
    except Exception as exc:  # noqa: BLE001 — any dispatch failure ⇒ job failed
        with worker_log.open("ab") as log:
            log.write(f"\n[worker] dispatch error: {exc}\n".encode("utf-8"))
        state = "failed"

    artifact_root = host_artifact_root(settings, request)
    manifest_path = artifact_root / "manifest.json"
    job.finished_at = datetime.utcnow()
    if state == "succeeded" and manifest_path.exists():
        job.status = "succeeded"
        job.stage = "complete"
        request.status = "succeeded"
        request.manifest_path = str(manifest_path)
        request.error = None
        if not settings.keep_scratch_on_success:
            shutil.rmtree(settings.scratch_root / job.id, ignore_errors=True)
    else:
        job.status = "failed"
        job.stage = "failed"
        job.error = read_failure_detail(artifact_root, state)
        request.status = "failed"
        request.error = job.error
        if not settings.keep_scratch_on_failure:
            shutil.rmtree(settings.scratch_root / job.id, ignore_errors=True)
    session.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Parametic Report VESSL worker")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--init-db", action="store_true")
    args = parser.parse_args()

    settings = load_settings()
    if args.init_db:
        init_db(settings)
    SessionFactory = make_session_factory(settings)

    while True:
        with SessionFactory() as session:
            job = claim_job(session)
            if job is not None:
                run_job(settings, session, job)
            elif args.once:
                return
        if args.once:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
