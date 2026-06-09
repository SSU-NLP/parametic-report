from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

from sqlalchemy import select

from .config import load_settings
from .db import AnalysisRequest, Job, init_db, make_session_factory


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


def write_job_spec(settings, request: AnalysisRequest, job: Job) -> tuple[Path, Path, Path]:
    artifact_root = Path(request.artifact_root or settings.artifact_root / request.cache_key)
    scratch_root = settings.scratch_root / job.id
    log_dir = artifact_root / "logs"
    artifact_root.mkdir(parents=True, exist_ok=True)
    scratch_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    host_spec_path = artifact_root / "job_spec.json"
    runner_spec_path = Path(settings.runner_artifact_root) / request.cache_key / "job_spec.json"
    spec = {
        "request_id": request.id,
        "job_id": job.id,
        "cache_key": request.cache_key,
        "analysis": request.spec,
        "paths": {
            "repo_root": settings.runner_repo_root,
            "artifact_root": f"{settings.runner_artifact_root}/{request.cache_key}",
            "scratch_root": f"{settings.runner_scratch_root}/{job.id}",
        },
    }
    import json
    host_spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return host_spec_path, runner_spec_path, log_dir / "worker.log"


def docker_command(settings, job: Job, runner_spec_path: Path) -> list[str]:
    name = f"parametic-job-{job.id[:12]}"
    job.container_name = name
    cmd = [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        "--gpus",
        f"device={settings.gpu_device}",
        "--ipc=host",
        "--shm-size",
        settings.docker_shm_size,
        "-v",
        f"{settings.repo_root}:{settings.runner_repo_root}",
        "-v",
        f"{settings.artifact_root}:{settings.runner_artifact_root}",
        "-v",
        f"{settings.scratch_root}:{settings.runner_scratch_root}",
        "-v",
        f"{settings.hf_cache_dir}:{settings.runner_hf_cache_dir}",
        "-e",
        f"HF_HOME={settings.runner_hf_cache_dir}",
        "-e",
        "PYTHONUNBUFFERED=1",
    ]
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if hf_token:
        cmd.extend(["-e", f"HF_TOKEN={hf_token}"])
    cmd.extend([
        settings.runner_image,
        "python",
        "-m",
        "parametic_platform.runner",
        "--job-spec",
        str(runner_spec_path),
    ])
    return cmd


def read_failure_detail(artifact_root: Path, rc: int) -> str:
    """Surface which runner stage failed, from the error.json breadcrumb."""
    fallback = f"runner exited with code {rc}"
    error_path = artifact_root / "error.json"
    if not error_path.exists():
        return fallback
    try:
        import json

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
        return f"{detail} (exit code {rc})"
    return fallback


def run_job(settings, session, job: Job) -> None:
    request = session.get(AnalysisRequest, job.request_id)
    if request is None:
        raise RuntimeError(f"request not found: {job.request_id}")
    _, runner_spec_path, worker_log = write_job_spec(settings, request, job)
    job.log_path = str(worker_log)
    session.commit()

    cmd = docker_command(settings, job, runner_spec_path)
    job.stage = "docker_run"
    session.commit()
    with worker_log.open("ab") as log:
        log.write((" ".join(cmd) + "\n").encode("utf-8"))
        process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        rc = process.wait()

    artifact_root = Path(request.artifact_root or settings.artifact_root / request.cache_key)
    manifest_path = artifact_root / "manifest.json"
    job.finished_at = datetime.utcnow()
    if rc == 0 and manifest_path.exists():
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
        job.error = read_failure_detail(artifact_root, rc)
        request.status = "failed"
        request.error = job.error
        if not settings.keep_scratch_on_failure:
            shutil.rmtree(settings.scratch_root / job.id, ignore_errors=True)
    session.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Parametic Report Docker worker")
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
