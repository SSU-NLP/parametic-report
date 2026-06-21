from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

from .config import PlatformSettings


class Base(DeclarativeBase):
    pass


def now_utc() -> datetime:
    return datetime.utcnow()


class AnalysisRequest(Base):
    __tablename__ = "analysis_requests"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    cache_key: Mapped[str] = mapped_column(String(64), index=True)
    model_id: Mapped[str] = mapped_column(String(128), index=True)
    area_id: Mapped[str] = mapped_column(String(128), index=True)
    mode: Mapped[str] = mapped_column(String(128), index=True)
    k: Mapped[float] = mapped_column(Float)
    sample_size: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(32), index=True, default="queued")
    spec: Mapped[dict] = mapped_column(JSON)
    artifact_root: Mapped[str | None] = mapped_column(Text)
    manifest_path: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_utc)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=now_utc, onupdate=now_utc)

    jobs: Mapped[list["Job"]] = relationship(back_populates="request")


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    request_id: Mapped[str] = mapped_column(ForeignKey("analysis_requests.id"), index=True)
    cache_key: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(32), index=True, default="queued")
    stage: Mapped[str | None] = mapped_column(String(128))
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    container_name: Mapped[str | None] = mapped_column(String(128))
    log_path: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_utc)
    started_at: Mapped[datetime | None] = mapped_column(DateTime)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=now_utc, onupdate=now_utc)

    request: Mapped[AnalysisRequest] = relationship(back_populates="jobs")


class RegisteredModel(Base):
    """Operator-registered HF model. Mirrors catalog.ModelSpec fields so a resolved
    registration feeds the existing spec/cache_key builder unchanged. The static
    MODEL_CATALOG stays the curated source of truth; these accumulate alongside it."""

    __tablename__ = "registered_models"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    display_name: Mapped[str] = mapped_column(String(256))
    hf_model_id: Mapped[str] = mapped_column(String(256), index=True)
    config_path: Mapped[str] = mapped_column(Text)
    tokenizer_path: Mapped[str] = mapped_column(Text)
    model_output_name: Mapped[str] = mapped_column(String(256))
    expected_tensors: Mapped[int] = mapped_column(Integer)
    revision: Mapped[str] = mapped_column(String(128), default="main")
    # Provenance / compatibility captured at resolve time (model_type, params, status…).
    model_type: Mapped[str | None] = mapped_column(String(128))
    compatibility: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_utc)


def make_engine(settings: PlatformSettings):
    return create_engine(settings.database_url, pool_pre_ping=True)


def make_session_factory(settings: PlatformSettings) -> sessionmaker[Session]:
    return sessionmaker(make_engine(settings), expire_on_commit=False)


def init_db(settings: PlatformSettings) -> None:
    engine = make_engine(settings)
    Base.metadata.create_all(engine)


def find_completed_request(session: Session, key: str) -> AnalysisRequest | None:
    stmt = (
        select(AnalysisRequest)
        .where(AnalysisRequest.cache_key == key, AnalysisRequest.status == "succeeded")
        .order_by(AnalysisRequest.updated_at.desc())
        .limit(1)
    )
    return session.execute(stmt).scalar_one_or_none()


def find_active_request(session: Session, key: str) -> AnalysisRequest | None:
    stmt = (
        select(AnalysisRequest)
        .where(AnalysisRequest.cache_key == key, AnalysisRequest.status.in_(["queued", "running"]))
        .order_by(AnalysisRequest.created_at.asc())
        .limit(1)
    )
    return session.execute(stmt).scalar_one_or_none()
