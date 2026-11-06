"""SQLAlchemy 2 ORM tables for persisted scores, segments and monitoring (plan D6).

The same ORM works against PostgreSQL (Docker) and SQLite (tests), so the batch
scorer and the API share one schema definition.
"""

from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, Integer, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Customer(Base):
    __tablename__ = "customers"

    customer_id: Mapped[float] = mapped_column(Float, primary_key=True)
    primary_country: Mapped[str] = mapped_column(String(64), default="United Kingdom")
    first_seen: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    customer_id: Mapped[float] = mapped_column(Float, index=True)
    model_version: Mapped[str] = mapped_column(String(64), index=True)
    churn_probability: Mapped[float] = mapped_column(Float)
    calibrated_probability: Mapped[float] = mapped_column(Float)
    churn_prediction: Mapped[int] = mapped_column(Integer)
    risk_tier: Mapped[str] = mapped_column(String(16))
    predicted_value: Mapped[float] = mapped_column(Float)
    value_at_risk: Mapped[float] = mapped_column(Float, index=True)
    value_tier: Mapped[str] = mapped_column(String(16), default="mid")
    anomaly_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    anomaly_flag: Mapped[int] = mapped_column(Integer, default=0)
    shap_top: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    features: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    scored_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())



class Segment(Base):
    __tablename__ = "segments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    customer_id: Mapped[float] = mapped_column(Float, index=True)
    kmeans_cluster: Mapped[int] = mapped_column(Integer)
    gmm_cluster: Mapped[int] = mapped_column(Integer)
    persona: Mapped[str] = mapped_column(String(128), index=True)



class Recommendation(Base):
    __tablename__ = "recommendations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    customer_id: Mapped[float] = mapped_column(Float, index=True)
    action: Mapped[str] = mapped_column(String(256))
    channel: Mapped[str] = mapped_column(String(64))
    rule_id: Mapped[str] = mapped_column(String(64))
    reasons: Mapped[list | None] = mapped_column(JSON, nullable=True)
    priority_score: Mapped[float] = mapped_column(Float, index=True)



class ModelRegistry(Base):
    __tablename__ = "model_registry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_version: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    churn_model: Mapped[str] = mapped_column(String(128))
    clv_model: Mapped[str] = mapped_column(String(128))
    decision_threshold: Mapped[float] = mapped_column(Float)
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    registered_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class PredictionLog(Base):
    __tablename__ = "prediction_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    customer_id: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_version: Mapped[str] = mapped_column(String(64))
    latency_ms: Mapped[float] = mapped_column(Float)
    endpoint: Mapped[str] = mapped_column(String(64), default="/predict")
    logged_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class DriftReport(Base):
    __tablename__ = "drift_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    feature: Mapped[str] = mapped_column(String(64), index=True)
    psi: Mapped[float] = mapped_column(Float)
    drifted: Mapped[int] = mapped_column(Integer, default=0)
    batch_date: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class CustomerProduct(Base):
    __tablename__ = "customer_products"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    customer_id: Mapped[float] = mapped_column(Float, index=True)
    product_id: Mapped[str] = mapped_column(String(32), index=True)
    quantity: Mapped[float] = mapped_column(Float, default=0.0)
