from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from api.database import Base


class Session(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    channel_csv_path = Column(String, nullable=True)
    campaign_csv_path = Column(String, nullable=True)
    config_json = Column(Text, nullable=True)

    model_runs = relationship("ModelRun", back_populates="session")
    scenarios = relationship("Scenario", back_populates="session")
    tuning_configs = relationship("TuningConfig", back_populates="session")


class ModelRun(Base):
    __tablename__ = "model_runs"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    model_num = Column(Integer, default=1)
    iteration_num = Column(Integer, default=1)
    name = Column(String, nullable=True)
    status = Column(String, default="pending")  # pending/running/complete/failed
    error_message = Column(Text, nullable=True)

    adj_r2 = Column(Float, nullable=True)
    mape = Column(Float, nullable=True)
    rhat_pass_pct = Column(Float, nullable=True)
    confidence_width = Column(Float, nullable=True)
    contributions_json = Column(Text, nullable=True)
    metrics_json = Column(Text, nullable=True)
    idata_path = Column(String, nullable=True)
    fit_config_json = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("Session", back_populates="model_runs")
    scenarios = relationship("Scenario", back_populates="model_run")


class Scenario(Base):
    __tablename__ = "scenarios"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    model_run_id = Column(Integer, ForeignKey("model_runs.id"), nullable=True)
    name = Column(String, nullable=False)
    scenario_type = Column(String)  # forward / reverse
    inputs_json = Column(Text, nullable=True)
    results_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("Session", back_populates="scenarios")
    model_run = relationship("ModelRun", back_populates="scenarios")


class TuningConfig(Base):
    __tablename__ = "tuning_configs"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    holidays_json = Column(Text, default="[]")
    seasonality_json = Column(Text, default="{}")
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("Session", back_populates="tuning_configs")
