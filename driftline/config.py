from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration. Every field can be overridden with a DRIFTLINE_* env var."""

    model_config = SettingsConfigDict(env_prefix="DRIFTLINE_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    warehouse_path: Path = Path("data/warehouse.duckdb")

    mlflow_tracking_uri: str = "sqlite:///data/mlflow.db"
    mlflow_experiment: str = "trip-duration"
    model_name: str = "trip-duration"

    redis_url: str | None = None  # unset -> in-process stores (tests, local dev)

    # Rows sampled per month for training; the full month is ~3M rows.
    train_sample_rows: int = 750_000
    eval_sample_rows: int = 250_000
    seed: int = 7

    # Challenger must beat champion MAE by this relative margin to be promoted.
    promotion_min_improvement: float = 0.01

    # Serving
    routing_mode: str = "shadow"  # "shadow" | "canary" | "champion"
    canary_fraction: float = 0.1
    registry_poll_seconds: int = 30

    # Monitoring
    psi_alert_threshold: float = 0.2
    mae_degradation_threshold: float = 0.10  # relative to MAE at training time
    drift_window: int = 20_000  # most recent logged predictions to test

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def features_dir(self) -> Path:
        return self.data_dir / "features"


@lru_cache
def get_settings() -> Settings:
    return Settings()
