import os
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from driftline.config import Settings  # noqa: E402
from driftline.data.ingest import ingest_month  # noqa: E402
from driftline.data.validate import month_bounds  # noqa: E402

ZONES = list(range(1, 13))  # few zones so every pair clears MIN_PAIR_TRIPS


def make_raw_month(month: str, n: int = 6000, seed: int = 0, slowdown: float = 1.0,
                   n_bad: int = 60) -> pl.DataFrame:
    """Synthetic trips in the raw TLC schema. Speed depends on hour and pickup zone, so the
    zone statistics carry real signal; `slowdown` simulates a traffic regime change."""
    rng = np.random.default_rng(seed)
    start, end = month_bounds(month)
    span = (end - start).total_seconds()
    pickup = [start + timedelta(seconds=float(s)) for s in rng.uniform(0, span - 4000, n)]
    pu = rng.choice(ZONES, n)
    do = rng.choice(ZONES, n)
    distance = np.round(rng.gamma(2.0, 1.5, n) + 0.1, 2)
    hours = np.array([p.hour for p in pickup])
    rush = np.isin(hours, [8, 9, 17, 18])
    mph = (14 - 6 * rush - (pu % 5)) / slowdown
    minutes = np.clip(distance / mph * 60 + rng.normal(0, 1.5, n), 1.5, 110)
    dropoff = [p + timedelta(minutes=float(m)) for p, m in zip(pickup, minutes)]
    df = pl.DataFrame({
        "VendorID": pl.Series([2] * n, dtype=pl.Int32),
        "tpep_pickup_datetime": pickup,
        "tpep_dropoff_datetime": dropoff,
        "passenger_count": pl.Series(rng.integers(1, 4, n), dtype=pl.Int64),
        "trip_distance": distance,
        "RatecodeID": pl.Series([1] * n, dtype=pl.Int64),
        "PULocationID": pl.Series(pu, dtype=pl.Int32),
        "DOLocationID": pl.Series(do, dtype=pl.Int32),
    })
    # Rows that each break exactly one validation rule.
    bad = df.head(n_bad).with_columns(
        pl.when(pl.int_range(pl.len()) % 3 == 0).then(0.0)
        .otherwise(pl.col("trip_distance")).alias("trip_distance"),
        pl.when(pl.int_range(pl.len()) % 3 == 1).then(264)
        .otherwise(pl.col("PULocationID")).cast(pl.Int32).alias("PULocationID"),
        pl.when(pl.int_range(pl.len()) % 3 == 2)
        .then(pl.col("tpep_pickup_datetime") + timedelta(hours=5))
        .otherwise(pl.col("tpep_dropoff_datetime")).alias("tpep_dropoff_datetime"),
    )
    return pl.concat([df, bad])


MONTHS = ["2024-01", "2024-02", "2024-03", "2024-04"]


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        warehouse_path=tmp_path / "data" / "warehouse.duckdb",
        mlflow_tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
        mlflow_experiment="test",
        model_name="test-model",
        train_sample_rows=5000,
        eval_sample_rows=3000,
        routing_mode="shadow",
    )


@pytest.fixture
def warehouse(settings) -> Settings:
    settings.raw_dir.mkdir(parents=True)
    for i, m in enumerate(MONTHS):
        make_raw_month(m, seed=i).write_parquet(settings.raw_dir / f"yellow_tripdata_{m}.parquet")
        ingest_month(m, settings.raw_dir, settings.warehouse_path)
    return settings


@pytest.fixture
def trained(warehouse) -> Settings:
    """Warehouse + one promoted champion (trained Feb-Mar, evaluated Apr)."""
    from driftline.training import registry
    from driftline.training.train import features_for_month, train

    train(warehouse, ["2024-02", "2024-03"], "2024-04")
    registry.promote_if_better(warehouse, features_for_month(warehouse, "2024-04", 3000))
    return warehouse


def sample_trip(**overrides) -> dict:
    trip = {"pickup_at": datetime(2024, 5, 3, 8, 30).isoformat(), "pu_location_id": 3,
            "do_location_id": 7, "trip_distance": 2.4, "passenger_count": 1}
    return trip | overrides
