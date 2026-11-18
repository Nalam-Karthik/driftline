"""Feature engineering shared by training (vectorised, millions of rows) and serving (one row).

Zone statistics are point-in-time: features for month M are always computed from month M-1,
so a training row never sees aggregates that include its own target.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

AIRPORT_ZONES = (1, 132, 138)  # EWR, JFK, LaGuardia
MIN_PAIR_TRIPS = 20

CATEGORICAL = ["pu_location_id", "do_location_id", "ratecode_id"]
FEATURES = [
    "pu_location_id", "do_location_id", "ratecode_id",
    "trip_distance", "passenger_count",
    "hour", "dow", "is_weekend", "is_airport", "same_zone",
    "pair_median_min", "pair_trips", "pu_hour_median_mph", "pu_hour_trips",
]
TARGET = "duration_min"


def base_features(df: pl.DataFrame) -> pl.DataFrame:
    """Features derived only from the request itself."""
    return df.with_columns(
        pl.col("pickup_at").dt.hour().cast(pl.Int8).alias("hour"),
        (pl.col("pickup_at").dt.weekday() - 1).cast(pl.Int8).alias("dow"),
        (pl.col("pickup_at").dt.weekday() >= 6).cast(pl.Int8).alias("is_weekend"),
        (pl.col("pu_location_id").is_in(AIRPORT_ZONES)
         | pl.col("do_location_id").is_in(AIRPORT_ZONES)).cast(pl.Int8).alias("is_airport"),
        (pl.col("pu_location_id") == pl.col("do_location_id")).cast(pl.Int8).alias("same_zone"),
    )


@dataclass
class ZoneStats:
    """Aggregates from one historical window, keyed for both joins and online lookups."""

    source_month: str
    pair: pl.DataFrame     # pu, do -> pair_median_min, pair_trips
    pu_hour: pl.DataFrame  # pu, hour -> pu_hour_median_mph, pu_hour_trips

    @classmethod
    def compute(cls, trips: pl.DataFrame, source_month: str) -> "ZoneStats":
        pair = (
            trips.group_by("pu_location_id", "do_location_id")
            .agg(pl.col("duration_min").median().alias("pair_median_min"),
                 pl.len().cast(pl.Int32).alias("pair_trips"))
            .filter(pl.col("pair_trips") >= MIN_PAIR_TRIPS)
        )
        mph = pl.col("trip_distance") / (pl.col("duration_min") / 60.0)
        pu_hour = (
            trips.with_columns(pl.col("pickup_at").dt.hour().cast(pl.Int8).alias("hour"))
            .group_by("pu_location_id", "hour")
            .agg(mph.median().alias("pu_hour_median_mph"),
                 pl.len().cast(pl.Int32).alias("pu_hour_trips"))
        )
        return cls(source_month, pair, pu_hour)

    def join(self, df: pl.DataFrame) -> pl.DataFrame:
        return (
            df.join(self.pair, on=["pu_location_id", "do_location_id"], how="left")
            .join(self.pu_hour, on=["pu_location_id", "hour"], how="left")
        )

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.pair.write_parquet(directory / "pair.parquet")
        self.pu_hour.write_parquet(directory / "pu_hour.parquet")
        (directory / "SOURCE_MONTH").write_text(self.source_month)

    @classmethod
    def load(cls, directory: Path) -> "ZoneStats":
        return cls(
            (directory / "SOURCE_MONTH").read_text().strip(),
            pl.read_parquet(directory / "pair.parquet"),
            pl.read_parquet(directory / "pu_hour.parquet"),
        )


def build_training_frame(trips: pl.DataFrame, stats: ZoneStats) -> pl.DataFrame:
    return stats.join(base_features(trips)).select(FEATURES + [TARGET])


CATEGORICAL_INDEX = [FEATURES.index(c) for c in CATEGORICAL]


def to_model_input(df: pl.DataFrame) -> np.ndarray:
    """Float matrix in FEATURES order; nulls become NaN, which LightGBM treats as missing.

    Categoricals stay as raw integer ids (LightGBM's native categorical handling), so there is
    no category-to-code mapping that could differ between training and serving.
    """
    return df.select(pl.col(FEATURES).cast(pl.Float64)).to_numpy()


def previous_month(month: str) -> str:
    year, mon = map(int, month.split("-"))
    year, mon = (year - 1, 12) if mon == 1 else (year, mon - 1)
    return f"{year:04d}-{mon:02d}"
