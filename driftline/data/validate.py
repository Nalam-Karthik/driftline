"""Row-level data-quality rules plus a Pandera contract for what reaches the warehouse.

Bad rows are not silently dropped: each one is tagged with the first rule it broke and
written to a `quarantine` table, so data-quality regressions show up as numbers.
"""

from datetime import datetime

import pandera.polars as pa
import polars as pl

MAX_ZONE_ID = 263  # 264/265 are "Unknown" / "Outside of NYC" in the TLC lookup
MAX_DURATION_MIN = 120.0
MAX_DISTANCE_MI = 100.0
MAX_SPEED_MPH = 80.0


def month_bounds(month: str) -> tuple[datetime, datetime]:
    year, mon = map(int, month.split("-"))
    start = datetime(year, mon, 1)
    end = datetime(year + mon // 12, mon % 12 + 1, 1)
    return start, end


def rules(month: str) -> list[tuple[str, pl.Expr]]:
    """(reason, expression that is True when the row is BAD). Order matters: first hit wins."""
    start, end = month_bounds(month)
    speed = pl.col("trip_distance") / (pl.col("duration_min") / 60.0)
    return [
        ("null_required", pl.any_horizontal(
            pl.col(c).is_null() for c in
            ("pickup_at", "dropoff_at", "pu_location_id", "do_location_id", "trip_distance"))),
        ("pickup_outside_month", (pl.col("pickup_at") < start) | (pl.col("pickup_at") >= end)),
        ("duration_out_of_range",
         (pl.col("duration_min") < 1.0) | (pl.col("duration_min") > MAX_DURATION_MIN)),
        ("distance_out_of_range",
         (pl.col("trip_distance") <= 0) | (pl.col("trip_distance") > MAX_DISTANCE_MI)),
        ("unknown_zone", ~pl.col("pu_location_id").is_between(1, MAX_ZONE_ID)
         | ~pl.col("do_location_id").is_between(1, MAX_ZONE_ID)),
        ("implausible_speed", speed > MAX_SPEED_MPH),
    ]


def split_valid_invalid(df: pl.DataFrame, month: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    reason = pl.lit(None, dtype=pl.String)
    for name, bad in reversed(rules(month)):
        reason = pl.when(bad.fill_null(True)).then(pl.lit(name)).otherwise(reason)
    tagged = df.with_columns(reason.alias("reason"))

    valid = tagged.filter(pl.col("reason").is_null()).drop("reason").with_columns(
        pl.col("passenger_count").fill_null(1).clip(0, 9),
        pl.col("ratecode_id").fill_null(1),
    )
    TripContract.validate(valid)
    return valid, tagged.filter(pl.col("reason").is_not_null())


class TripContract(pa.DataFrameModel):
    """Guarantees downstream feature code can rely on."""

    pickup_at: pl.Datetime = pa.Field(nullable=False)
    dropoff_at: pl.Datetime = pa.Field(nullable=False)
    pu_location_id: pl.Int32 = pa.Field(ge=1, le=MAX_ZONE_ID)
    do_location_id: pl.Int32 = pa.Field(ge=1, le=MAX_ZONE_ID)
    trip_distance: float = pa.Field(gt=0, le=MAX_DISTANCE_MI)
    passenger_count: pl.Int32 = pa.Field(ge=0, le=9, nullable=False)
    ratecode_id: pl.Int32 = pa.Field(nullable=False)
    duration_min: float = pa.Field(ge=1.0, le=MAX_DURATION_MIN)

    class Config:
        strict = True
