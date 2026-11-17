"""Raw TLC parquet -> validated DuckDB tables (`trips` + `quarantine`)."""

import logging
from dataclasses import dataclass
from pathlib import Path

import duckdb
import polars as pl

from driftline.data.validate import split_valid_invalid

log = logging.getLogger(__name__)

RAW_COLUMNS = {
    "tpep_pickup_datetime": "pickup_at",
    "tpep_dropoff_datetime": "dropoff_at",
    "PULocationID": "pu_location_id",
    "DOLocationID": "do_location_id",
    "trip_distance": "trip_distance",
    "passenger_count": "passenger_count",
    "RatecodeID": "ratecode_id",
}


@dataclass
class IngestReport:
    month: str
    raw_rows: int
    valid_rows: int
    quarantined_rows: int
    reasons: dict[str, int]

    @property
    def quarantine_rate(self) -> float:
        return self.quarantined_rows / max(self.raw_rows, 1)


def raw_path(raw_dir: Path, month: str) -> Path:
    return raw_dir / f"yellow_tripdata_{month}.parquet"


def load_raw(path: Path) -> pl.DataFrame:
    df = pl.read_parquet(path, columns=list(RAW_COLUMNS))
    return df.rename(RAW_COLUMNS).with_columns(
        pl.col("pu_location_id").cast(pl.Int32),
        pl.col("do_location_id").cast(pl.Int32),
        pl.col("passenger_count").cast(pl.Int32),
        pl.col("ratecode_id").cast(pl.Int32),
        ((pl.col("dropoff_at") - pl.col("pickup_at")).dt.total_seconds() / 60.0).alias("duration_min"),
    )


def ingest_month(month: str, raw_dir: Path, warehouse: Path) -> IngestReport:
    """Validate one month and upsert it into the warehouse. Re-running a month is idempotent."""
    df = load_raw(raw_path(raw_dir, month))
    valid, quarantined = split_valid_invalid(df, month)

    warehouse.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(warehouse)) as con:
        _ensure_tables(con, valid, quarantined)
        con.execute("BEGIN")
        con.execute("DELETE FROM trips WHERE month = ?", [month])
        con.execute("DELETE FROM quarantine WHERE month = ?", [month])
        con.register("valid_df", valid.with_columns(pl.lit(month).alias("month")))
        con.register("bad_df", quarantined.with_columns(pl.lit(month).alias("month")))
        con.execute("INSERT INTO trips SELECT * FROM valid_df")
        con.execute("INSERT INTO quarantine SELECT * FROM bad_df")
        con.execute("COMMIT")

    reasons = (
        dict(quarantined.group_by("reason").len().iter_rows()) if quarantined.height else {}
    )
    report = IngestReport(month, df.height, valid.height, quarantined.height, reasons)
    log.info("ingested %s: %d valid, %d quarantined (%.1f%%)", month, report.valid_rows,
             report.quarantined_rows, 100 * report.quarantine_rate)
    return report


def _ensure_tables(con: duckdb.DuckDBPyConnection, valid: pl.DataFrame, bad: pl.DataFrame) -> None:
    con.register("v0", valid.head(0).with_columns(pl.lit("").alias("month")))
    con.register("q0", bad.head(0).with_columns(pl.lit("").alias("month")))
    con.execute("CREATE TABLE IF NOT EXISTS trips AS SELECT * FROM v0")
    con.execute("CREATE TABLE IF NOT EXISTS quarantine AS SELECT * FROM q0")


def read_trips(warehouse: Path, months: list[str], sample_rows: int | None = None,
               seed: int = 7) -> pl.DataFrame:
    with duckdb.connect(str(warehouse), read_only=True) as con:
        frames = []
        for month in months:
            q = "SELECT * EXCLUDE (month) FROM trips WHERE month = $month"
            df = con.sql(q, params={"month": month}).pl()
            if df.is_empty():
                raise ValueError(f"month {month} not ingested")
            if sample_rows and df.height > sample_rows:
                df = df.sample(sample_rows, seed=seed)
            frames.append(df)
    return pl.concat(frames)
