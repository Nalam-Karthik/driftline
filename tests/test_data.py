import duckdb
import polars as pl

from driftline.data.ingest import ingest_month, load_raw, read_trips
from driftline.data.validate import TripContract, split_valid_invalid
from tests.conftest import make_raw_month


def test_each_bad_row_is_quarantined_with_its_reason(tmp_path):
    path = tmp_path / "raw.parquet"
    make_raw_month("2024-01", n=500, n_bad=30).write_parquet(path)
    valid, bad = split_valid_invalid(load_raw(path), "2024-01")

    assert valid.height == 500 and bad.height == 30
    assert dict(bad.group_by("reason").len().iter_rows()) == {
        "distance_out_of_range": 10, "unknown_zone": 10, "duration_out_of_range": 10}
    TripContract.validate(valid)  # contract holds for everything that gets through


def test_pickups_outside_the_file_month_are_rejected(tmp_path):
    path = tmp_path / "raw.parquet"
    df = make_raw_month("2024-01", n=100, n_bad=0)
    df = df.with_columns(pl.col("tpep_pickup_datetime").dt.offset_by("-2y"),
                         pl.col("tpep_dropoff_datetime").dt.offset_by("-2y"))
    df.write_parquet(path)
    valid, bad = split_valid_invalid(load_raw(path), "2024-01")
    assert valid.is_empty()
    assert set(bad.get_column("reason")) == {"pickup_outside_month"}


def test_reingesting_a_month_is_idempotent(settings):
    settings.raw_dir.mkdir(parents=True)
    make_raw_month("2024-01", n=800).write_parquet(settings.raw_dir / "yellow_tripdata_2024-01.parquet")
    first = ingest_month("2024-01", settings.raw_dir, settings.warehouse_path)
    second = ingest_month("2024-01", settings.raw_dir, settings.warehouse_path)

    assert first.valid_rows == second.valid_rows == 800
    with duckdb.connect(str(settings.warehouse_path), read_only=True) as con:
        assert con.sql("SELECT count(*) FROM trips").fetchall()[0][0] == 800
        assert con.sql("SELECT count(*) FROM quarantine").fetchall()[0][0] == 60
    assert read_trips(settings.warehouse_path, ["2024-01"], sample_rows=100).height == 100
