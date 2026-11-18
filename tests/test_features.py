from datetime import datetime

import numpy as np
import polars as pl
import pytest

from driftline.data.ingest import read_trips
from driftline.features.build import FEATURES, ZoneStats, build_training_frame, previous_month, to_model_input
from driftline.features.online import InMemoryFeatureStore, RedisFeatureStore
from driftline.serving.predictor import Trip, build_features


def test_previous_month_wraps_year():
    assert previous_month("2024-01") == "2023-12"
    assert previous_month("2024-07") == "2024-06"


def test_training_and_serving_features_are_identical(warehouse):
    """The single most common production ML bug is train/serve skew. Build the same rows both
    ways and require exact equality."""
    stats = ZoneStats.compute(read_trips(warehouse.warehouse_path, ["2024-03"]), "2024-03")
    trips = read_trips(warehouse.warehouse_path, ["2024-04"], sample_rows=200)
    offline = build_training_frame(trips, stats).select(FEATURES)

    store = InMemoryFeatureStore()
    store.publish(stats)
    online = build_features([
        Trip(r["pickup_at"], r["pu_location_id"], r["do_location_id"], r["trip_distance"],
             r["passenger_count"], r["ratecode_id"]) for r in trips.iter_rows(named=True)
    ], store)

    np.testing.assert_array_equal(to_model_input(offline), to_model_input(online))


def test_zone_stats_only_come_from_the_given_window(warehouse):
    march = read_trips(warehouse.warehouse_path, ["2024-03"])
    stats = ZoneStats.compute(march, "2024-03")
    pair = stats.pair.row(0, named=True)
    expected = (march.filter((pl.col("pu_location_id") == pair["pu_location_id"])
                             & (pl.col("do_location_id") == pair["do_location_id"]))
                .get_column("duration_min").median())
    assert pair["pair_median_min"] == pytest.approx(expected)


def test_unknown_pair_falls_back_to_missing_not_error(warehouse):
    stats = ZoneStats.compute(read_trips(warehouse.warehouse_path, ["2024-03"]), "2024-03")
    store = InMemoryFeatureStore()
    store.publish(stats)
    row = build_features([Trip(datetime(2024, 4, 1, 3), 200, 201, 1.0)], store).row(0, named=True)
    assert row["pair_median_min"] is None and row["pu_hour_median_mph"] is None
    assert np.isnan(to_model_input(pl.DataFrame([row])).astype(float)).any()


class FakeRedis:
    """Just enough of redis-py for the feature store (avoids a server in unit tests)."""

    def __init__(self):
        self.h, self.kv = {}, {}

    def pipeline(self, transaction=False):
        return FakePipe(self)

    def hset(self, key, mapping):
        self.h.setdefault(key, {}).update(mapping)

    def hget(self, key, field):
        return self.h.get(key, {}).get(field)

    def delete(self, *keys):
        for k in keys:
            self.h.pop(k, None)

    def expire(self, key, ttl):
        pass

    def set(self, key, value):
        self.kv[key] = value

    def get(self, key):
        return self.kv.get(key)


class FakePipe:
    def __init__(self, r):
        self.r, self.ops = r, []

    def __getattr__(self, name):
        return lambda *a, **kw: self.ops.append((name, a, kw))

    def execute(self):
        return [getattr(self.r, n)(*a, **kw) for n, a, kw in self.ops]


def test_redis_store_matches_in_memory_and_flips_version(warehouse):
    feb = ZoneStats.compute(read_trips(warehouse.warehouse_path, ["2024-02"]), "2024-02")
    mar = ZoneStats.compute(read_trips(warehouse.warehouse_path, ["2024-03"]), "2024-03")
    redis_store, mem = RedisFeatureStore(FakeRedis()), InMemoryFeatureStore()
    redis_store.VERSION_TTL_S = 0  # no pointer caching in the test

    for stats in (feb, mar):
        redis_store.publish(stats)
        mem.publish(stats)
        assert redis_store.version() == mem.version() == stats.source_month
        for pu, do, hour in [(3, 7, 8), (1, 1, 0), (40, 2, 23)]:
            assert redis_store.lookup(pu, do, hour) == mem.lookup(pu, do, hour)
