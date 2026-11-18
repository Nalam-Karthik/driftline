"""Online feature store: zone statistics served at request time.

Each publish writes a complete, versioned copy (`fs:{month}:*`) and then flips a single
`fs:current` pointer, so readers never observe a half-written feature set.
"""

import time
from typing import Protocol

import polars as pl

from driftline.features.build import ZoneStats

STAT_FEATURES = ("pair_median_min", "pair_trips", "pu_hour_median_mph", "pu_hour_trips")


class FeatureStore(Protocol):
    def publish(self, stats: ZoneStats) -> None: ...
    def lookup(self, pu: int, do: int, hour: int) -> dict[str, float | None]: ...
    def version(self) -> str | None: ...


def _pair_rows(stats: ZoneStats):
    for pu, do, med, n in stats.pair.select(
        "pu_location_id", "do_location_id", "pair_median_min", "pair_trips"
    ).iter_rows():
        yield f"{pu}:{do}", f"{med}|{n}"


def _pu_hour_rows(stats: ZoneStats):
    for pu, hour, mph, n in stats.pu_hour.select(
        "pu_location_id", "hour", "pu_hour_median_mph", "pu_hour_trips"
    ).iter_rows():
        yield f"{pu}:{hour}", f"{mph}|{n}"


def _decode(raw: str | bytes | None) -> tuple[float | None, float | None]:
    if raw is None:
        return None, None
    if isinstance(raw, bytes):
        raw = raw.decode()
    a, b = raw.split("|")
    return float(a), float(b)


def _result(pair_raw, pu_hour_raw) -> dict[str, float | None]:
    pair_med, pair_n = _decode(pair_raw)
    mph, pu_n = _decode(pu_hour_raw)
    return {"pair_median_min": pair_med, "pair_trips": pair_n,
            "pu_hour_median_mph": mph, "pu_hour_trips": pu_n}


class InMemoryFeatureStore:
    def __init__(self) -> None:
        self._version: str | None = None
        self._pair: dict[str, str] = {}
        self._pu_hour: dict[str, str] = {}

    def publish(self, stats: ZoneStats) -> None:
        pair, pu_hour = dict(_pair_rows(stats)), dict(_pu_hour_rows(stats))
        self._pair, self._pu_hour, self._version = pair, pu_hour, stats.source_month

    def lookup(self, pu: int, do: int, hour: int) -> dict[str, float | None]:
        return _result(self._pair.get(f"{pu}:{do}"), self._pu_hour.get(f"{pu}:{hour}"))

    def version(self) -> str | None:
        return self._version


class RedisFeatureStore:
    VERSION_TTL_S = 2.0  # cache the pointer so a lookup is one round trip, not two

    def __init__(self, client) -> None:
        self.r = client
        self._cached: tuple[float, str | None] = (0.0, None)

    def _current(self) -> str | None:
        now = time.monotonic()
        if now - self._cached[0] > self.VERSION_TTL_S:
            self._cached = (now, self.version())
        return self._cached[1]

    def publish(self, stats: ZoneStats) -> None:
        v = stats.source_month
        pipe = self.r.pipeline(transaction=False)
        pipe.delete(f"fs:{v}:pair", f"fs:{v}:pu_hour")
        pipe.hset(f"fs:{v}:pair", mapping=dict(_pair_rows(stats)))
        pipe.hset(f"fs:{v}:pu_hour", mapping=dict(_pu_hour_rows(stats)))
        pipe.execute()
        previous = self.version()
        self.r.set("fs:current", v)  # atomic pointer flip
        if previous and previous != v:
            # Expire rather than delete: readers may still hold the old pointer for a moment.
            self.r.expire(f"fs:{previous}:pair", 60)
            self.r.expire(f"fs:{previous}:pu_hour", 60)

    def lookup(self, pu: int, do: int, hour: int) -> dict[str, float | None]:
        v = self._current()
        if v is None:
            return _result(None, None)
        pipe = self.r.pipeline(transaction=False)
        pipe.hget(f"fs:{v}:pair", f"{pu}:{do}")
        pipe.hget(f"fs:{v}:pu_hour", f"{pu}:{hour}")
        pair_raw, pu_hour_raw = pipe.execute()
        return _result(pair_raw, pu_hour_raw)

    def version(self) -> str | None:
        v = self.r.get("fs:current")
        return v.decode() if isinstance(v, bytes) else v


def make_feature_store(redis_url: str | None) -> FeatureStore:
    if redis_url:
        import redis

        return RedisFeatureStore(redis.Redis.from_url(redis_url))
    return InMemoryFeatureStore()


def stats_frame(values: dict[str, float | None]) -> pl.DataFrame:
    return pl.DataFrame({k: [values[k]] for k in STAT_FEATURES},
                        schema={k: pl.Float64 for k in STAT_FEATURES})
