"""State shared between the API and the monitoring worker: prediction log, ground-truth
feedback, and the latest monitoring report. Redis in deployment, in-process for tests."""

import json
import time
from collections import deque
from typing import Protocol

PREDICTION_STREAM = "predictions"
STREAM_MAXLEN = 200_000
FEEDBACK_TTL_S = 7 * 24 * 3600


class StateStore(Protocol):
    def log_predictions(self, records: list[dict]) -> None: ...
    def recent_predictions(self, n: int) -> list[dict]: ...
    def record_feedback(self, request_id: str, actual_min: float) -> None: ...
    def feedback_for(self, request_ids: list[str]) -> dict[str, float]: ...
    def put_report(self, name: str, report: dict) -> None: ...
    def get_report(self, name: str) -> dict | None: ...
    def acquire_lock(self, name: str, ttl_s: int) -> bool: ...


class InMemoryStateStore:
    def __init__(self) -> None:
        self._log: deque[dict] = deque(maxlen=STREAM_MAXLEN)
        self._feedback: dict[str, float] = {}
        self._reports: dict[str, dict] = {}
        self._locks: dict[str, float] = {}

    def acquire_lock(self, name: str, ttl_s: int) -> bool:
        now = time.monotonic()
        if self._locks.get(name, 0.0) > now:
            return False
        self._locks[name] = now + ttl_s
        return True

    def log_predictions(self, records: list[dict]) -> None:
        self._log.extend(records)

    def recent_predictions(self, n: int) -> list[dict]:
        return list(self._log)[-n:]

    def record_feedback(self, request_id: str, actual_min: float) -> None:
        self._feedback[request_id] = actual_min

    def feedback_for(self, request_ids: list[str]) -> dict[str, float]:
        return {i: self._feedback[i] for i in request_ids if i in self._feedback}

    def put_report(self, name: str, report: dict) -> None:
        self._reports[name] = report

    def get_report(self, name: str) -> dict | None:
        return self._reports.get(name)


class RedisStateStore:
    def __init__(self, client) -> None:
        self.r = client

    def acquire_lock(self, name: str, ttl_s: int) -> bool:
        return bool(self.r.set(name, "1", nx=True, ex=ttl_s))  # SET NX EX is atomic

    def log_predictions(self, records: list[dict]) -> None:
        pipe = self.r.pipeline(transaction=False)
        for rec in records:
            pipe.xadd(PREDICTION_STREAM, {"v": json.dumps(rec)}, maxlen=STREAM_MAXLEN,
                      approximate=True)
        pipe.execute()

    def recent_predictions(self, n: int) -> list[dict]:
        entries = self.r.xrevrange(PREDICTION_STREAM, count=n)
        return [json.loads(fields[b"v"]) for _, fields in reversed(entries)]

    def record_feedback(self, request_id: str, actual_min: float) -> None:
        self.r.set(f"feedback:{request_id}", actual_min, ex=FEEDBACK_TTL_S)

    def feedback_for(self, request_ids: list[str]) -> dict[str, float]:
        if not request_ids:
            return {}
        values = self.r.mget([f"feedback:{i}" for i in request_ids])
        return {i: float(v) for i, v in zip(request_ids, values) if v is not None}

    def put_report(self, name: str, report: dict) -> None:
        self.r.set(f"report:{name}", json.dumps(report))

    def get_report(self, name: str) -> dict | None:
        raw = self.r.get(f"report:{name}")
        return json.loads(raw) if raw else None


def make_state_store(redis_url: str | None) -> StateStore:
    if redis_url:
        import redis

        return RedisStateStore(redis.Redis.from_url(redis_url))
    return InMemoryStateStore()
