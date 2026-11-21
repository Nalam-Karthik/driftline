"""Request -> features -> champion/challenger routing -> prediction log."""

import hashlib
import logging
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import polars as pl

from driftline.config import Settings
from driftline.features.build import FEATURES, base_features, to_model_input
from driftline.features.online import STAT_FEATURES, FeatureStore
from driftline.serving import metrics
from driftline.state import StateStore
from driftline.training.registry import ModelBundle, alias_versions, load_bundle

log = logging.getLogger(__name__)


@dataclass
class Trip:
    pickup_at: datetime
    pu_location_id: int
    do_location_id: int
    trip_distance: float
    passenger_count: int = 1
    ratecode_id: int = 1
    request_id: str | None = None


@dataclass
class Prediction:
    request_id: str
    duration_min: float
    model_version: str
    model_role: str
    shadow_duration_min: float | None
    feature_version: str | None


def build_features(trips: list[Trip], store: FeatureStore) -> pl.DataFrame:
    """Same `base_features` used in training, plus zone stats from the online store."""
    df = base_features(pl.DataFrame({
        "pickup_at": [t.pickup_at for t in trips],
        "pu_location_id": pl.Series([t.pu_location_id for t in trips], dtype=pl.Int32),
        "do_location_id": pl.Series([t.do_location_id for t in trips], dtype=pl.Int32),
        "trip_distance": pl.Series([t.trip_distance for t in trips], dtype=pl.Float64),
        "passenger_count": pl.Series([t.passenger_count for t in trips], dtype=pl.Int32),
        "ratecode_id": pl.Series([t.ratecode_id for t in trips], dtype=pl.Int32),
    }))
    hours = df.get_column("hour").to_list()
    looked_up = [store.lookup(t.pu_location_id, t.do_location_id, h) for t, h in zip(trips, hours)]
    for name in STAT_FEATURES:
        values = [row[name] for row in looked_up]
        misses = sum(v is None for v in values)
        if misses:
            metrics.FEATURE_MISSES.labels(name).inc(misses)
        df = df.with_columns(pl.Series(name, values, dtype=pl.Float64))
    return df.select(FEATURES)


def in_canary(request_id: str, fraction: float) -> bool:
    """Deterministic, sticky bucketing: the same request id always lands on the same model."""
    bucket = int.from_bytes(hashlib.blake2b(request_id.encode(), digest_size=8).digest()) % 10_000
    return bucket < fraction * 10_000


class Predictor:
    def __init__(self, settings: Settings, feature_store: FeatureStore, state: StateStore,
                 champion: ModelBundle | None = None, challenger: ModelBundle | None = None):
        self.settings = settings
        self.features = feature_store
        self.state = state
        self._lock = threading.Lock()
        self.champion, self.challenger = champion, challenger
        self._aliases: dict[str, str] = {}
        self._stop = threading.Event()

    # -- model lifecycle -------------------------------------------------------------------
    def reload(self) -> dict[str, str]:
        """Load whatever the registry aliases point at now; swap atomically when ready."""
        aliases = alias_versions(self.settings)
        champion = self.champion
        if aliases.get("champion") != (champion.version if champion else None):
            champion = load_bundle(self.settings, "champion")
        challenger = self.challenger
        want = aliases.get("challenger")
        if want is None or want == aliases.get("champion"):
            challenger = None  # nothing distinct to compare against
        elif want != (challenger.version if challenger else None):
            challenger = load_bundle(self.settings, "challenger")

        with self._lock:
            self.champion, self.challenger, self._aliases = champion, challenger, aliases
        if champion and self.features.version() is None:
            # Seed an empty store. Never overwrite: the monthly refresh flow may have
            # published stats newer than the snapshot stored with the model.
            self.features.publish(champion.stats)
        metrics.set_model_info(champion, challenger)
        log.info("models: champion=%s challenger=%s", champion and champion.version,
                 challenger and challenger.version)
        return {"champion": champion and champion.version,
                "challenger": challenger and challenger.version}

    def start_polling(self) -> None:
        def loop():
            while not self._stop.wait(self.settings.registry_poll_seconds):
                try:
                    if alias_versions(self.settings) != self._aliases:
                        self.reload()
                except Exception:  # keep serving the current models if the registry is down
                    log.exception("registry poll failed")

        threading.Thread(target=loop, name="registry-poller", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    @property
    def ready(self) -> bool:
        return self.champion is not None

    # -- inference -------------------------------------------------------------------------
    def predict(self, trips: list[Trip]) -> list[Prediction]:
        with self._lock:
            champion, challenger = self.champion, self.challenger
        if champion is None:
            raise RuntimeError("no champion model loaded")

        for t in trips:
            t.request_id = t.request_id or uuid.uuid4().hex
        feats = build_features(trips, self.features)
        X = to_model_input(feats)

        t0 = time.perf_counter()
        champ_pred = champion.predict(X)
        chall_pred = challenger.predict(X) if challenger else None
        metrics.MODEL_LATENCY.observe(time.perf_counter() - t0)

        mode = self.settings.routing_mode
        out, records = [], []
        for i, trip in enumerate(trips):
            use_challenger = (challenger is not None and mode == "canary"
                              and in_canary(trip.request_id, self.settings.canary_fraction))
            served = chall_pred[i] if use_challenger else champ_pred[i]
            bundle = challenger if use_challenger else champion
            role = "challenger" if use_challenger else "champion"
            shadow = None
            if chall_pred is not None and mode == "shadow":
                shadow = float(chall_pred[i])
                metrics.SHADOW_DIFF.observe(abs(shadow - champ_pred[i]))

            value = float(max(served, 1.0))
            metrics.PREDICTIONS.labels(role, bundle.version).inc()
            out.append(Prediction(trip.request_id, round(value, 2), bundle.version, role,
                                  shadow, self.features.version()))
            records.append({
                "request_id": trip.request_id, "ts": time.time(), "role": role,
                "version": bundle.version, "prediction": value,
                "champion_version": champion.version,
                "challenger_version": challenger.version if challenger else None,
                "challenger_prediction": float(chall_pred[i]) if chall_pred is not None else None,
                "champion_prediction": float(champ_pred[i]),
                "features": {k: _jsonable(v) for k, v in zip(FEATURES, feats.row(i))},
            })
        try:
            self.state.log_predictions(records)
        except Exception:  # logging must never fail a prediction
            metrics.LOG_FAILURES.inc()
            log.exception("prediction log write failed")
        return out


def _jsonable(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return v
