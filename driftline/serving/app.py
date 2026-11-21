import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, Response
from fastapi.concurrency import run_in_threadpool
from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, generate_latest
from pydantic import BaseModel, Field

from driftline.config import get_settings
from driftline.features.online import make_feature_store
from driftline.serving import metrics
from driftline.serving.predictor import Predictor, Trip
from driftline.state import make_state_store
from driftline.training import registry

log = logging.getLogger(__name__)


class TripIn(BaseModel):
    pickup_at: datetime
    pu_location_id: int = Field(ge=1, le=263)
    do_location_id: int = Field(ge=1, le=263)
    trip_distance: float = Field(gt=0, le=100)
    passenger_count: int = Field(1, ge=0, le=9)
    ratecode_id: int = 1
    request_id: str | None = Field(None, max_length=64)


class PredictionOut(BaseModel):
    request_id: str
    duration_min: float
    model_version: str
    model_role: str
    shadow_duration_min: float | None = None
    feature_version: str | None = None


class BatchIn(BaseModel):
    trips: list[TripIn] = Field(min_length=1, max_length=1000)


class FeedbackIn(BaseModel):
    request_id: str
    actual_duration_min: float = Field(gt=0, le=600)


def create_app(predictor: Predictor | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal predictor
        if predictor is None:
            s = get_settings()
            predictor = Predictor(s, make_feature_store(s.redis_url), make_state_store(s.redis_url))
            try:
                await run_in_threadpool(predictor.reload)
            except Exception:  # start anyway; /ready stays false until a model loads
                log.exception("initial model load failed")
            predictor.start_polling()
        app.state.predictor = predictor
        collector = metrics.DriftCollector(predictor.state)
        REGISTRY.register(collector)
        yield
        REGISTRY.unregister(collector)
        predictor.stop()

    app = FastAPI(title="driftline", version="1.0", lifespan=lifespan)

    def _predict(trips: list[TripIn], endpoint: str) -> list[PredictionOut]:
        p: Predictor = app.state.predictor
        if not p.ready:
            raise HTTPException(503, "no champion model loaded")
        t0 = time.perf_counter()
        out = p.predict([Trip(**t.model_dump()) for t in trips])
        metrics.REQUEST_LATENCY.labels(endpoint).observe(time.perf_counter() - t0)
        return [PredictionOut(**vars(o)) for o in out]

    # Plain `def` handlers: inference is CPU-bound, so FastAPI runs them in its threadpool
    # instead of blocking the event loop.
    @app.post("/predict", response_model=PredictionOut)
    def predict(trip: TripIn):
        return _predict([trip], "predict")[0]

    @app.post("/predict/batch", response_model=list[PredictionOut])
    def predict_batch(batch: BatchIn):
        return _predict(batch.trips, "predict_batch")

    @app.post("/feedback", status_code=202)
    def feedback(fb: FeedbackIn):
        app.state.predictor.state.record_feedback(fb.request_id, fb.actual_duration_min)
        return {"accepted": True}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        if not app.state.predictor.ready:
            raise HTTPException(503, "not ready")
        return {"status": "ready"}

    @app.get("/models")
    def models():
        p: Predictor = app.state.predictor
        return {
            "champion": p.champion and {"version": p.champion.version, "eval_mae": p.champion.eval_mae},
            "challenger": p.challenger and {"version": p.challenger.version,
                                            "eval_mae": p.challenger.eval_mae},
            "routing_mode": p.settings.routing_mode,
            "canary_fraction": p.settings.canary_fraction,
            "feature_version": p.features.version(),
        }

    @app.post("/admin/reload")
    def reload():
        return app.state.predictor.reload()

    @app.post("/admin/rollback")
    def rollback():
        p: Predictor = app.state.predictor
        try:
            version = registry.rollback(p.settings)
        except RuntimeError as e:
            raise HTTPException(409, str(e)) from e
        p.reload()
        return {"champion": version}

    @app.get("/monitoring/drift")
    def drift():
        report = app.state.predictor.state.get_report("drift")
        if report is None:
            raise HTTPException(404, "no drift report yet")
        return report

    @app.get("/metrics")
    def prometheus():
        return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
