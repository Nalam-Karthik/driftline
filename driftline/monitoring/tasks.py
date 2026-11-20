"""Celery worker + beat schedule: periodic drift checks that can trigger the retraining flow.

    celery -A driftline.monitoring.tasks worker --beat --loglevel=info
"""

import logging
import os

import duckdb
from celery import Celery

from driftline.config import get_settings
from driftline.monitoring.checks import evaluate_window
from driftline.state import make_state_store
from driftline.training.registry import load_bundle

log = logging.getLogger(__name__)

broker = os.environ.get("DRIFTLINE_REDIS_URL", "redis://localhost:6379/0")
app = Celery("driftline", broker=broker, backend=broker)
app.conf.update(
    task_acks_late=True,               # a crashed worker's task is redelivered, not lost
    worker_prefetch_multiplier=1,      # retraining is long; don't hoard tasks
    task_time_limit=3600,
    beat_schedule={
        "drift-check": {"task": "driftline.check_drift",
                        "schedule": float(os.environ.get("DRIFTLINE_DRIFT_CHECK_SECONDS", 300))},
    },
)

RETRAIN_LOCK = "lock:retrain"
RETRAIN_COOLDOWN_S = 6 * 3600


@app.task(name="driftline.check_drift")
def check_drift() -> dict:
    s = get_settings()
    state = make_state_store(s.redis_url)
    champion = load_bundle(s, "champion")
    if champion is None:
        return {"skipped": "no champion"}

    records = state.recent_predictions(s.drift_window)
    feedback = state.feedback_for([r["request_id"] for r in records])
    report = evaluate_window(records, feedback, champion.profile, champion.version,
                             champion.eval_mae, s)

    if report["retrain_triggered"]:
        # At most one retrain per cooldown window, across any number of workers.
        acquired = state.acquire_lock(RETRAIN_LOCK, RETRAIN_COOLDOWN_S)
        report["retrain_enqueued"] = acquired
        if acquired:
            retrain.delay(reasons=report["reasons"])
    state.put_report("drift", report)
    log.info("drift check: %d records, reasons=%s", report["records"], report["reasons"])
    return {k: report[k] for k in ("records", "labelled", "reasons", "retrain_triggered")}


def latest_windows(warehouse) -> tuple[list[str], str]:
    """Train on the two newest complete months before the newest one; evaluate on the newest."""
    with duckdb.connect(str(warehouse), read_only=True) as con:
        months = [m for (m,) in con.sql("SELECT DISTINCT month FROM trips ORDER BY month").fetchall()]
    if len(months) < 4:  # 2 train + 1 eval + 1 for the first month's point-in-time stats
        raise RuntimeError(f"need at least 4 ingested months, have {months}")
    return months[-3:-1], months[-1]


@app.task(name="driftline.retrain", bind=True, max_retries=2, default_retry_delay=600)
def retrain(self, reasons: list[str] | None = None) -> dict:
    from driftline.pipeline.flows import training_flow  # heavy import, only in the worker

    s = get_settings()
    train_months, eval_month = latest_windows(s.warehouse_path)
    log.info("retraining on %s, eval %s (reasons: %s)", train_months, eval_month, reasons)
    try:
        result = training_flow(train_months, eval_month)
    except Exception as exc:
        raise self.retry(exc=exc) from exc
    return {"version": result["trained"]["version"], "promoted": result["decision"]["promoted"]}
