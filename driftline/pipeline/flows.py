"""Prefect flows: the same DAG runs locally, from the CLI, or from the Celery retrain trigger."""

from dataclasses import asdict

from prefect import flow, get_run_logger, task

from driftline.config import get_settings
from driftline.data.ingest import ingest_month, raw_path, read_trips
from driftline.features.build import ZoneStats, previous_month
from driftline.features.online import make_feature_store
from driftline.training import registry
from driftline.training.train import features_for_month, train


@task(retries=2, retry_delay_seconds=10)
def ingest(month: str) -> dict:
    s = get_settings()
    if not raw_path(s.raw_dir, month).exists():
        raise FileNotFoundError(f"raw file for {month} missing; run `make data`")
    report = ingest_month(month, s.raw_dir, s.warehouse_path)
    if report.quarantine_rate > 0.15:
        # A sudden jump in bad rows usually means an upstream schema/feed problem; stop here
        # rather than train on a silently shrunken dataset.
        raise ValueError(f"{month}: quarantine rate {report.quarantine_rate:.1%} exceeds 15%")
    return asdict(report) | {"quarantine_rate": report.quarantine_rate}


@task
def train_challenger(train_months: list[str], eval_month: str) -> dict:
    result = train(get_settings(), train_months, eval_month)
    return {"version": result.version, "run_id": result.run_id, **result.metrics}


@task
def gate(eval_month: str) -> dict:
    s = get_settings()
    eval_frame = features_for_month(s, eval_month, s.eval_sample_rows)
    return asdict(registry.promote_if_better(s, eval_frame))


@task
def publish_online_features() -> str | None:
    s = get_settings()
    champion = registry.load_bundle(s, "champion")
    if champion is None:
        return None
    make_feature_store(s.redis_url).publish(champion.stats)
    return champion.stats.source_month


@flow(name="refresh-online-features")
def refresh_features_flow(month: str) -> str:
    """Monthly: once `month` is fully ingested, serve its zone stats. No model change needed;
    the model consumes one-month-lagged stats by construction."""
    s = get_settings()
    ingest(month)
    stats = ZoneStats.compute(read_trips(s.warehouse_path, [month]), month)
    make_feature_store(s.redis_url).publish(stats)
    return month


@flow(name="ingest")
def ingest_flow(months: list[str]) -> list[dict]:
    needed = sorted(set(months) | {previous_month(m) for m in months})
    return [ingest(m) for m in needed]


@flow(name="train-and-promote")
def training_flow(train_months: list[str], eval_month: str) -> dict:
    logger = get_run_logger()
    ingest_flow(train_months + [eval_month])
    trained = train_challenger(train_months, eval_month)
    decision = gate(eval_month)
    published = publish_online_features() if decision["promoted"] else None
    logger.info("challenger v%s mae=%.3f promoted=%s", trained["version"],
                trained["eval_mae"], decision["promoted"])
    return {"trained": trained, "decision": decision, "published_features": published}
