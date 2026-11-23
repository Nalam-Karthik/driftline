"""Walk-forward backtest of the retraining policy on real 2024 NYC taxi data.

At the start of each month M we only know labels up to M-1. Three strategies are compared on
every month's actual trips:

  frozen          model trained Jan-Feb, zone stats frozen at deploy time (no MLOps loop)
  fresh-features  same frozen model, zone stats refreshed monthly (feature pipeline only)
  driftline       monthly feature refresh + drift/MAE-triggered retraining behind the
                  promotion gate (the full loop)

Uses its own MLflow database so it never touches the serving registry.

    python scripts/backtest.py --first-eval 2024-03 --last 2024-12
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from driftline.config import Settings
from driftline.data.ingest import ingest_month, read_trips
from driftline.features.build import TARGET, ZoneStats, build_training_frame, previous_month, to_model_input
from driftline.monitoring.checks import retrain_reasons
from driftline.monitoring.drift import score_drift
from driftline.training import registry
from driftline.training.train import features_for_month, train

log = logging.getLogger("backtest")


def months_between(first: str, last: str) -> list[str]:
    out, m = [], first
    while m <= last:
        out.append(m)
        y, mo = map(int, m.split("-"))
        m = f"{y + mo // 12:04d}-{mo % 12 + 1:02d}"
    return out


def score(bundle, trips, stats: ZoneStats) -> tuple[float, np.ndarray, object]:
    frame = build_training_frame(trips, stats)
    pred = bundle.predict(to_model_input(frame))
    return float(np.abs(pred - frame.get_column(TARGET).to_numpy()).mean()), pred, frame


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first-eval", default="2024-03")
    ap.add_argument("--last", default="2024-12")
    ap.add_argument("--train-window", type=int, default=2, help="months of history per retrain")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    for noisy in ("mlflow", "alembic", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    s = Settings(mlflow_tracking_uri=f"sqlite:///data/backtest_w{args.train_window}_mlflow.db",
                 model_name="trip-duration-backtest", mlflow_experiment="backtest")

    first_train = previous_month(previous_month(args.first_eval))
    all_months = months_between(previous_month(first_train), args.last)
    for m in all_months:
        ingest_month(m, s.raw_dir, s.warehouse_path)

    # Initial deployment: train on the two months before first-eval, gate on first-eval.
    t0 = time.perf_counter()
    train(s, [first_train, previous_month(args.first_eval)], args.first_eval)
    registry.promote_if_better(s, features_for_month(s, args.first_eval, s.eval_sample_rows))
    v1 = registry.load_bundle(s, "champion")
    champion = v1

    rows = []
    for month in months_between(args.first_eval, args.last)[1:]:
        last_labelled = previous_month(month)
        event = None

        # --- monitoring on the newest labelled month, featurised exactly as it was served
        served_stats_month = previous_month(last_labelled)
        served_stats = ZoneStats.compute(read_trips(s.warehouse_path, [served_stats_month]),
                                         served_stats_month)
        trips_prev = read_trips(s.warehouse_path, [last_labelled], s.eval_sample_rows, s.seed)
        prev_mae, prev_pred, prev_frame = score(champion, trips_prev, served_stats)
        drift = score_drift(champion.profile, prev_frame.with_columns(
            prediction=prev_pred).drop(TARGET), s.psi_alert_threshold)
        reasons = retrain_reasons(drift, prev_mae, prev_frame.height, champion.eval_mae, s)

        if reasons:
            train_months = [previous_month(last_labelled)]
            while len(train_months) < args.train_window and train_months[0] > first_train:
                train_months.insert(0, previous_month(train_months[0]))
            trained = train(s, train_months, last_labelled)
            decision = registry.promote_if_better(
                s, features_for_month(s, last_labelled, s.eval_sample_rows))
            if decision.promoted:
                champion = registry.load_bundle(s, "champion")
            event = {"reasons": reasons, "trained_version": trained.version,
                     "promoted": decision.promoted, "gate": decision.reason}

        # --- what each strategy actually scores on this month's trips
        trips = read_trips(s.warehouse_path, [month], s.eval_sample_rows, s.seed)
        fresh_stats = ZoneStats.compute(read_trips(s.warehouse_path, [last_labelled]), last_labelled)
        row = {
            "month": month,
            "frozen_mae": score(v1, trips, v1.stats)[0],
            "fresh_features_mae": score(v1, trips, fresh_stats)[0],
            "driftline_mae": score(champion, trips, fresh_stats)[0],
            "champion_version": champion.version,
            "max_psi": max(d.psi for d in drift),
            "top_drift": [(d.feature, d.psi) for d in drift[:3]],
            "event": event,
        }
        rows.append(row)
        log.info("%s frozen=%.3f fresh=%.3f driftline=%.3f (v%s) %s", month, row["frozen_mae"],
                 row["fresh_features_mae"], row["driftline_mae"], champion.version,
                 f"RETRAIN: {reasons}" if reasons else "")

    summary = {
        "months": len(rows),
        "mean_mae": {k: float(np.mean([r[f"{k}_mae"] for r in rows]))
                     for k in ("frozen", "fresh_features", "driftline")},
        "retrains": sum(r["event"] is not None for r in rows),
        "promotions": sum(bool(r["event"] and r["event"]["promoted"]) for r in rows),
        "wall_seconds": time.perf_counter() - t0,
    }
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    summary["train_window"] = args.train_window
    (out / f"backtest_w{args.train_window}.json").write_text(
        json.dumps({"summary": summary, "months": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
