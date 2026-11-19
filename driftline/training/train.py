"""Train a LightGBM trip-duration model, log everything to MLflow, register it as `challenger`."""

import json
import logging
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import polars as pl
from mlflow.models import infer_signature

from driftline.config import Settings
from driftline.data.ingest import read_trips
from driftline.features.build import (
    CATEGORICAL,
    CATEGORICAL_INDEX,
    FEATURES,
    TARGET,
    ZoneStats,
    build_training_frame,
    previous_month,
    to_model_input,
)
from driftline.monitoring.drift import build_profile, profile_to_json

log = logging.getLogger(__name__)

PARAMS = {
    "objective": "huber",
    "alpha": 6.0,
    "learning_rate": 0.08,
    "num_leaves": 255,
    "min_child_samples": 100,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "cat_smooth": 20,
    "max_cat_to_onehot": 4,
    "n_estimators": 1500,
    "verbose": -1,
}


@dataclass
class TrainResult:
    run_id: str
    version: str
    metrics: dict[str, float]


def regression_metrics(y: np.ndarray, pred: np.ndarray, prefix: str = "") -> dict[str, float]:
    err = np.abs(y - pred)
    return {
        f"{prefix}mae": float(err.mean()),
        f"{prefix}rmse": float(np.sqrt(((y - pred) ** 2).mean())),
        f"{prefix}p90_abs_err": float(np.quantile(err, 0.9)),
        f"{prefix}within_3min": float((err <= 3.0).mean()),
    }


def features_for_month(settings: Settings, month: str, sample_rows: int | None) -> pl.DataFrame:
    """Rows of `month` with zone stats taken strictly from the month before (point-in-time)."""
    prev = previous_month(month)
    stats = ZoneStats.compute(read_trips(settings.warehouse_path, [prev]), prev)
    trips = read_trips(settings.warehouse_path, [month], sample_rows, settings.seed)
    return build_training_frame(trips, stats)


def train(settings: Settings, train_months: list[str], eval_month: str) -> TrainResult:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)

    t0 = time.perf_counter()
    train_df = pl.concat([features_for_month(settings, m, settings.train_sample_rows)
                          for m in train_months])
    eval_df = features_for_month(settings, eval_month, settings.eval_sample_rows)
    # The model learned from stats that lag the trip by one month, so for traffic after
    # `eval_month` the matching (and freshest labelled) stats are those of `eval_month`.
    serving_stats = ZoneStats.compute(read_trips(settings.warehouse_path, [eval_month]), eval_month)
    feature_seconds = time.perf_counter() - t0

    # Early-stopping holdout comes from the training window only; eval month stays untouched.
    shuffled = train_df.sample(fraction=1.0, shuffle=True, seed=settings.seed)
    n_val = int(0.1 * shuffled.height)
    fit_df, val_df = shuffled.slice(n_val), shuffled.head(n_val)

    X_fit, y_fit = to_model_input(fit_df), fit_df.get_column(TARGET).to_numpy()
    X_val, y_val = to_model_input(val_df), val_df.get_column(TARGET).to_numpy()
    X_eval, y_eval = to_model_input(eval_df), eval_df.get_column(TARGET).to_numpy()

    with mlflow.start_run(run_name=f"train-{train_months[0]}..{train_months[-1]}") as run:
        mlflow.log_params({**PARAMS, "train_months": ",".join(train_months),
                           "eval_month": eval_month, "train_rows": fit_df.height,
                           "features": ",".join(FEATURES)})

        t1 = time.perf_counter()
        model = lgb.LGBMRegressor(**PARAMS, random_state=settings.seed)
        model.fit(X_fit, y_fit, eval_X=X_val, eval_y=y_val, eval_metric="l1",
                  categorical_feature=CATEGORICAL_INDEX, feature_name=FEATURES,
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        fit_seconds = time.perf_counter() - t1

        pred = model.predict(X_eval)
        metrics = regression_metrics(y_eval, pred, "eval_")
        # Naive baseline: "same trip took this long last month" (fallback: global median).
        baseline = eval_df.get_column("pair_median_min").fill_null(float(np.median(y_fit))).to_numpy()
        metrics |= regression_metrics(y_eval, baseline, "baseline_")
        metrics |= {"best_iteration": float(model.best_iteration_),
                    "feature_seconds": feature_seconds, "fit_seconds": fit_seconds}
        mlflow.log_metrics(metrics)

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            serving_stats.save(tmp / "zone_stats")
            mlflow.log_artifacts(str(tmp / "zone_stats"), "zone_stats")

            # Reference distribution for drift monitoring: eval-like inputs + predictions.
            ref = eval_df.select(FEATURES).with_columns(pl.Series("prediction", pred))
            profile = build_profile(ref, CATEGORICAL, settings.seed)
            (tmp / "profile.json").write_text(json.dumps(profile_to_json(profile)))
            mlflow.log_artifact(str(tmp / "profile.json"))

            importance = dict(zip(FEATURES, model.booster_.feature_importance("gain").tolist()))
            (tmp / "feature_importance.json").write_text(json.dumps(importance, indent=2))
            mlflow.log_artifact(str(tmp / "feature_importance.json"))

        mlflow.lightgbm.log_model(
            model, name="model",
            signature=infer_signature(X_eval[:100], pred[:100]),
            registered_model_name=settings.model_name,
        )

    client = mlflow.MlflowClient()
    version = str(max(int(mv.version) for mv in
                      client.search_model_versions(f"run_id='{run.info.run_id}'")))
    client.set_registered_model_alias(settings.model_name, "challenger", version)
    client.set_model_version_tag(settings.model_name, version, "eval_month", eval_month)
    client.set_model_version_tag(settings.model_name, version, "eval_mae", f"{metrics['eval_mae']:.4f}")
    log.info("trained v%s eval_mae=%.3f baseline_mae=%.3f", version,
             metrics["eval_mae"], metrics["baseline_mae"])
    return TrainResult(run.info.run_id, version, metrics)
