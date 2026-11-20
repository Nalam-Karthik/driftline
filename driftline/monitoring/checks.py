"""Decide whether the live model needs retraining, from the prediction log + feedback."""

import time
from collections import defaultdict

import numpy as np
import polars as pl

from driftline.config import Settings
from driftline.features.build import FEATURES
from driftline.monitoring.drift import DriftResult, FeatureProfile, score_drift

MIN_RECORDS_FOR_DRIFT = 1_000
MIN_LABELS_FOR_MAE = 500
# Drift on these is expected to move the error; drift on e.g. passenger_count mostly isn't.
KEY_FEATURES = ("trip_distance", "pair_median_min", "pu_hour_median_mph", "hour", "prediction")


def records_frame(records: list[dict]) -> pl.DataFrame:
    rows = [{**r["features"], "prediction": r["champion_prediction"]} for r in records]
    schema = {f: pl.Float64 for f in FEATURES} | {"prediction": pl.Float64}
    return pl.DataFrame(rows, schema=schema)


def live_mae(records: list[dict], feedback: dict[str, float]) -> list[dict]:
    errors: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in records:
        actual = feedback.get(r["request_id"])
        if actual is None:
            continue
        # Both models score every request, so both get an error regardless of which one served.
        errors[("champion", r["champion_version"])].append(abs(r["champion_prediction"] - actual))
        if r.get("challenger_prediction") is not None:
            errors[("challenger", r["challenger_version"])].append(
                abs(r["challenger_prediction"] - actual))
    return [{"role": role, "version": version, "mae": float(np.mean(errs)), "n": len(errs)}
            for (role, version), errs in errors.items()]


def evaluate_window(records: list[dict], feedback: dict[str, float],
                    profile: dict[str, FeatureProfile], champion_version: str,
                    champion_eval_mae: float | None, settings: Settings) -> dict:
    # Only judge the current champion: records from before a promotion describe another model.
    records = [r for r in records if r["champion_version"] == champion_version]
    report: dict = {"checked_at": time.time(), "records": len(records),
                    "labelled": sum(r["request_id"] in feedback for r in records),
                    "champion_version": champion_version, "reasons": [], "features": [],
                    "live_mae": [], "retrain_triggered": False}
    if len(records) < MIN_RECORDS_FOR_DRIFT:
        report["skipped"] = f"insufficient records ({len(records)} < {MIN_RECORDS_FOR_DRIFT})"
        return report

    drift = score_drift(profile, records_frame(records), settings.psi_alert_threshold)
    report["features"] = [vars(d) for d in drift]
    report["live_mae"] = live_mae(records, feedback)
    champ = next((m for m in report["live_mae"]
                  if m["role"] == "champion" and m["version"] == champion_version), None)
    report["reasons"] = retrain_reasons(drift, champ and champ["mae"], champ and champ["n"],
                                        champion_eval_mae, settings)
    report["retrain_triggered"] = bool(report["reasons"])
    return report


def retrain_reasons(drift: list[DriftResult], current_mae: float | None, n_labelled: int | None,
                    offline_mae: float | None, settings: Settings) -> list[str]:
    """The retraining policy, shared by the live worker and the offline backtest."""
    reasons = []
    drifted = [d.feature for d in drift if d.drifted and d.feature in KEY_FEATURES]
    if drifted:
        reasons.append(f"PSI > {settings.psi_alert_threshold} on {', '.join(drifted)}")
    if current_mae is not None and offline_mae and (n_labelled or 0) >= MIN_LABELS_FOR_MAE:
        degradation = current_mae / offline_mae - 1
        if degradation > settings.mae_degradation_threshold:
            reasons.append(f"MAE {current_mae:.2f} is {degradation:+.0%} vs offline {offline_mae:.2f}")
    return reasons
