"""Model registry access and the champion/challenger promotion gate."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import polars as pl

from driftline.config import Settings
from driftline.features.build import TARGET, ZoneStats, to_model_input
from driftline.monitoring.drift import FeatureProfile, profile_from_json

log = logging.getLogger(__name__)


@dataclass
class ModelBundle:
    """Everything needed to serve and monitor one registered model version."""

    version: str
    run_id: str
    model: object  # lightgbm.LGBMRegressor
    stats: ZoneStats
    profile: dict[str, FeatureProfile]
    eval_mae: float | None

    def predict(self, X: np.ndarray) -> np.ndarray:
        # For a handful of rows, spinning up OpenMP threads costs more than the trees do.
        threads = 1 if X.shape[0] < 256 else 0
        return self.model.booster_.predict(X, num_threads=threads)


def load_bundle(settings: Settings, alias: str) -> ModelBundle | None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.MlflowClient()
    try:
        mv = client.get_model_version_by_alias(settings.model_name, alias)
    except mlflow.exceptions.MlflowException:
        return None
    model = mlflow.lightgbm.load_model(f"models:/{settings.model_name}/{mv.version}")
    stats_dir = Path(mlflow.artifacts.download_artifacts(run_id=mv.run_id, artifact_path="zone_stats"))
    profile_path = mlflow.artifacts.download_artifacts(run_id=mv.run_id, artifact_path="profile.json")
    profile = profile_from_json(json.loads(Path(profile_path).read_text()))
    eval_mae = mv.tags.get("eval_mae")
    return ModelBundle(str(mv.version), mv.run_id, model, ZoneStats.load(stats_dir), profile,
                       float(eval_mae) if eval_mae else None)


def alias_versions(settings: Settings) -> dict[str, str]:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.MlflowClient()
    try:
        aliases = client.get_registered_model(settings.model_name).aliases
        return {alias: str(version) for alias, version in aliases.items()}
    except mlflow.exceptions.MlflowException:
        return {}


@dataclass
class PromotionDecision:
    promoted: bool
    challenger_version: str
    champion_version: str | None
    challenger_mae: float
    champion_mae: float | None
    reason: str


def promote_if_better(settings: Settings, eval_frame: pl.DataFrame) -> PromotionDecision:
    """Score champion and challenger on the *same* held-out frame; promote only on a clear win."""
    client = mlflow.MlflowClient()
    challenger = load_bundle(settings, "challenger")
    if challenger is None:
        raise RuntimeError("no challenger registered")
    champion = load_bundle(settings, "champion")

    X, y = to_model_input(eval_frame), eval_frame.get_column(TARGET).to_numpy()
    chall_mae = float(np.abs(challenger.predict(X) - y).mean())

    if champion is None:
        decision = PromotionDecision(True, challenger.version, None, chall_mae, None,
                                     "no existing champion")
    elif champion.version == challenger.version:
        decision = PromotionDecision(False, challenger.version, champion.version, chall_mae,
                                     chall_mae, "challenger is already champion")
    else:
        champ_mae = float(np.abs(champion.predict(X) - y).mean())
        gain = (champ_mae - chall_mae) / champ_mae
        promoted = gain >= settings.promotion_min_improvement
        decision = PromotionDecision(
            promoted, challenger.version, champion.version, chall_mae, champ_mae,
            f"relative MAE gain {gain:+.2%} (need >= {settings.promotion_min_improvement:.0%})")

    if decision.promoted:
        client.set_registered_model_alias(settings.model_name, "champion", challenger.version)
        if decision.champion_version:
            client.set_registered_model_alias(settings.model_name, "previous_champion",
                                              decision.champion_version)
    client.set_model_version_tag(settings.model_name, challenger.version, "promotion",
                                 f"{'promoted' if decision.promoted else 'rejected'}: {decision.reason}")
    log.info("promotion gate v%s: %s (%s)", challenger.version,
             "PROMOTED" if decision.promoted else "rejected", decision.reason)
    return decision


def rollback(settings: Settings) -> str:
    """Point `champion` back at `previous_champion` (one-command rollback)."""
    aliases = alias_versions(settings)
    prev = aliases.get("previous_champion")
    if not prev:
        raise RuntimeError("no previous champion to roll back to")
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(settings.model_name, "champion", prev)
    if "champion" in aliases:
        client.set_registered_model_alias(settings.model_name, "previous_champion", aliases["champion"])
    return prev
