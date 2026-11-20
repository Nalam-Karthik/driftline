import numpy as np
import polars as pl
import pytest

from driftline.config import Settings
from driftline.features.build import FEATURES
from driftline.monitoring.checks import MIN_RECORDS_FOR_DRIFT, evaluate_window
from driftline.monitoring.drift import build_profile, profile_from_json, profile_to_json, psi, score_drift
from driftline.state import InMemoryStateStore

rng = np.random.default_rng(0)


def frame(n=5000, distance_scale=1.5, zones=(1, 2, 3, 4)):
    return pl.DataFrame({
        "trip_distance": rng.gamma(2.0, distance_scale, n),
        "pu_location_id": rng.choice(zones, n),
    })


def test_psi_is_zero_for_identical_and_large_for_disjoint():
    assert psi([0.25] * 4, [0.25] * 4) == pytest.approx(0.0)
    assert psi([0.5, 0.5, 0, 0], [0, 0, 0.5, 0.5]) > 5


def test_same_distribution_is_stable_shifted_is_flagged():
    profile = build_profile(frame(), ["pu_location_id"])
    same = {d.feature: d for d in score_drift(profile, frame())}
    shifted = {d.feature: d for d in score_drift(profile, frame(distance_scale=3.0, zones=(5, 6)))}

    assert same["trip_distance"].psi < 0.05 and not same["trip_distance"].drifted
    assert same["pu_location_id"].psi < 0.05
    assert shifted["trip_distance"].drifted and shifted["trip_distance"].ks_pvalue < 1e-6
    assert shifted["pu_location_id"].drifted  # all mass moved to the "other" bucket


def test_profile_round_trips_through_json():
    profile = build_profile(frame(), ["pu_location_id"])
    restored = profile_from_json(profile_to_json(profile))
    current = frame()
    assert score_drift(restored, current) == score_drift(profile, current)


def _records(n, profile_frame, *, champion="1", pred_offset=0.0):
    rows = profile_frame.to_dicts()
    return [{"request_id": f"r{i}", "role": "champion", "version": champion,
             "champion_version": champion, "challenger_version": "2",
             "champion_prediction": rows[i % len(rows)]["prediction"] + pred_offset,
             "challenger_prediction": rows[i % len(rows)]["prediction"],
             "features": {k: rows[i % len(rows)].get(k) for k in FEATURES}} for i in range(n)]


@pytest.fixture
def reference():
    n = 4000
    df = pl.DataFrame({f: rng.normal(5, 1, n) for f in FEATURES} | {"prediction": rng.normal(12, 3, n)})
    return df, build_profile(df, ["pu_location_id", "do_location_id", "ratecode_id"])


def test_quiet_window_does_not_trigger(reference):
    df, profile = reference
    report = evaluate_window(_records(3000, df), {}, profile, "1", 3.0, Settings())
    assert report["records"] == 3000 and not report["retrain_triggered"]


def test_drift_on_key_feature_triggers_retrain(reference):
    df, profile = reference
    shifted = df.with_columns(pl.col("trip_distance") * 3)
    report = evaluate_window(_records(3000, shifted), {}, profile, "1", 3.0, Settings())
    assert report["retrain_triggered"]
    assert any("trip_distance" in r for r in report["reasons"])


def test_live_mae_degradation_triggers_and_scores_both_models(reference):
    df, profile = reference
    records = _records(2000, df, pred_offset=5.0)  # champion is now 5 minutes off
    feedback = {r["request_id"]: r["challenger_prediction"] for r in records}
    report = evaluate_window(records, feedback, profile, "1", 3.0, Settings())

    by_role = {m["role"]: m for m in report["live_mae"]}
    assert by_role["champion"]["mae"] == pytest.approx(5.0)
    assert by_role["challenger"]["mae"] == pytest.approx(0.0)
    assert any("MAE" in r for r in report["reasons"])


def test_small_or_stale_windows_are_skipped(reference):
    df, profile = reference
    small = evaluate_window(_records(MIN_RECORDS_FOR_DRIFT - 1, df), {}, profile, "1", 3.0, Settings())
    old_model = evaluate_window(_records(3000, df, champion="0"), {}, profile, "1", 3.0, Settings())
    assert "skipped" in small and not small["retrain_triggered"]
    assert old_model["records"] == 0  # records from a previous champion are ignored


def test_retrain_lock_allows_one_holder():
    store = InMemoryStateStore()
    assert store.acquire_lock("lock:retrain", 60)
    assert not store.acquire_lock("lock:retrain", 60)
