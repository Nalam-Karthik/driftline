from driftline.training import registry
from driftline.training import train as train_mod
from driftline.training.train import features_for_month, train


def test_first_model_becomes_champion_with_artifacts(trained):
    aliases = registry.alias_versions(trained)
    assert aliases["champion"] == aliases["challenger"] == "1"

    bundle = registry.load_bundle(trained, "champion")
    assert bundle.stats.source_month == "2024-04"  # serving stats = eval month
    assert {"trip_distance", "prediction", "pu_location_id"} <= set(bundle.profile)
    assert bundle.eval_mae is not None and bundle.eval_mae < 5


def test_worse_challenger_is_rejected(trained, monkeypatch):
    # A crippled model: 3 trees, no zone signal.
    monkeypatch.setitem(train_mod.PARAMS, "n_estimators", 3)
    monkeypatch.setitem(train_mod.PARAMS, "num_leaves", 2)
    train(trained, ["2024-02", "2024-03"], "2024-04")

    decision = registry.promote_if_better(trained, features_for_month(trained, "2024-04", 3000))
    assert not decision.promoted
    assert decision.challenger_mae > decision.champion_mae
    assert registry.alias_versions(trained)["champion"] == "1"


def test_better_challenger_is_promoted_and_can_be_rolled_back(warehouse, monkeypatch):
    monkeypatch.setitem(train_mod.PARAMS, "n_estimators", 3)
    train(warehouse, ["2024-02", "2024-03"], "2024-04")
    registry.promote_if_better(warehouse, features_for_month(warehouse, "2024-04", 3000))
    monkeypatch.undo()

    train(warehouse, ["2024-02", "2024-03"], "2024-04")
    decision = registry.promote_if_better(warehouse, features_for_month(warehouse, "2024-04", 3000))
    assert decision.promoted and decision.champion_version == "1"
    assert registry.alias_versions(warehouse)["champion"] == "2"

    assert registry.rollback(warehouse) == "1"
    aliases = registry.alias_versions(warehouse)
    assert aliases["champion"] == "1" and aliases["previous_champion"] == "2"
