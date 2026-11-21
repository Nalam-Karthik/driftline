from datetime import datetime

import grpc
import pytest
from fastapi.testclient import TestClient

from driftline.features.online import InMemoryFeatureStore
from driftline.serving.app import create_app
from driftline.serving.grpc_server import build_server
from driftline.serving.predictor import Predictor, in_canary
from driftline.serving.proto import predict_pb2, predict_pb2_grpc
from driftline.state import InMemoryStateStore
from driftline.training import train as train_mod
from driftline.training.train import train
from tests.conftest import sample_trip


@pytest.fixture
def predictor(trained, monkeypatch):
    # Register a distinct challenger so shadow/canary paths are exercised.
    monkeypatch.setitem(train_mod.PARAMS, "n_estimators", 20)
    train(trained, ["2024-02", "2024-03"], "2024-04")
    p = Predictor(trained, InMemoryFeatureStore(), InMemoryStateStore())
    p.reload()
    return p


@pytest.fixture
def client(predictor):
    with TestClient(create_app(predictor)) as c:
        yield c


def test_predict_shadow_mode_serves_champion_and_logs_both(client, predictor):
    r = client.post("/predict", json=sample_trip(request_id="abc"))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model_role"] == "champion" and body["model_version"] == "1"
    assert body["shadow_duration_min"] is not None
    assert body["feature_version"] == "2024-04"
    assert 1 <= body["duration_min"] < 120

    logged = predictor.state.recent_predictions(1)[0]
    assert logged["request_id"] == "abc" and logged["challenger_version"] == "2"
    assert logged["features"]["trip_distance"] == 2.4


def test_canary_routing_is_sticky_and_close_to_fraction(predictor):
    predictor.settings.routing_mode = "canary"
    predictor.settings.canary_fraction = 0.2
    ids = [f"req-{i}" for i in range(5000)]
    share = sum(in_canary(i, 0.2) for i in ids) / len(ids)
    assert 0.18 < share < 0.22

    from driftline.serving.predictor import Trip
    trips = [Trip(datetime(2024, 5, 1, 9), 3, 7, 2.0, request_id=i) for i in ids[:200]]
    first = [p.model_role for p in predictor.predict(trips)]
    trips = [Trip(datetime(2024, 5, 1, 9), 3, 7, 2.0, request_id=i) for i in ids[:200]]
    assert first == [p.model_role for p in predictor.predict(trips)]
    assert "challenger" in first and "champion" in first


def test_batch_validation_and_feedback(client, predictor):
    r = client.post("/predict/batch", json={"trips": [sample_trip(), sample_trip(trip_distance=9.0)]})
    assert r.status_code == 200 and len(r.json()) == 2
    short, long = r.json()
    assert long["duration_min"] > short["duration_min"]

    assert client.post("/predict", json=sample_trip(pu_location_id=999)).status_code == 422
    assert client.post("/predict/batch", json={"trips": []}).status_code == 422

    rid = short["request_id"]
    assert client.post("/feedback", json={"request_id": rid, "actual_duration_min": 9.5}).status_code == 202
    assert predictor.state.feedback_for([rid]) == {rid: 9.5}


def test_health_models_and_metrics(client):
    assert client.get("/ready").json() == {"status": "ready"}
    models = client.get("/models").json()
    assert models["champion"]["version"] == "1" and models["challenger"]["version"] == "2"
    client.post("/predict", json=sample_trip())
    text = client.get("/metrics").text
    assert 'driftline_predictions_total{role="champion",version="1"}' in text
    assert "driftline_shadow_abs_diff_minutes_bucket" in text
    assert client.get("/monitoring/drift").status_code == 404  # no worker run yet


def test_not_ready_without_champion(settings):
    p = Predictor(settings, InMemoryFeatureStore(), InMemoryStateStore())
    with TestClient(create_app(p)) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/ready").status_code == 503
        assert c.post("/predict", json=sample_trip()).status_code == 503


def test_grpc_predict_and_batch(predictor):
    server, port = build_server(predictor, 0, workers=2)
    server.start()
    try:
        with grpc.insecure_channel(f"localhost:{port}") as channel:
            stub = predict_pb2_grpc.TripDurationStub(channel)
            req = predict_pb2.TripRequest(pickup_unix_s=int(datetime(2024, 5, 3, 8, 30).timestamp()),
                                          pu_location_id=3, do_location_id=7, trip_distance=2.4,
                                          request_id="g1")
            one = stub.Predict(req)
            assert one.request_id == "g1" and one.model_version == "1" and one.duration_min >= 1

            batch = stub.PredictBatch(predict_pb2.TripBatch(trips=[req, req]))
            assert len(batch.predictions) == 2

            with pytest.raises(grpc.RpcError) as err:
                stub.Predict(predict_pb2.TripRequest(pu_location_id=500, do_location_id=1,
                                                     trip_distance=1.0))
            assert err.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    finally:
        server.stop(0)
