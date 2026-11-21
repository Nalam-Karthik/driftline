"""gRPC front-end over the same Predictor the REST API uses."""

import logging
import signal
from concurrent import futures
from datetime import UTC, datetime

import grpc

from driftline.config import get_settings
from driftline.features.online import make_feature_store
from driftline.serving.predictor import Predictor, Trip
from driftline.serving.proto import predict_pb2, predict_pb2_grpc
from driftline.state import make_state_store

log = logging.getLogger(__name__)
MAX_BATCH = 1000


def _to_trip(req: predict_pb2.TripRequest) -> Trip:
    if not (1 <= req.pu_location_id <= 263 and 1 <= req.do_location_id <= 263):
        raise ValueError("location ids must be in 1..263")
    if not (0 < req.trip_distance <= 100):
        raise ValueError("trip_distance must be in (0, 100]")
    return Trip(
        pickup_at=datetime.fromtimestamp(req.pickup_unix_s, UTC).replace(tzinfo=None),
        pu_location_id=req.pu_location_id,
        do_location_id=req.do_location_id,
        trip_distance=req.trip_distance,
        passenger_count=req.passenger_count or 1,
        ratecode_id=req.ratecode_id or 1,
        request_id=req.request_id or None,
    )


def _to_pb(p) -> predict_pb2.TripPrediction:
    return predict_pb2.TripPrediction(request_id=p.request_id, duration_min=p.duration_min,
                                      model_version=p.model_version, model_role=p.model_role)


class TripDurationService(predict_pb2_grpc.TripDurationServicer):
    def __init__(self, predictor: Predictor) -> None:
        self.predictor = predictor

    def _run(self, requests, context):
        if not self.predictor.ready:
            context.abort(grpc.StatusCode.UNAVAILABLE, "no champion model loaded")
        if len(requests) > MAX_BATCH:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"batch larger than {MAX_BATCH}")
        try:
            trips = [_to_trip(r) for r in requests]
        except ValueError as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        return self.predictor.predict(trips)

    def Predict(self, request, context):
        return _to_pb(self._run([request], context)[0])

    def PredictBatch(self, request, context):
        preds = self._run(list(request.trips), context)
        return predict_pb2.TripPredictions(predictions=[_to_pb(p) for p in preds])


def build_server(predictor: Predictor, port: int, workers: int = 8) -> tuple[grpc.Server, int]:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=workers))
    predict_pb2_grpc.add_TripDurationServicer_to_server(TripDurationService(predictor), server)
    bound = server.add_insecure_port(f"[::]:{port}")
    return server, bound


def main(port: int = 50051) -> None:
    logging.basicConfig(level=logging.INFO)
    s = get_settings()
    predictor = Predictor(s, make_feature_store(s.redis_url), make_state_store(s.redis_url))
    predictor.reload()
    predictor.start_polling()
    server, bound = build_server(predictor, port)
    server.start()
    log.info("gRPC listening on :%d", bound)
    signal.signal(signal.SIGTERM, lambda *_: server.stop(grace=5))
    server.wait_for_termination()


if __name__ == "__main__":
    main()
