"""driftline CLI — thin wrapper over the Prefect flows and services."""

import argparse
import json
import logging


def main() -> None:
    ap = argparse.ArgumentParser(prog="driftline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", help="validate + load months into the warehouse")
    p.add_argument("months", nargs="+")

    p = sub.add_parser("train", help="train a challenger and run the promotion gate")
    p.add_argument("--train", nargs="+", required=True, help="training months, e.g. 2024-01 2024-02")
    p.add_argument("--eval", required=True, help="held-out month for metrics and the gate")

    p = sub.add_parser("refresh-features", help="publish a month's zone stats to the online store")
    p.add_argument("month")

    sub.add_parser("rollback", help="point champion back at previous_champion")

    p = sub.add_parser("serve", help="REST API")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--workers", type=int, default=1)

    p = sub.add_parser("grpc", help="gRPC API")
    p.add_argument("--port", type=int, default=50051)

    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    if args.cmd == "ingest":
        from driftline.pipeline.flows import ingest_flow
        print(json.dumps(ingest_flow(args.months), indent=2, default=str))
    elif args.cmd == "train":
        from driftline.pipeline.flows import training_flow
        print(json.dumps(training_flow(args.train, args.eval), indent=2, default=str))
    elif args.cmd == "refresh-features":
        from driftline.pipeline.flows import refresh_features_flow
        print(refresh_features_flow(args.month))
    elif args.cmd == "rollback":
        from driftline.config import get_settings
        from driftline.training.registry import rollback
        print(f"champion -> v{rollback(get_settings())}")
    elif args.cmd == "serve":
        import uvicorn
        uvicorn.run("driftline.serving.app:app", host=args.host, port=args.port, workers=args.workers)
    elif args.cmd == "grpc":
        from driftline.serving.grpc_server import main as grpc_main
        grpc_main(args.port)


if __name__ == "__main__":
    main()
