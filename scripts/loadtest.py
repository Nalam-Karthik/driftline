"""Closed-loop load test against a running API using real trips from a held-out month.

    python scripts/loadtest.py --url http://localhost:8000 --month 2024-04 --requests 5000 --concurrency 32
    python scripts/loadtest.py ... --feedback   # also post true durations so live MAE can be computed
"""

import argparse
import asyncio
import json
import time

import httpx
import numpy as np

from driftline.config import get_settings
from driftline.data.ingest import read_trips


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--month", default="2024-04")
    ap.add_argument("--requests", type=int, default=5000)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--feedback", action="store_true")
    args = ap.parse_args()

    trips = read_trips(get_settings().warehouse_path, [args.month], args.requests, seed=1).to_dicts()
    queue: asyncio.Queue = asyncio.Queue()
    for t in trips:
        queue.put_nowait(t)
    latencies, errors, abs_err = [], 0, []

    async def worker(client: httpx.AsyncClient):
        nonlocal errors
        while not queue.empty():
            t = queue.get_nowait()
            body = {"pickup_at": t["pickup_at"].isoformat(), "pu_location_id": t["pu_location_id"],
                    "do_location_id": t["do_location_id"], "trip_distance": t["trip_distance"],
                    "passenger_count": t["passenger_count"], "ratecode_id": t["ratecode_id"]}
            t0 = time.perf_counter()
            r = await client.post("/predict", json=body)
            latencies.append(time.perf_counter() - t0)
            if r.status_code != 200:
                errors += 1
                continue
            pred = r.json()
            abs_err.append(abs(pred["duration_min"] - t["duration_min"]))
            if args.feedback:
                await client.post("/feedback", json={"request_id": pred["request_id"],
                                                     "actual_duration_min": t["duration_min"]})

    limits = httpx.Limits(max_connections=args.concurrency)
    async with httpx.AsyncClient(base_url=args.url, timeout=10, limits=limits) as client:
        start = time.perf_counter()
        await asyncio.gather(*(worker(client) for _ in range(args.concurrency)))
        wall = time.perf_counter() - start

    ms = np.array(latencies) * 1000
    print(json.dumps({
        "requests": len(latencies), "errors": errors, "concurrency": args.concurrency,
        "throughput_rps": round(len(latencies) / wall, 1),
        "latency_ms": {"p50": round(float(np.percentile(ms, 50)), 2),
                       "p95": round(float(np.percentile(ms, 95)), 2),
                       "p99": round(float(np.percentile(ms, 99)), 2)},
        "served_mae_min": round(float(np.mean(abs_err)), 3) if abs_err else None,
        "with_feedback": args.feedback,
    }, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
