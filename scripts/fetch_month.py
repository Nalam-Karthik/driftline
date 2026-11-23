"""Download NYC TLC yellow-taxi months into data/raw (skips files already present).

    python scripts/fetch_month.py 2024-01 2024-02
    python scripts/fetch_month.py --previous            # last calendar month
"""

import argparse
import os
import sys
import urllib.request
from datetime import date
from pathlib import Path

BASE = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{}.parquet"


def previous_month(today: date) -> str:
    y, m = (today.year - 1, 12) if today.month == 1 else (today.year, today.month - 1)
    return f"{y:04d}-{m:02d}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("months", nargs="*")
    ap.add_argument("--previous", action="store_true")
    ap.add_argument("--print-month", action="store_true")
    args = ap.parse_args()
    months = args.months + ([previous_month(date.today())] if args.previous else [])
    if args.print_month:
        print(" ".join(months))
        return

    raw = Path(os.environ.get("DRIFTLINE_DATA_DIR", "data")) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for month in months:
        dest = raw / f"yellow_tripdata_{month}.parquet"
        if dest.exists():
            continue
        tmp = dest.with_suffix(".part")
        print(f"downloading {month} ...", file=sys.stderr)
        urllib.request.urlretrieve(BASE.format(month), tmp)
        tmp.rename(dest)  # never leave a half-written parquet where ingest can see it


if __name__ == "__main__":
    main()
