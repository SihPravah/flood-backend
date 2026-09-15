"""Publish a canonical PRAVAHA sensor event to the backend ingestion API."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_URL = "http://127.0.0.1:8000/api/v1/ingest/sensors"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "POST a canonical raw sensor observation through the same backend "
            "ingestion endpoint used by IoT gateways."
        )
    )
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--device", default="UK-SNS-00127")
    parser.add_argument("--rainfall", type=float, required=True)
    parser.add_argument(
        "--soil",
        type=float,
        required=True,
        help="Soil saturation fraction 0..1 or soil moisture percentage 0..100.",
    )
    parser.add_argument(
        "--tilt",
        type=float,
        default=0.0,
        help="Optional local sensor tilt in degrees; defaults to 0.0.",
    )
    parser.add_argument("--lat", type=float, default=30.285029)
    parser.add_argument("--lon", type=float, default=77.978689)
    parser.add_argument("--village", default="Chandrabani")
    parser.add_argument("--ward", default="Chandrabani settlement point")
    parser.add_argument("--observed-at", default=None)
    parser.add_argument("--received-at", default=None)
    parser.add_argument(
        "--provenance",
        choices=["OBSERVED", "DERIVED", "ESTIMATED", "SIMULATED", "MISSING"],
        default="OBSERVED",
    )
    args = parser.parse_args()

    payload = build_payload(args)
    request = Request(
        args.url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=10) as response:
            body = response.read().decode("utf-8")
            print_json(json.loads(body))
            return 0
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(body or str(exc), file=sys.stderr)
        return 1
    except URLError as exc:
        print(f"Failed to reach backend ingestion endpoint: {exc}", file=sys.stderr)
        return 1


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "device_id": args.device,
        "observed_at": args.observed_at or utc_now(),
        "location": {
            "village": args.village,
            "ward": args.ward,
            "lat": args.lat,
            "lon": args.lon,
        },
        "sensor_metrics": {
            "rainfall_mm_per_hr": args.rainfall,
            "soil_moisture_percentage": normalize_soil(args.soil),
        },
        "provenance": args.provenance,
    }
    if args.received_at:
        payload["received_at"] = args.received_at
    payload["sensor_metrics"]["slope_tilt_degrees"] = args.tilt
    return payload


def normalize_soil(value: float) -> float:
    if 0.0 <= value <= 1.0:
        return round(value * 100.0, 3)
    return value


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
