from prometheus_client import Counter, Gauge, Histogram
from prometheus_client.core import GaugeMetricFamily
from prometheus_client.registry import Collector

PREDICTIONS = Counter("driftline_predictions_total", "Predictions served", ["role", "version"])
REQUEST_LATENCY = Histogram(
    "driftline_request_latency_seconds", "End-to-end handler latency", ["endpoint"],
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0))
MODEL_LATENCY = Histogram(
    "driftline_model_latency_seconds", "Champion + challenger inference time",
    buckets=(0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.05))
SHADOW_DIFF = Histogram(
    "driftline_shadow_abs_diff_minutes", "|champion - challenger| in shadow mode",
    buckets=(0.25, 0.5, 1, 2, 3, 5, 10, 20))
FEATURE_MISSES = Counter(
    "driftline_feature_store_miss_total", "Online feature lookups with no value", ["feature"])
LOG_FAILURES = Counter("driftline_prediction_log_failures_total", "Prediction log write errors")
MODEL_INFO = Gauge("driftline_model_info", "Loaded model versions", ["role", "version"])


def set_model_info(champion, challenger) -> None:
    MODEL_INFO.clear()
    for role, bundle in (("champion", champion), ("challenger", challenger)):
        if bundle is not None:
            MODEL_INFO.labels(role, bundle.version).set(1)


class DriftCollector(Collector):
    """Exposes the worker's latest drift report at scrape time, so the worker needs no
    HTTP endpoint of its own."""

    def __init__(self, state) -> None:
        self.state = state

    def collect(self):
        psi = GaugeMetricFamily("driftline_feature_psi", "PSI vs training reference", labels=["feature"])
        mae = GaugeMetricFamily("driftline_live_mae_minutes", "MAE on predictions with feedback",
                                labels=["role", "version"])
        alert = GaugeMetricFamily("driftline_drift_alert", "1 when the last check fired a retrain")
        report = self.state.get_report("drift") if self.state else None
        if report:
            for row in report.get("features", []):
                psi.add_metric([row["feature"]], row["psi"])
            for row in report.get("live_mae", []):
                mae.add_metric([row["role"], row["version"]], row["mae"])
            alert.add_metric([], 1.0 if report.get("retrain_triggered") else 0.0)
        yield psi
        yield mae
        yield alert
