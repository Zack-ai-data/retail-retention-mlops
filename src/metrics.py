"""Prometheus metrics for the API.

What is measured, and why:

  retail_api_requests_total / retail_api_request_duration_seconds
      traffic, errors and latency per route. The route label is the route *template*
      (/customers/{customer_id}/score), never the URL that was requested: a label per customer
      would create a new time series for every customer and put customer ids into the metrics.
      A request that matches no route is counted under "unmatched" for the same reason. The
      traffic label is "demo" for requests sent by the demo traffic script (src/demo_traffic.py,
      which sets a header) and "real" for everything else, so a demonstration can never be mistaken
      for usage. It can only ever take those two values, whatever a caller sends.
  retail_api_model_version
      the version of each champion model the API is serving. The API loads the champions when it
      starts and on POST /models/reload, so after a retrain promotes a new champion it keeps
      serving the old one until someone reloads. Compared with the champion in the registry, this
      shows that gap.
  retail_api_model_loads_total
      how often loading the models worked or failed.

Each app gets its own registry, so tests that build several apps do not collide on metric names.
"""

import time
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

METRICS_PATH = "/metrics"
UNMATCHED = "unmatched"
DEMO_TRAFFIC_HEADER = "X-Demo-Traffic"  # set to "true" by src/demo_traffic.py
REAL, DEMO = "real", "demo"

# Seconds. Most answers take milliseconds; a what-if score or the outreach plan can take longer.
_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)


class ApiMetrics:
    """The API's metrics, its own registry, and the middleware that records each request."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "retail_api_requests_total",
            "Requests handled, by method, route template, status code and traffic type",
            ["method", "route", "status", "traffic"],
            registry=self.registry,
        )
        self.duration = Histogram(
            "retail_api_request_duration_seconds",
            "Time to answer a request, by method, route template and traffic type",
            ["method", "route", "traffic"],
            buckets=_BUCKETS,
            registry=self.registry,
        )
        self.in_progress = Gauge(
            "retail_api_requests_in_progress",
            "Requests being handled right now",
            registry=self.registry,
        )
        self.model_version = Gauge(
            "retail_api_model_version",
            "Version of the champion model the API is serving (0 when none is loaded)",
            ["model"],
            registry=self.registry,
        )
        self.model_loads = Counter(
            "retail_api_model_loads_total",
            "Attempts to load the champion models, by result (ok or failed)",
            ["result"],
            registry=self.registry,
        )
        for model in ("churn", "value"):
            self.model_version.labels(model=model).set(0)

    def observe_models(self, champions: Any | None) -> None:
        """Record the result of loading the champions (None when loading failed)."""
        if champions is None:
            self.model_loads.labels(result="failed").inc()
            versions = {"churn": 0, "value": 0}
        else:
            self.model_loads.labels(result="ok").inc()
            versions = {
                "churn": int(champions.churn_version),
                "value": int(champions.value_version),
            }
        for model, version in versions.items():
            self.model_version.labels(model=model).set(version)

    async def middleware(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Time the request and count it under its route template and status."""
        if request.url.path == METRICS_PATH:  # scraping must not count itself
            return await call_next(request)
        traffic = DEMO if request.headers.get(DEMO_TRAFFIC_HEADER, "").lower() == "true" else REAL
        started = time.perf_counter()
        status = 500  # what an unhandled error becomes
        self.in_progress.inc()
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            self.in_progress.dec()
            # the router puts the matched route in the scope; without one nothing matched
            route = getattr(request.scope.get("route"), "path", None) or UNMATCHED
            self.requests.labels(request.method, route, str(status), traffic).inc()
            self.duration.labels(request.method, route, traffic).observe(
                time.perf_counter() - started
            )

    def render(self) -> Response:
        """The metrics in the Prometheus text format."""
        return Response(generate_latest(self.registry), media_type=CONTENT_TYPE_LATEST)
