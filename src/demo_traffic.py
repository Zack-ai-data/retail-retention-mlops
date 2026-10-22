"""Synthetic traffic for the API, so a demonstration has something to show.

THIS IS NOT REAL USAGE. Without it the API's dashboards are flat, because nobody is calling a
local service. Every request carries the header X-Demo-Traffic: true, and the API counts such
requests under traffic="demo" (and everything else under traffic="real"), so the Grafana
dashboards can show them apart and a demonstration is never mistaken for usage.

The rate rises and falls in a wave, and a small share of requests are ones that fail on purpose
the way real callers' requests do (a customer that does not exist, a malformed id, a route that
is not there), so the error panels move too. None of them can cause a server error (5xx).

    docker compose up -d api
    docker compose run --rm app python -m src.demo_traffic --api-url http://api:8000 --minutes 5
"""

import argparse
import logging
import math
import random
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import requests

from src.metrics import DEMO_TRAFFIC_HEADER

log = logging.getLogger("demo_traffic")

BANNER = (
    "DEMO TRAFFIC: synthetic requests, not real usage. Each carries the header "
    f'{DEMO_TRAFFIC_HEADER}: true and is counted under traffic="demo" in the API\'s metrics.'
)
WAVE_SECONDS = 60.0  # the rate rises and falls once a minute
WAVE_DEPTH = 0.6  # by up to this share of the average rate, either way
MAX_FAILURES_IN_A_ROW = 30  # then the API is taken to be down
REQUEST_TIMEOUT_SECONDS = 10.0
NO_SUCH_CUSTOMER = 10_000_000  # ids from here up are not in the dataset


@dataclass(frozen=True)
class Call:
    """One request to send."""

    method: str
    path: str
    params: dict[str, str | int] | None = None


# What callers do, and how often. The shares add up to 100. The last three kinds are requests that
# fail on purpose with a client error (404 or 422), never a server error.
MIX: list[tuple[str, int]] = [
    ("score", 35),
    ("prediction", 25),
    ("features", 10),
    ("health", 10),
    ("outreach", 8),
    ("unknown_customer", 6),
    ("malformed_id", 3),
    ("unknown_route", 3),
]


def pick_call(rng: random.Random, customers: Sequence[int]) -> Call:
    """The next request: a kind drawn from MIX, and a customer drawn from those that exist."""
    kind = rng.choices([name for name, _ in MIX], weights=[share for _, share in MIX])[0]
    customer = rng.choice(customers)
    if kind == "score":
        return Call("GET", f"/customers/{customer}/score")
    if kind == "prediction":
        return Call("GET", f"/customers/{customer}/prediction")
    if kind == "features":
        return Call("GET", f"/customers/{customer}/features")
    if kind == "health":
        return Call("GET", "/health")
    if kind == "outreach":
        return Call("GET", "/outreach/latest", {"limit": rng.choice([10, 50, 100, 200])})
    if kind == "unknown_customer":
        return Call("GET", f"/customers/{NO_SUCH_CUSTOMER + rng.randint(0, 999)}/score")
    if kind == "malformed_id":
        return Call("GET", "/customers/not-a-number/score")
    return Call("GET", "/demo/no/such/route")


def requests_this_second(rng: random.Random, average: float, elapsed: float) -> int:
    """How many requests to send in the second starting `elapsed` seconds in: a wave around the
    average with a little noise, never negative."""
    wave = 1 + WAVE_DEPTH * math.sin(2 * math.pi * elapsed / WAVE_SECONDS)
    return max(0, round(average * wave + rng.uniform(-0.5, 0.5)))


class DemoTrafficError(Exception):
    """The demo cannot run (the API is down, or has no customers to ask about)."""


def fetch_customers(session: requests.Session, api_url: str, limit: int = 500) -> list[int]:
    """Ids of customers that exist, from the latest outreach plan."""
    try:
        response = session.get(
            f"{api_url}/outreach/latest",
            params={"limit": limit},
            headers={DEMO_TRAFFIC_HEADER: "true"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise DemoTrafficError(
            f"Cannot reach the API at {api_url}: {exc}. Is it running? "
            "Try: docker compose up -d api"
        ) from exc
    if response.status_code != 200:
        raise DemoTrafficError(
            f"The API answered {response.status_code} for {api_url}/outreach/latest, so there "
            "are no customers to ask about. Run the pipeline first: python -m src.main --stage plan"
        )
    customers = [int(row["customer_id"]) for row in response.json().get("customers", [])]
    if not customers:
        raise DemoTrafficError("The latest outreach plan lists no customers.")
    return customers


def run(
    session: requests.Session,
    api_url: str,
    customers: Sequence[int],
    seconds: float,
    rate: float,
    rng: random.Random,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Counter[str]:
    """Send traffic for `seconds`, and return how many requests got which status."""
    outcomes: Counter[str] = Counter()
    failures_in_a_row = 0
    started = clock()
    while (elapsed := clock() - started) < seconds:
        second_started = clock()
        count = requests_this_second(rng, rate, elapsed)
        for _ in range(count):
            call = pick_call(rng, customers)
            try:
                response = session.request(
                    call.method,
                    f"{api_url}{call.path}",
                    params=call.params,
                    headers={DEMO_TRAFFIC_HEADER: "true"},
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
            except requests.RequestException:
                outcomes["no answer"] += 1
                failures_in_a_row += 1
                if failures_in_a_row >= MAX_FAILURES_IN_A_ROW:
                    raise DemoTrafficError(
                        f"The API at {api_url} stopped answering. Is it still running?"
                    ) from None
            else:
                failures_in_a_row = 0
                outcomes[str(response.status_code)] += 1
            # pace every request, answered or not, so a struggling API is never hammered
            sleep(max(0.0, 1.0 / max(count, 1) - 0.005))
        sleep(max(0.0, 1.0 - (clock() - second_started)) if count == 0 else 0.0)
    return outcomes


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--api-url", default="http://localhost:8000", help="default %(default)s")
    parser.add_argument(
        "--minutes", type=float, default=5.0, help="how long to run (default %(default)s)"
    )
    parser.add_argument(
        "--rate", type=float, default=3.0, help="average requests per second (default %(default)s)"
    )
    parser.add_argument("--seed", type=int, default=None, help="make the traffic repeatable")
    args = parser.parse_args(argv)
    if args.minutes <= 0 or args.rate <= 0:
        parser.error("--minutes and --rate must be above zero")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log.info(BANNER)
    session = requests.Session()
    api_url = args.api_url.rstrip("/")
    try:
        customers = fetch_customers(session, api_url)
        log.info(
            "Sending about %.1f requests a second for %.1f minutes (rising and falling), "
            "asking about %d customers",
            args.rate,
            args.minutes,
            len(customers),
        )
        outcomes = run(
            session,
            api_url,
            customers,
            args.minutes * 60,
            args.rate,
            random.Random(args.seed),
            time.monotonic,
            time.sleep,
        )
    except DemoTrafficError as exc:
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        log.info("Stopped.")
        return 0
    total = sum(outcomes.values())
    log.info(
        "Done: %d demo requests: %s. This was synthetic traffic, not real usage.",
        total,
        ", ".join(f"{status}: {count}" for status, count in sorted(outcomes.items())),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
