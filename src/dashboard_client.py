"""What the Streamlit dashboard knows about the API, kept out of the UI so it can be tested.

The dashboard never touches the database or the models: it asks the API, so a number on the
screen is always one the API would give. This module holds the HTTP client and the small
calculations behind each page. It imports only requests and pandas, so the dashboard image does not
need the training stack.
"""

import csv
import io
import math
import os
from typing import Any

import pandas as pd
import requests

DEFAULT_API_URL = "http://localhost:8000"

# What an outreach list for the people doing the contacting contains. The holdout group is never
# in it: those customers are deliberately left alone so impact can be measured later.
CONTACT_LIST_COLUMNS = ["customer_id", "rank", "p_churn", "spend_if_retained", "margin_at_risk"]

SCORE_FIELDS = [
    "p_churn",
    "spend_if_retained",
    "expected_spend",
    "margin_at_risk",
    "margin_at_risk_after_cost",
]


class ApiError(Exception):
    """The API could not be reached or refused the request; the message is fit to show a person."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def _detail(response: requests.Response) -> str:
    """The reason in an error response, as one readable line."""
    try:
        detail = response.json().get("detail", response.text)
    except (ValueError, AttributeError):
        return response.text or f"HTTP {response.status_code}"
    if isinstance(detail, str):
        return detail
    if isinstance(detail, list):  # request validation errors
        return "; ".join(
            f"{'.'.join(str(p) for p in item.get('loc', [])[1:])}: {item.get('msg', 'invalid')}"
            for item in detail
        )
    return str(detail.get("problem", detail)) if isinstance(detail, dict) else str(detail)


class ApiClient:
    def __init__(
        self,
        base_url: str | None = None,
        session: requests.Session | None = None,
        timeout: float = 20.0,
    ) -> None:
        self.base_url = (base_url or os.getenv("API_URL") or DEFAULT_API_URL).rstrip("/")
        self.session = session or requests.Session()
        self.timeout = timeout

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        try:
            return self.session.request(
                method, f"{self.base_url}{path}", timeout=self.timeout, **kwargs
            )
        except requests.ConnectionError as exc:
            raise ApiError(
                f"The API is not reachable at {self.base_url}. "
                "Start it with: docker compose up -d api"
            ) from exc
        except requests.Timeout as exc:
            raise ApiError(f"The API at {self.base_url} took too long to answer.") from exc

    def _json(self, method: str, path: str, **kwargs: Any) -> Any:
        response = self._request(method, path, **kwargs)
        if response.status_code >= 400:
            raise ApiError(_detail(response), response.status_code)
        return response.json()

    def health(self) -> dict[str, Any]:
        """The API's health. A degraded API (models not loaded, database down) is reported, not
        raised, so the dashboard can show what is wrong."""
        response = self._request("GET", "/health")
        if response.status_code == 503:
            detail = response.json().get("detail")
            if isinstance(detail, dict):
                return detail
        if response.status_code >= 400:
            raise ApiError(_detail(response), response.status_code)
        return dict(response.json())

    def reload_models(self) -> dict[str, Any]:
        return dict(self._json("POST", "/models/reload"))

    def score(self, customer_id: int) -> dict[str, Any]:
        return dict(self._json("GET", f"/customers/{int(customer_id)}/score"))

    def prediction(self, customer_id: int) -> dict[str, Any]:
        return dict(self._json("GET", f"/customers/{int(customer_id)}/prediction"))

    def customer_features(self, customer_id: int) -> dict[str, Any]:
        return dict(self._json("GET", f"/customers/{int(customer_id)}/features"))

    def what_if(self, features: dict[str, float | None]) -> dict[str, Any]:
        return dict(self._json("POST", "/score", json=features))

    def outreach_plan(
        self, group: str | None = None, limit: int = 5000, offset: int = 0
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if group:
            params["group"] = group
        return dict(self._json("GET", "/outreach/latest", params=params))

    def sensitivity(
        self, k: int, margins: list[float], contact_costs: list[float]
    ) -> list[dict[str, Any]]:
        rows = self._json(
            "GET",
            "/outreach/sensitivity",
            params={"k": k, "margins": margins, "contact_costs": contact_costs},
        )
        return [dict(row) for row in rows]


# ------------------------------------------------------------------ page calculations


def contact_list_csv(plan: dict[str, Any]) -> bytes:
    """The list to hand to whoever does the contacting: the 'contact' group only, never the
    holdout, in rank order."""
    rows = sorted(
        (c for c in plan["customers"] if c["group"] == "contact"), key=lambda c: c["rank"]
    )
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CONTACT_LIST_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def plan_totals(plan: dict[str, Any]) -> dict[str, Any]:
    """Customers, cost and margin at risk for the plan's groups.

    complete is False when the customer rows given are fewer than the plan says it selected, so
    the margin figures would be understated.
    """
    customers = plan["customers"]
    contact = [c for c in customers if c["group"] == "contact"]
    holdout = [c for c in customers if c["group"] == "holdout"]
    return {
        "contact": plan["contact_customers"],
        "holdout": plan["holdout_customers"],
        "cost": plan["contact_customers"] * plan["contact_cost"],
        "margin_at_risk_contacted": sum(c["margin_at_risk"] for c in contact),
        "margin_at_risk_held_out": sum(c["margin_at_risk"] for c in holdout),
        "complete": len(contact) == plan["contact_customers"]
        and len(holdout) == plan["holdout_customers"],
    }


def compare_scores(online: dict[str, Any], stored: dict[str, Any]) -> pd.DataFrame:
    """The online and stored answers for a customer side by side, and whether they agree.

    They should always agree for the same model versions; if the models were retrained since the
    last scoring run they can differ, and the versions say why.
    """
    rows = []
    for field in SCORE_FIELDS:
        rows.append(
            {
                "measure": field,
                "online": online[field],
                "stored": stored[field],
                "agree": math.isclose(online[field], stored[field], rel_tol=1e-9, abs_tol=1e-9),
            }
        )
    for field in ("churn_model_version", "value_model_version"):
        rows.append(
            {
                "measure": field,
                "online": online[field],
                "stored": stored[field],
                "agree": online[field] == stored[field],
            }
        )
    return pd.DataFrame(rows)


def sensitivity_tables(rows: list[dict[str, Any]]) -> dict[str, pd.DataFrame]:
    """The sensitivity as two grids (rows = margin, columns = contact cost): how many customers
    are selected, and how much of the baseline list is still selected."""
    frame = pd.DataFrame(rows)
    return {
        "selected": frame.pivot(index="margin", columns="contact_cost", values="selected"),
        "overlap": frame.pivot(
            index="margin", columns="contact_cost", values="overlap_with_baseline"
        ),
    }


def assumptions_make_no_difference(rows: list[dict[str, Any]]) -> bool:
    """True when every combination selects exactly the baseline list: the margin and contact-cost
    assumptions do not matter, because the fixed budget decides who is on the list."""
    return bool(rows) and all(r["overlap_with_baseline"] == 1.0 for r in rows)


def money(amount: float) -> str:
    return f"£{amount:,.0f}"


def percent(share: float) -> str:
    return f"{share:.0%}"
