"""Unit tests for §17.2.1 OSM extractor.

The Overpass HTTP path is replaced with a deterministic in-memory fake so
tests don't depend on the real network. Query building, response parsing,
retry behaviour, and the global target totals are validated here; the actual
2 500-entry acceptance check fires only when `fetch_all_addresses` runs
against a real Overpass instance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from eval.synthetic.extract_osm import (
    BUDAPEST_DISTRICT_TARGETS,
    MIN_TOTAL_ADDRESSES,
    OVERPASS_URLS,
    REGIONAL_CITY_TARGETS,
    RURAL_TARGETS,
    Address,
    build_overpass_query,
    fetch_all_addresses,
    fetch_region,
    parse_overpass_response,
    validate_address_file,
)

# ---------------------------------------------------------------------------
# Fake httpx client + response
# ---------------------------------------------------------------------------


@dataclass
class _FakeResponse:
    status_code: int
    payload: dict[str, Any]
    request: httpx.Request

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=self.request, response=cast(httpx.Response, self))

    def json(self) -> dict[str, Any]:
        return self.payload


class _FakeClient:
    """Records post() calls; returns canned responses keyed by region."""

    def __init__(self, *, responses_by_region: dict[tuple[str, str | None], dict[str, Any]] | None = None) -> None:
        self._responses = responses_by_region or {}
        self.calls: list[str] = []
        self.failures: dict[tuple[str, str | None], int] = {}
        self.aclosed = False

    async def post(
        self,
        url: str,
        *,
        data: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,  # noqa: ASYNC109 — fake mirrors httpx.AsyncClient.post
    ) -> _FakeResponse:
        _ = timeout
        assert url in OVERPASS_URLS
        assert headers
        query = (data or {}).get("data") or ""
        self.calls.append(query)
        region = self._region_from_query(query)

        # If this region is set to fail N times, decrement and raise.
        remaining = self.failures.get(region, 0)
        if remaining > 0:
            self.failures[region] = remaining - 1
            request = httpx.Request("POST", url)
            return _FakeResponse(status_code=500, payload={}, request=request)

        payload = self._responses.get(region, {"elements": []})
        request = httpx.Request("POST", url)
        return _FakeResponse(status_code=200, payload=payload, request=request)

    async def aclose(self) -> None:
        self.aclosed = True

    @staticmethod
    def _region_from_query(query: str) -> tuple[str, str | None]:
        """Recover (city, district) from the query string."""
        if 'admin_level"="9"' in query:
            # Budapest district ref like `"ref"="13"`, scoped to the Budapest
            # administrative area before map_to_area.
            start = query.find('"ref"="')
            if start == -1:
                return ("Budapest", None)
            district = query[start + len('"ref"="') :].split('"', maxsplit=1)[0]
            return ("Budapest", district)
        # Otherwise the query names the city as name="<city>".
        start = query.find('"name"="')
        if start == -1:
            return ("?", None)
        rest = query[start + len('"name"="') :]
        city = rest.split('"')[0]
        return (city, None)


# ---------------------------------------------------------------------------
# Query + parser
# ---------------------------------------------------------------------------


def test_build_overpass_query_for_budapest_district() -> None:
    q = build_overpass_query(city="Budapest", district="13", limit=50)
    assert '"name"="Budapest"' in q
    assert 'admin_level"="9"' in q
    assert '"ref"="13"' in q
    assert "map_to_area->.searchArea;" in q
    assert 'node["addr:housenumber"]["addr:street"]' in q
    assert 'way["building"]["addr:housenumber"]["addr:street"]' in q
    assert "out center 50;" in q


def test_build_overpass_query_for_named_city() -> None:
    q = build_overpass_query(city="Pécs", district=None, limit=100)
    assert '"name"="Pécs"' in q
    assert "out center 100;" in q
    # No district clause when district is None.
    assert 'admin_level"="10"' not in q


def test_parse_overpass_response_extracts_addresses() -> None:
    body = {
        "elements": [
            {
                "tags": {"addr:street": "Andrássy út", "addr:housenumber": "12"},
                "lat": 47.5,
                "lon": 19.06,
            },
            {
                "tags": {"addr:street": "Margit körút", "addr:housenumber": "5/A"},
                "center": {"lat": 47.51, "lon": 19.03},
            },
        ]
    }
    addresses = parse_overpass_response(city="Budapest", district="6", body=body)
    assert len(addresses) == 2
    assert addresses[0] == Address(city="Budapest", district="6", street="Andrássy út 12", lat=47.5, lng=19.06)


def test_parse_overpass_response_skips_incomplete_entries() -> None:
    body: dict[str, Any] = {
        "elements": [
            {"tags": {"addr:street": "Csak utca"}, "center": {"lat": 47.5, "lon": 19.0}},  # no housenumber
            {"tags": {"addr:housenumber": "9"}, "center": {"lat": 47.5, "lon": 19.0}},  # no street
            {
                "tags": {"addr:street": "Foo", "addr:housenumber": "1"},
                "center": {"lat": None, "lon": 19.0},
            },
            {"tags": {}, "center": {}},  # empty
        ]
    }
    assert parse_overpass_response(city="Pécs", district=None, body=body) == []


def test_parse_overpass_response_returns_empty_on_no_elements() -> None:
    assert parse_overpass_response(city="Szeged", district=None, body={}) == []


# ---------------------------------------------------------------------------
# fetch_region — retry behaviour
# ---------------------------------------------------------------------------


async def test_fetch_region_returns_addresses_on_success() -> None:
    client = _FakeClient(
        responses_by_region={
            ("Budapest", "13"): {
                "elements": [
                    {
                        "tags": {"addr:street": "Lehel utca", "addr:housenumber": "1"},
                        "center": {"lat": 47.51, "lon": 19.08},
                    }
                ]
            }
        }
    )
    addresses = await fetch_region(
        city="Budapest",
        district="13",
        target=10,
        client=cast(Any, client),
        retries=1,
        sleep=_no_sleep,
    )
    assert len(addresses) == 1
    assert len(client.calls) == 1


async def test_fetch_region_retries_on_5xx_then_succeeds() -> None:
    client = _FakeClient(
        responses_by_region={
            ("Budapest", "13"): {
                "elements": [
                    {
                        "tags": {"addr:street": "Lehel utca", "addr:housenumber": "1"},
                        "center": {"lat": 47.51, "lon": 19.08},
                    }
                ]
            }
        }
    )
    client.failures[("Budapest", "13")] = 2  # two 500s before the 200
    addresses = await fetch_region(
        city="Budapest",
        district="13",
        target=10,
        client=cast(Any, client),
        retries=3,
        sleep=_no_sleep,
    )
    assert len(addresses) == 1
    assert len(client.calls) == 3


async def test_fetch_region_raises_after_exhausted_retries() -> None:
    client = _FakeClient()
    client.failures[("Budapest", "13")] = 99  # always fail

    with pytest.raises(httpx.HTTPError):
        await fetch_region(
            city="Budapest",
            district="13",
            target=10,
            client=cast(Any, client),
            retries=2,
            sleep=_no_sleep,
        )
    assert len(client.calls) == 2


async def _no_sleep(_seconds: float) -> None:
    return None


# ---------------------------------------------------------------------------
# fetch_all_addresses + targets
# ---------------------------------------------------------------------------


def test_target_totals_sum_above_acceptance_bar() -> None:
    """Without any per-region drops, the requested totals must exceed 2 500."""
    total = sum(BUDAPEST_DISTRICT_TARGETS.values()) + sum(REGIONAL_CITY_TARGETS.values()) + sum(RURAL_TARGETS.values())
    assert total >= MIN_TOTAL_ADDRESSES


def test_target_distribution_matches_spec_outline() -> None:
    """Spec §17.2.1: BP 1-23 weighted by pop, 6 regional cities, ~200 smaller-town entries."""
    assert set(BUDAPEST_DISTRICT_TARGETS) == {str(i) for i in range(1, 24)}
    assert set(REGIONAL_CITY_TARGETS) == {"Pécs", "Szeged", "Debrecen", "Miskolc", "Győr", "Sopron"}
    rural_total = sum(RURAL_TARGETS.values())
    assert 150 <= rural_total <= 250  # "~200" leaves a ±50 window


async def test_fetch_all_addresses_writes_output_file(tmp_path: Path) -> None:
    # One address per region so we keep the test fast.
    payload_per_region: dict[tuple[str, str | None], dict[str, Any]] = {}
    for district in BUDAPEST_DISTRICT_TARGETS:
        payload_per_region[("Budapest", district)] = {
            "elements": [
                {
                    "tags": {"addr:street": "Teszt utca", "addr:housenumber": "1"},
                    "center": {"lat": 47.5, "lon": 19.0},
                }
            ]
        }
    for city in (*REGIONAL_CITY_TARGETS, *RURAL_TARGETS):
        payload_per_region[(city, None)] = {
            "elements": [
                {
                    "tags": {"addr:street": "Teszt utca", "addr:housenumber": "2"},
                    "center": {"lat": 47.0, "lon": 20.0},
                }
            ]
        }

    client = _FakeClient(responses_by_region=payload_per_region)
    output = tmp_path / "addresses.json"
    await fetch_all_addresses(client=cast(Any, client), output_path=output)

    data = json.loads(output.read_text(encoding="utf-8"))
    expected_regions = len(BUDAPEST_DISTRICT_TARGETS) + len(REGIONAL_CITY_TARGETS) + len(RURAL_TARGETS)
    assert len(data) == expected_regions  # one address per region from the fake
    assert all({"city", "street", "lat", "lng"} <= entry.keys() for entry in data)


async def test_fetch_all_addresses_isolates_failed_regions(tmp_path: Path) -> None:
    payload_per_region: dict[tuple[str, str | None], dict[str, Any]] = {}
    for district in BUDAPEST_DISTRICT_TARGETS:
        payload_per_region[("Budapest", district)] = {
            "elements": [
                {
                    "tags": {"addr:street": "Teszt utca", "addr:housenumber": "1"},
                    "center": {"lat": 47.5, "lon": 19.0},
                }
            ]
        }
    for city in (*REGIONAL_CITY_TARGETS, *RURAL_TARGETS):
        payload_per_region[(city, None)] = {
            "elements": [
                {
                    "tags": {"addr:street": "Teszt utca", "addr:housenumber": "2"},
                    "center": {"lat": 47.0, "lon": 20.0},
                }
            ]
        }

    client = _FakeClient(responses_by_region=payload_per_region)
    client.failures[("Budapest", "13")] = 99
    output = tmp_path / "addresses.json"
    await fetch_all_addresses(client=cast(Any, client), output_path=output)

    data = json.loads(output.read_text(encoding="utf-8"))
    expected_regions = len(BUDAPEST_DISTRICT_TARGETS) + len(REGIONAL_CITY_TARGETS) + len(RURAL_TARGETS)
    assert len(data) == expected_regions - 1


# ---------------------------------------------------------------------------
# validate_address_file
# ---------------------------------------------------------------------------


def test_validate_address_file_passes_above_threshold(tmp_path: Path) -> None:
    fixture = tmp_path / "addresses.json"
    fixture.write_text(json.dumps([{"x": i} for i in range(MIN_TOTAL_ADDRESSES + 10)]))
    ok, message = validate_address_file(path=fixture)
    assert ok
    assert str(MIN_TOTAL_ADDRESSES + 10) in message


def test_validate_address_file_fails_below_threshold(tmp_path: Path) -> None:
    fixture = tmp_path / "addresses.json"
    fixture.write_text(json.dumps([{"x": i} for i in range(MIN_TOTAL_ADDRESSES - 1)]))
    ok, message = validate_address_file(path=fixture)
    assert not ok
    assert "need" in message


def test_validate_address_file_missing_path(tmp_path: Path) -> None:
    ok, message = validate_address_file(path=tmp_path / "__definitely_not_here__.json")
    assert not ok
    assert "not found" in message


def test_validate_address_file_rejects_non_array(tmp_path: Path) -> None:
    fixture = tmp_path / "addresses.json"
    fixture.write_text('{"not": "a list"}')
    ok, message = validate_address_file(path=fixture)
    assert not ok
    assert "must be a JSON array" in message
