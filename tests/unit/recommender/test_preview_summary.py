"""The strict-constraint summary of the homepage preview, section by section, in both locales."""

import pytest

from ingovo_search.recommender.parser_schema import Anchors, GeoHints, HardFilters, ParsedQuery
from ingovo_search.recommender.preview_models import PreviewLocale
from ingovo_search.recommender.preview_summary import constraint_summary, constraint_summary_with_keys

_BEYOND_SAFE_INTEGER = 2**53 + 7


@pytest.mark.parametrize(
    ("parsed", "locale", "summary", "keys"),
    [
        pytest.param(
            ParsedQuery(
                hard_filters=HardFilters(
                    transaction_type="sale",
                    property_type_in=["Lakás", "Ház"],
                    city="Budapest",
                    district_in=["2", "11"],
                    rooms=3,
                    area_m2_min=50.5,
                    area_m2_max=80,
                    price_huf_max=85_000_000,
                    property_type_exclude=["Telek"],
                ),
                geo_hints=GeoHints(side_of_river="Buda"),
            ),
            "hu",
            "Eladó · Lakás, Ház · Budapest · 2. kerület, 11. kerület · 3 szoba · 50,5-80 m² · max. 85 M Ft · nem Telek",
            [
                "transaction:sale",
                "property_type:Lakás,Ház",
                "city:Budapest",
                "district:2,11",
                "rooms:3",
                "area:50.5:80.0",
                "price:max:85000000",
                "excluded_type:Telek",
            ],
            id="every-section-hu-districts-win-over-river-side",
        ),
        pytest.param(
            ParsedQuery(
                hard_filters=HardFilters(
                    transaction_type="rent",
                    property_type_in=["Lakás", "Iroda"],
                    rooms_min=2,
                    area_m2_min=40,
                    property_type_exclude=["Garázs"],
                ),
                anchors=Anchors(price_huf=350_000),
                geo_hints=GeoHints(side_of_river="Pest", region_name="Belváros"),
            ),
            "en",
            "Rent · Apartment, Office · Pest · min. 2 bedrooms · min. 40 m² · around HUF 350 000 · excluding Garázs",
            [
                "transaction:rent",
                "property_type:Lakás,Iroda",
                "river_side:Pest",
                "rooms_min:2",
                "area:40.0:",
                "price:anchor:350000",
                "excluded_type:Garázs",
            ],
            id="en-river-side-wins-over-region",
        ),
        pytest.param(
            ParsedQuery(
                hard_filters=HardFilters(area_m2_max=65.25, price_huf_max=_BEYOND_SAFE_INTEGER),
                anchors=Anchors(rooms=4),
                geo_hints=GeoHints(region_name="Balaton-felvidék"),
            ),
            "hu",
            "Balaton-felvidék · kb. 4 szoba · max. 65,25 m² · max. 9 007 199 254 740 999 Ft",
            ["region:Balaton-felvidék", "rooms_anchor:4", "area::65.25", "price:max:9007199254740999"],
            id="hu-region-anchor-rooms-bigint-price",
        ),
        pytest.param(
            ParsedQuery(
                hard_filters=HardFilters(transaction_type="sale", area_m2_max=90),
                anchors=Anchors(rooms=2, price_huf=64_500_000),
            ),
            "en",
            "Sale · around 2 bedrooms · max. 90 m² · around HUF 64.5M",
            ["transaction:sale", "rooms_anchor:2", "area::90.0", "price:anchor:64500000"],
            id="en-anchors",
        ),
        pytest.param(
            ParsedQuery(anchors=Anchors(price_huf=999_999)),
            "hu",
            "kb. 999 999 Ft",
            ["price:anchor:999999"],
            id="hu-price-below-a-million",
        ),
        pytest.param(
            ParsedQuery(hard_filters=HardFilters(price_huf_max=_BEYOND_SAFE_INTEGER)),
            "en",
            "max. HUF 9 007 199 254 740 999",
            ["price:max:9007199254740999"],
            id="en-bigint-price",
        ),
        pytest.param(ParsedQuery(), "hu", "A megadott keresési feltételek", [], id="hu-nothing-recognised"),
        pytest.param(ParsedQuery(), "en", "Your stated search criteria", [], id="en-nothing-recognised"),
    ],
)
def test_constraint_summary_reads_each_section_in_order(
    parsed: ParsedQuery, locale: PreviewLocale, summary: str, keys: list[str]
) -> None:
    assert constraint_summary_with_keys(parsed, locale) == (summary, keys)
    assert constraint_summary(parsed, locale) == summary
