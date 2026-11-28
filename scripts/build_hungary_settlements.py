"""Build `src/ingovo_search/recommender/data/hungary_settlements_2025.json` from public sources.

Inputs, both under CC BY 4.0:

- the KSH gazetteer (Magyarország helységnévtára), state date 2025-01-01:
  https://www.ksh.hu/docs/helysegnevtar/hnt_letoltes_2025.xlsx
- the GeoNames Hungary dump: https://download.geonames.org/export/dump/HU.zip

Every KSH row (3,155 settlements and the 23 Budapest districts) gets the GeoNames
point whose name and county match best. The build fails unless every row gets a
coordinate, the ids are unique and all 23 districts are present, and the output
records the SHA-256 of both inputs.

    uv run python scripts/build_hungary_settlements.py --ksh-xlsx hnt_letoltes_2025.xlsx \
        --geonames-zip HU.zip --output src/ingovo_search/recommender/data/hungary_settlements_2025.json

Standard library only: the XLSX is read as a zip of XML parts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

_XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_KSH_SOURCE_URL = "https://www.ksh.hu/docs/helysegnevtar/hnt_letoltes_2025.xlsx"
_GEONAMES_SOURCE_URL = "https://download.geonames.org/export/dump/HU.zip"
_KSH_AS_OF = "2025-01-01"
# The snapshot date of the HU.zip the bundled file was built from; update it with a new download.
_GEONAMES_AS_OF = "2026-06-21"

_COUNTY_ADMIN1 = {
    "Bács-Kiskun": "01",
    "Baranya": "02",
    "Békés": "03",
    "Borsod-Abaúj-Zemplén": "04",
    "főváros": "05",
    "Csongrád-Csanád": "06",
    "Fejér": "08",
    "Győr-Moson-Sopron": "09",
    "Hajdú-Bihar": "10",
    "Heves": "11",
    "Komárom-Eszgom": "12",
    "Komárom-Esztergom": "12",
    "Nógrád": "14",
    "Pest": "16",
    "Somogy": "17",
    "Szabolcs-Szatmár-Bereg": "18",
    "Jász-Nagykun-Szolnok": "20",
    "Tolna": "21",
    "Vas": "22",
    "Veszprém": "23",
    "Zala": "24",
}


def _fold(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _column_index(reference: str) -> int:
    match = re.match(r"[A-Z]+", reference)
    if match is None:
        raise ValueError(f"invalid XLSX cell reference: {reference}")
    value = 0
    for char in match.group(0):
        value = value * 26 + ord(char) - ord("A") + 1
    return value - 1


def _load_ksh_rows(path: Path) -> list[dict[str, str]]:
    # The input is the official KSH download the operator chose (its SHA-256 is pinned
    # in the data README), and the standard-library expat parser does not resolve
    # external entities.
    with zipfile.ZipFile(path) as archive:
        shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))  # noqa: S314
        shared = ["".join(node.text or "" for node in item.iter(f"{_XLSX_NS}t")) for item in shared_root]
        sheet_root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))  # noqa: S314

    rows: list[list[str | None]] = []
    for row in sheet_root.iter(f"{_XLSX_NS}row"):
        values: list[str | None] = [None] * 9
        for cell in row.findall(f"{_XLSX_NS}c"):
            index = _column_index(cell.attrib["r"])
            if index >= len(values):
                continue
            value_node = cell.find(f"{_XLSX_NS}v")
            if value_node is None or value_node.text is None:
                continue
            values[index] = shared[int(value_node.text)] if cell.attrib.get("t") == "s" else value_node.text
        rows.append(values)

    return [
        {
            "name": row[0] or "",
            "ksh_id": row[1] or "",
            "status": row[2] or "",
            "county": row[3] or "",
            "district_id": (row[4] or "").replace(" ", ""),
            "district_name": row[5] or "",
            "district_seat": row[6] or "",
        }
        for row in rows[3:]
        if row[0] and row[1]
    ]


def _load_geonames_rows(path: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        lines = archive.read("HU.txt").decode("utf-8").splitlines()

    rows: list[dict[str, Any]] = []
    for line in lines:
        fields = line.split("\t")
        feature_class = fields[6]
        feature_code = fields[7]
        is_current_populated_place = feature_class == "P" and feature_code not in {"PPLH", "PPLQ", "PPLW"}
        is_budapest_district = feature_class == "A" and feature_code == "ADM2" and fields[10] == "05"
        if not (is_current_populated_place or is_budapest_district):
            continue
        rows.append(
            {
                "geonames_id": int(fields[0]),
                "name": fields[1],
                "ascii_name": fields[2],
                "alternate_names": fields[3].split(",") if fields[3] else [],
                "latitude": float(fields[4]),
                "longitude": float(fields[5]),
                "feature_class": feature_class,
                "feature_code": feature_code,
                "admin1": fields[10],
                "admin2": fields[11],
                "population": int(fields[14] or 0),
                "modified_at": fields[18],
            }
        )
    return rows


def _candidate_index(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        names = {row["name"], row["ascii_name"], *row["alternate_names"]}
        for name in names:
            index[_fold(str(name))].append(row)
    return index


def _district_number(name: str) -> int | None:
    match = re.search(r"Budapest\s+(\d{2})\.\s*ker", name)
    return int(match.group(1)) if match is not None else None


def _score_candidate(ksh: dict[str, str], candidate: dict[str, Any]) -> int:
    expected_admin1 = _COUNTY_ADMIN1.get(ksh["county"])
    exact_primary = _fold(str(candidate["name"])) == _fold(ksh["name"])
    exact_spelling = str(candidate["name"]).casefold() == ksh["name"].casefold()
    alternate_exact = any(str(alias).casefold() == ksh["name"].casefold() for alias in candidate["alternate_names"])
    score = 100 if exact_primary else 0
    score += 60 if candidate["admin1"] == expected_admin1 else 0
    score += 20 if exact_spelling else 0
    score += 8 if candidate["feature_code"] in {"PPLC", "PPLA", "PPLA2", "PPLX", "PPL"} else 0
    score += 4 if candidate["population"] > 0 else 0
    score += 2 if alternate_exact else 0
    return score


def _resolve_coordinate(
    ksh: dict[str, str],
    index: dict[str, list[dict[str, Any]]],
    geonames_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    district_number = _district_number(ksh["name"])
    if district_number is not None:
        admin2 = f"{district_number:03d}"
        district_candidates = [
            row
            for row in geonames_rows
            if row["admin1"] == "05" and row["admin2"] == admin2 and row["feature_code"] in {"PPLX", "ADM2"}
        ]
        if district_candidates:
            district_candidates.sort(
                key=lambda row: (row["feature_code"] != "PPLX", -row["population"], row["geonames_id"])
            )
            return district_candidates[0]

    candidates_by_id = {int(row["geonames_id"]): row for row in index.get(_fold(ksh["name"]), [])}
    if not candidates_by_id:
        raise ValueError(f"no GeoNames coordinate candidate for {ksh['name']} ({ksh['ksh_id']})")
    ranked = sorted(
        candidates_by_id.values(),
        key=lambda row: (
            -_score_candidate(ksh, row),
            -row["population"],
            -int(row["modified_at"].replace("-", "")),
            row["geonames_id"],
        ),
    )
    # GeoNames occasionally contains multiple point features with the same
    # primary name inside one county. The sort remains deterministic: prefer
    # populated and more recently verified records, then the stable GeoNames id.
    return ranked[0]


def build_dataset(ksh_xlsx: Path, geonames_zip: Path) -> dict[str, Any]:
    ksh_rows = _load_ksh_rows(ksh_xlsx)
    geonames_rows = _load_geonames_rows(geonames_zip)
    index = _candidate_index(geonames_rows)
    settlements: list[dict[str, Any]] = []

    for row in ksh_rows:
        coordinate = _resolve_coordinate(row, index, geonames_rows)
        district_number = _district_number(row["name"])
        county = "Budapest" if row["county"] == "főváros" or row["name"] == "Budapest" else row["county"]
        settlements.append(
            {
                "id": f"ksh:{row['ksh_id']}",
                "name": row["name"],
                "kind": "budapest_district" if district_number is not None else "municipality",
                "legal_status": row["status"],
                "county": county,
                "district_id": row["district_id"] or None,
                "district_name": row["district_name"] or None,
                "district_seat": row["district_seat"] or None,
                "parent_id": "ksh:13578" if district_number is not None else None,
                "budapest_district": district_number,
                "latitude": coordinate["latitude"],
                "longitude": coordinate["longitude"],
                "geonames_id": coordinate["geonames_id"],
            }
        )

    ids = [row["id"] for row in settlements]
    if len(ksh_rows) != 3178:
        raise ValueError(f"expected 3178 KSH rows, got {len(ksh_rows)}")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate KSH stable IDs in generated dataset")
    if sum(row["kind"] == "budapest_district" for row in settlements) != 23:
        raise ValueError("generated dataset must contain all 23 Budapest districts")
    if any(row["latitude"] is None or row["longitude"] is None for row in settlements):
        raise ValueError("generated dataset contains a settlement without coordinates")

    settlements.sort(key=lambda row: row["id"])
    return {
        "metadata": {
            "schema_version": "hungary-settlements-v1",
            "ksh_as_of": _KSH_AS_OF,
            "geonames_as_of": _GEONAMES_AS_OF,
            "ksh_source_url": _KSH_SOURCE_URL,
            "geonames_source_url": _GEONAMES_SOURCE_URL,
            "license": "CC BY 4.0",
            "ksh_sha256": _sha256(ksh_xlsx),
            "geonames_sha256": _sha256(geonames_zip),
            "settlement_count": len(settlements),
        },
        "settlements": settlements,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ksh-xlsx", type=Path, required=True)
    parser.add_argument("--geonames-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = build_dataset(args.ksh_xlsx, args.geonames_zip)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dataset, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {dataset['metadata']['settlement_count']} settlements to {args.output}")  # noqa: T201


if __name__ == "__main__":
    main()
