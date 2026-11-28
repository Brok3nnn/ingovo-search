# Hungarian settlement dataset

`hungary_settlements_2025.json` is a generated, versioned runtime asset: 3,178 entries,
the 3,155 settlements of the KSH gazetteer and the 23 Budapest districts. The parser and
the location scoring use it to resolve place names, counties and distances. Do not edit
it by hand.

## Sources and licence

- **KSH** (Hungarian Central Statistical Office), *Magyarország helységnévtára*
  (Gazetteer of Hungary), `hnt_letoltes_2025.xlsx`, state date 2025-01-01:
  <https://www.ksh.hu/docs/helysegnevtar/hnt_letoltes_2025.xlsx>. Supplies the canonical
  settlement names, KSH identifiers, legal status, and county and district relationships.
- **GeoNames** Hungary daily dump, snapshot 2026-06-21:
  <https://download.geonames.org/export/dump/HU.zip>. Supplies the WGS84 point
  coordinates and GeoNames identifiers.

Both sources are distributed under the Creative Commons Attribution 4.0 licence
(CC BY 4.0, <https://creativecommons.org/licenses/by/4.0/>). Attribution and warranty
terms:

- KSH: <https://www.ksh.hu/copyright>
- GeoNames: <https://download.geonames.org/export/dump/readme.txt>

## How it is built

`scripts/build_hungary_settlements.py` joins coordinates to the KSH rows by canonical or
alternate name and county code; Budapest districts additionally use their three-digit
district code. It fails unless every KSH row receives a coordinate, all identifiers are
unique, and all 23 Budapest districts are present. With the two downloads above:

```bash
uv run python scripts/build_hungary_settlements.py --ksh-xlsx hnt_letoltes_2025.xlsx \
    --geonames-zip HU.zip --output src/ingovo_search/recommender/data/hungary_settlements_2025.json
```

Pinned input SHA-256 hashes (the build records them in the file's `metadata`):

- KSH XLSX: `d059a14883a27963daa234d9f3954eb68b4cbf967e796710a4bcdb4f03437991`
- GeoNames ZIP: `d0487bb74ac16b8607edc934fff76c30f48e36e4e0a280e0be285b3ef543ba3d`
