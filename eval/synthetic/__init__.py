"""Synthetic dataset generation.

Sub-components:

- `extract_osm` — OSM Overpass API → 2 500+ real addresses across HU.
- `price_sampler` — KSH price-per-m² distributions; samples plausible
  listing prices given district + area + property type.
- `describe` — three quality tiers of HU descriptions (great /
  templated / skeletal), all local templates.
- `assemble` + `validate_distribution` — combine the three into
  a 1 500-2 000 row synthetic dataset consumable by the seed command or the
  Postgres test DB.

All sub-modules are pure-Python helpers; only `extract_osm` touches the
network (the Overpass fetch), and only when its CLI runs.
"""
