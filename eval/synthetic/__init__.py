"""§17.2 Stage 1 synthetic dataset generation.

Sub-components:

- §17.2.1 `extract_osm` — OSM Overpass API → 2 500+ real addresses across HU.
- §17.2.2 `price_sampler` — KSH price-per-m² distributions; samples plausible
  listing prices given district + area + property type.
- §17.2.3 `describe` — three quality tiers of HU descriptions (great /
  templated / skeletal), the top tier written by Gemini.
- §17.2.4 `assemble` + `validate_distribution` — combine §17.2.1-§17.2.3 into
  a 1 500-2 000 row synthetic dataset consumable by the seed command or the
  Postgres test DB.

All sub-modules are pure-Python helpers; the assembler is the only
network-touching CLI entrypoint and even that is opt-in via a flag.
"""
