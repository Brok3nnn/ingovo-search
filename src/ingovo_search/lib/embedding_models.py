"""Embedding model identifiers.

Single source of truth. Swap providers / versions here, never at call sites.
v0 lock per the design spec §0: Cohere `embed-multilingual-v3.0` (1024 dim,
multilingual EN/HU). The §17 calibration may swap to bge-m3 or e5-large; that
decision lives in `embedding_benchmark.py` and updates this constant.
"""

EMBED_MULTILINGUAL: str = "embed-multilingual-v3.0"
