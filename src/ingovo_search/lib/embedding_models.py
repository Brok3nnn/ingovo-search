"""Embedding model identifiers.

Single source of truth. Swap providers / versions here, never at call sites.
v0 lock: Cohere `embed-multilingual-v3.0` (1024 dim, multilingual EN/HU).
"""

EMBED_MULTILINGUAL: str = "embed-multilingual-v3.0"
