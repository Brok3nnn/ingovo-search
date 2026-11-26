"""The canonical ``properties.status`` vocabulary.

Five values, fenced by the ``ck_properties_status_canonical`` CHECK in
``db/schema.sql``. Only ``active`` listings that have not expired are searchable.
"""

from __future__ import annotations

from typing import Final

DRAFT: Final = "draft"
PAYMENT_PENDING: Final = "payment_pending"
ACTIVE: Final = "active"
EXPIRED: Final = "expired"
ARCHIVED: Final = "archived"

CANONICAL_STATUSES: Final[tuple[str, ...]] = (DRAFT, PAYMENT_PENDING, ACTIVE, EXPIRED, ARCHIVED)

# The public visibility predicate shared by every candidate SQL statement. The
# expiry is checked at read time: a periodic status sweep would only be cleanup,
# never the authorization boundary for an already-expired listing.
PUBLIC_VISIBILITY_SQL: Final = f"status = '{ACTIVE}' AND (expires_at IS NULL OR expires_at > now())"
