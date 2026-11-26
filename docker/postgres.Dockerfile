# PostgreSQL 16 with pgvector and PostGIS.
# The pgvector image ships pgvector but not PostGIS, so PostGIS is installed from apt.
FROM pgvector/pgvector:0.8.0-pg16

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       postgresql-16-postgis-3 \
       postgresql-16-postgis-3-scripts \
    && rm -rf /var/lib/apt/lists/*
