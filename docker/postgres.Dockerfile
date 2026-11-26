# PostgreSQL 16 with pgvector and PostGIS.
# The pgvector image ships pgvector but not PostGIS, so PostGIS is installed from apt.
# postgresql-16 is upgraded in the same step: the PostGIS package from the PostgreSQL
# apt repository is built against the newest 16.x minor release and fails to load
# ("undefined symbol") on an older server.
FROM pgvector/pgvector:0.8.0-pg16

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       postgresql-16 \
       postgresql-16-postgis-3 \
       postgresql-16-postgis-3-scripts \
    && rm -rf /var/lib/apt/lists/*
