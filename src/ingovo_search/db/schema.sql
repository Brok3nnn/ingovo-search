-- ingovo-search database schema (PostgreSQL 16 with PostGIS 3 and pgvector).
--
-- Idempotent: every statement is IF NOT EXISTS / OR REPLACE, so it is safe to run on
-- every start. `docker compose up` applies it on first boot (docker-entrypoint-initdb.d)
-- and `ingovo-search-seed` applies it again before loading data. The SQLAlchemy models
-- in `ingovo_search/db/models/` map these tables; tests/unit/db/test_schema.py keeps
-- the two in step.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS postgis;

-- The numeric room count of a free-text `rooms` value ("3", "2,5", "2 + 1"):
-- the first number, with the first comma read as a decimal point; NULL for
-- anything unusable or non-positive. IMMUTABLE, so it can back a generated column.
CREATE OR REPLACE FUNCTION ingovo_room_count(txt text)
RETURNS numeric LANGUAGE plpgsql IMMUTABLE AS $fn$
DECLARE
    extracted text;
    parsed numeric;
BEGIN
    IF txt IS NULL THEN RETURN NULL; END IF;
    extracted := substring(regexp_replace(txt, ',', '.') from '([0-9]+(?:[.][0-9]+)?)');
    IF extracted IS NULL THEN RETURN NULL; END IF;
    parsed := extracted::numeric;
    IF parsed > 0 AND parsed <= 1.7976931348623157e308 THEN
        RETURN parsed;
    END IF;
    RETURN NULL;
EXCEPTION WHEN others THEN
    RETURN NULL;
END
$fn$;

CREATE TABLE IF NOT EXISTS properties (
    id varchar PRIMARY KEY,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    title varchar NOT NULL,
    description text,
    title_translations jsonb,
    description_translations jsonb,
    price_huf bigint,
    currency char(3) NOT NULL DEFAULT 'HUF',
    transaction_type varchar NOT NULL DEFAULT 'sale',
    city varchar NOT NULL,
    district varchar,
    address varchar,
    property_type varchar,
    rooms varchar,
    -- Derived in the database so room filters run in SQL; read by recommender/sql_filter.py.
    rooms_count numeric GENERATED ALWAYS AS (ingovo_room_count(rooms)) STORED,
    area_m2 double precision,
    floor integer,
    total_floors integer,
    year_built integer,
    heating varchar,
    status varchar NOT NULL DEFAULT 'active',
    expires_at timestamptz,
    images varchar[] NOT NULL DEFAULT '{}',
    features jsonb,
    has_balcony boolean,
    has_terrace boolean,
    has_parking boolean,
    has_garage boolean,
    has_garden boolean,
    has_basement boolean,
    has_pool boolean,
    elevator boolean,
    air_conditioning boolean,
    renovation_state varchar,
    orientation varchar,
    noise_score double precision,
    light_quality_score double precision,
    red_flags jsonb,
    description_quality_score double precision,
    templated_phrase_fraction double precision,
    cleaned_description text,
    one_paragraph_summary text,
    short_summary text,
    view_count integer NOT NULL DEFAULT 0,
    save_count integer NOT NULL DEFAULT 0,
    contact_count integer NOT NULL DEFAULT 0,
    photo_count integer NOT NULL DEFAULT 0,
    source varchar,
    source_id varchar,
    quality_score double precision,
    last_enriched_at timestamptz,
    factual_embedding vector(1024),
    descriptive_embedding vector(1024),
    geo geography(POINT, 4326),
    CONSTRAINT uq_properties_source_source_id UNIQUE (source, source_id),
    CONSTRAINT ck_properties_price_huf_positive CHECK (price_huf IS NULL OR price_huf > 0),
    CONSTRAINT ck_properties_currency_huf CHECK (currency = 'HUF'),
    CONSTRAINT ck_properties_status_canonical
        CHECK (status IN ('draft', 'payment_pending', 'active', 'expired', 'archived'))
);

CREATE INDEX IF NOT EXISTS ix_properties_city ON properties (city);
CREATE INDEX IF NOT EXISTS ix_properties_district ON properties (district);
CREATE INDEX IF NOT EXISTS ix_properties_property_type ON properties (property_type);
CREATE INDEX IF NOT EXISTS ix_properties_price_huf ON properties (price_huf);
CREATE INDEX IF NOT EXISTS ix_properties_rooms_count ON properties (rooms_count) WHERE status = 'active';
CREATE INDEX IF NOT EXISTS ix_properties_geo_gist ON properties USING gist (geo);
-- Cosine-distance HNSW indexes for the kNN prefilter and the similarity terms.
CREATE INDEX IF NOT EXISTS ix_properties_factual_embedding_hnsw
    ON properties USING hnsw (factual_embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS ix_properties_descriptive_embedding_hnsw
    ON properties USING hnsw (descriptive_embedding vector_cosine_ops);

-- Buyer feedback on a listing, keyed by the query that produced it; pooled across
-- similar queries into the `pooled` ranking term (recommender/pooled_signal.py).
CREATE TABLE IF NOT EXISTS interactions (
    id varchar PRIMARY KEY,
    created_at timestamptz NOT NULL DEFAULT now(),
    user_id varchar,
    property_id varchar NOT NULL REFERENCES properties (id) ON DELETE CASCADE,
    query_text text,
    query_parsed jsonb,
    query_embedding vector(1024),
    signal_type varchar NOT NULL,
    signal_weight double precision NOT NULL DEFAULT 1.0
);
CREATE INDEX IF NOT EXISTS ix_interactions_property_id_created_at ON interactions (property_id, created_at);
CREATE INDEX IF NOT EXISTS ix_interactions_user_id_created_at ON interactions (user_id, created_at);
CREATE INDEX IF NOT EXISTS ix_interactions_query_embedding_hnsw
    ON interactions USING hnsw (query_embedding vector_cosine_ops);

-- One row per LLM call (parser, rerank, image analysis, region hint): tokens, cost,
-- latency, outcome. The budget guardrail (lib/budget_guard.py) sums the day's cost.
CREATE TABLE IF NOT EXISTS enrichment_log (
    id varchar PRIMARY KEY,
    created_at timestamptz NOT NULL DEFAULT now(),
    property_id varchar,
    model_used varchar NOT NULL,
    prompt_version varchar NOT NULL,
    input_tokens integer NOT NULL,
    output_tokens integer NOT NULL,
    cost_usd double precision NOT NULL,
    latency_ms integer NOT NULL,
    outcome varchar NOT NULL,
    cached_input_tokens integer
);
CREATE INDEX IF NOT EXISTS ix_enrichment_log_created_at ON enrichment_log (created_at);
CREATE INDEX IF NOT EXISTS ix_enrichment_log_property_id ON enrichment_log (property_id);

-- Named regions ("Balaton", "Budai hegyvidék") resolved to a point and a spread,
-- seeded by an LLM hint on first use (recommender/region_centroid.py).
CREATE TABLE IF NOT EXISTS region_centroid (
    region_name varchar PRIMARY KEY,
    centroid_geo geography(POINT, 4326),
    spread_km double precision,
    n_signals integer NOT NULL DEFAULT 0,
    last_updated timestamptz NOT NULL DEFAULT now(),
    llm_hint_districts jsonb
);
CREATE INDEX IF NOT EXISTS ix_region_centroid_centroid_geo_gist ON region_centroid USING gist (centroid_geo);

-- One row per search: the parsed query, its hash, candidate counts, latency, rerank outcome.
CREATE TABLE IF NOT EXISTS search_log (
    id varchar PRIMARY KEY,
    created_at timestamptz NOT NULL DEFAULT now(),
    user_id varchar,
    query_hash varchar NOT NULL,
    parsed_query jsonb,
    hard_filter_count integer NOT NULL,
    sql_candidate_count integer NOT NULL,
    vector_candidate_count integer,
    returned_count integer NOT NULL,
    cache_hit boolean NOT NULL DEFAULT false,
    latency_ms integer NOT NULL,
    score_version varchar NOT NULL,
    top_result_ids jsonb NOT NULL,
    rerank_backend varchar,
    rerank_outcome varchar
);
CREATE INDEX IF NOT EXISTS ix_search_log_created_at ON search_log (created_at);
CREATE INDEX IF NOT EXISTS ix_search_log_query_hash ON search_log (query_hash);

-- Written in AI_SEARCH_RERANK_BACKEND=shadow mode: both rerankers' orders for one head.
CREATE TABLE IF NOT EXISTS rerank_shadow_comparison (
    id varchar PRIMARY KEY,
    created_at timestamptz NOT NULL DEFAULT now(),
    query_hash varchar NOT NULL,
    semantic_query text,
    candidate_ids jsonb NOT NULL,
    served_backend varchar NOT NULL,
    gemini_order jsonb,
    jev_order jsonb,
    jev_scores jsonb,
    gemini_latency_ms integer,
    jev_latency_ms integer,
    jev_input_tokens integer,
    gemini_outcome varchar,
    jev_outcome varchar,
    jev_model varchar,
    prompt_version varchar
);
CREATE INDEX IF NOT EXISTS ix_rerank_shadow_comparison_created_at ON rerank_shadow_comparison (created_at);

-- Points of interest (OpenStreetMap) for the proximity term ("near a metro station").
CREATE TABLE IF NOT EXISTS poi (
    id varchar PRIMARY KEY,
    created_at timestamptz NOT NULL DEFAULT now(),
    category varchar NOT NULL,
    name varchar,
    geo geography(POINT, 4326) NOT NULL,
    source varchar NOT NULL DEFAULT 'osm',
    source_id varchar NOT NULL,
    CONSTRAINT uq_poi_source_source_id UNIQUE (source, source_id)
);
CREATE INDEX IF NOT EXISTS ix_poi_category ON poi (category);
CREATE INDEX IF NOT EXISTS ix_poi_geo_gist ON poi USING gist (geo);

-- Relevance ratings (LLM judge or human) of (query, listing) pairs, read by the
-- ranking evals and the weight calibration under eval/.
CREATE TABLE IF NOT EXISTS synthetic_eval_rating (
    id varchar PRIMARY KEY,
    created_at timestamptz NOT NULL DEFAULT now(),
    query_id varchar NOT NULL,
    property_id varchar NOT NULL REFERENCES properties (id) ON DELETE CASCADE,
    rater_id varchar NOT NULL,
    rating integer NOT NULL,
    rating_source varchar NOT NULL DEFAULT 'synthetic',
    inventory_source varchar NOT NULL DEFAULT 'synthetic',
    rank integer,
    location_score integer,
    price_score integer,
    size_score integer,
    preference_score integer,
    location_match boolean,
    human_match_percent integer,
    predicted_match_percent integer,
    score_version varchar,
    pipeline_versions jsonb,
    CONSTRAINT uq_synthetic_eval_rating_query_property_rater UNIQUE (query_id, property_id, rater_id)
);
CREATE INDEX IF NOT EXISTS ix_synthetic_eval_rating_query_id ON synthetic_eval_rating (query_id);

-- Durable cache of document embeddings, keyed by (model, input type, text hash).
CREATE TABLE IF NOT EXISTS embedding_cache (
    cache_key varchar PRIMARY KEY,
    text_hash varchar NOT NULL,
    model_used varchar NOT NULL,
    input_type varchar NOT NULL,
    embedding vector(1024) NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_embedding_cache_updated_at ON embedding_cache (updated_at DESC);
