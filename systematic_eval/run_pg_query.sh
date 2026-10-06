#!/usr/bin/env bash
# Run the SQL below against the pg_lab container's psql.
# Assumes the pg_lab container is already up (port 5432, db imdb).
# Swap the SQL between the BEGIN/END markers to test another query.

set -euo pipefail

CONTAINER="${CONTAINER:-pg_lab}"
DB="${DB:-imdb}"
USER="${PGUSER:-postgres}"

PG_BIN=$(docker exec "$CONTAINER" bash -lc '
  for p in /pg_lab/postgres-pglab*/build/bin \
           /pg_lab/build/bin \
           /pg_lab/*/bin \
           /usr/lib/postgresql/*/bin \
           /usr/local/pgsql/bin; do
    if [ -x "$p/psql" ]; then echo "$p"; exit 0; fi
  done
  find / -name psql -type f -executable 2>/dev/null | head -1 | xargs -r -n1 dirname
')

if [ -z "$PG_BIN" ]; then
  echo "ERROR: could not locate psql inside container '$CONTAINER'." >&2
  exit 1
fi
PSQL_BIN="$PG_BIN/psql"
echo "Using $PSQL_BIN"

read -r -d '' SQL <<'SQL_END' || true
-- ===== BEGIN SQL (edit this block) =====
\timing on
EXPLAIN (ANALYZE, BUFFERS)
SELECT MIN(mi.info) AS release_date,
       MIN(t.title) AS modern_american_internet_movie
FROM aka_title AS at,
     company_name AS cn,
     company_type AS ct,
     info_type AS it1,
     keyword AS k,
     movie_companies AS mc,
     movie_info AS mi,
     movie_keyword AS mk,
     title AS t
WHERE cn.country_code = '[us]'
  AND it1.info = 'release dates'
  AND mi.note like '%internet%'
  AND mi.info is not NULL
  AND (mi.info like 'USA:% 199%' or mi.info like 'USA:% 200%')
  AND t.production_year > 1990
  AND t.id = at.movie_id
  AND t.id = mi.movie_id
  AND t.id = mk.movie_id
  AND t.id = mc.movie_id
  AND mk.movie_id = mi.movie_id
  AND mk.movie_id = mc.movie_id
  AND mk.movie_id = at.movie_id
  AND mi.movie_id = mc.movie_id
  AND mi.movie_id = at.movie_id
  AND mc.movie_id = at.movie_id
  AND k.id = mk.keyword_id
  AND it1.id = mi.info_type_id
  AND cn.id = mc.company_id
  AND ct.id = mc.company_type_id;
-- ===== END SQL =====
SQL_END

printf '%s\n' "$SQL" | docker exec -i "$CONTAINER" "$PSQL_BIN" -U "$USER" -d "$DB"
