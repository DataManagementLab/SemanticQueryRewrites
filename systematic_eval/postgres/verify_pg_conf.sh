#!/bin/bash
# Verify that postgresql16.conf settings (notably random_page_cost) actually
# take effect inside the pg_lab container on the remote host (c06).
#
# Run LOCALLY from anywhere — the script:
#   1. scp's the latest pg_lab_setup.py + postgres/postgresql16.conf to c06
#   2. ssh's into c06 and starts the container with that conf
#   3. runs SELECT pg_reload_conf() + SHOW <setting> from inside the container
#   4. tears the container down again
#
# Usage: ./verify_pg_conf.sh [server] [remote_path] [port] [random_page_cost]
#   server:           ssh target (default: c06)
#   remote_path:      target dir on the remote (default: matches the pg configs in config/)
#   port:             host port to expose pg_lab on (default: 5432)
#   random_page_cost: value to inject, mirroring `execution.postgres_random_page_cost`
#                     in a config YAML. Omit to check the unset case, which must
#                     yield postgres' default of 4.

set -euo pipefail

SERVER="${1:-c06}"
REMOTE_PATH="${2:-/mnt/labstore/psiegler/c06_multi_query_comparison_pg/}"
PORT="${3:-5432}"
RPC="${4:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"  # systematic_eval/

CONF_LOCAL="$SCRIPT_DIR/postgresql16.conf"
SETUP_LOCAL="$SE_DIR/pg_lab_setup.py"

[ -f "$CONF_LOCAL" ]  || { echo "ERROR: missing $CONF_LOCAL"; exit 1; }
[ -f "$SETUP_LOCAL" ] || { echo "ERROR: missing $SETUP_LOCAL"; exit 1; }

echo "==> Syncing files to $SERVER:$REMOTE_PATH ..."
ssh "$SERVER" "mkdir -p ${REMOTE_PATH}postgres"
scp "$SETUP_LOCAL"  "$SERVER:${REMOTE_PATH}pg_lab_setup.py"
scp "$CONF_LOCAL"   "$SERVER:${REMOTE_PATH}postgres/postgresql16.conf"

echo "==> Running verification on $SERVER ..."
ssh "$SERVER" "PORT='$PORT' REMOTE_PATH='$REMOTE_PATH' RPC='$RPC' bash -s" <<'REMOTE'
set -euo pipefail

CONTAINER="pg_lab"
cd "$REMOTE_PATH"

# Activate the project venv if it exists (pg_lab_setup.py needs psycopg2).
if [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

cleanup() {
    echo
    echo "==> Tearing down pg_lab container ..."
    python3 pg_lab_setup.py --teardown || true
}
trap cleanup EXIT

SET_FLAGS=()
if [ -n "$RPC" ]; then
    SET_FLAGS=(--set "random_page_cost=$RPC")
fi

echo "==> Starting pg_lab container (port $PORT, random_page_cost=${RPC:-<unset, expect default>}) ..."
python3 pg_lab_setup.py --start --port "$PORT" --conf postgres/postgresql16.conf ${SET_FLAGS[@]+"${SET_FLAGS[@]}"}

echo
echo "==> Locating psql inside container ..."
PG_BIN="$(docker exec "$CONTAINER" bash -c '
    for p in /pg_lab/postgres-pglab*/build/bin /pg_lab/build/bin /pg_lab/*/bin /usr/lib/postgresql/*/bin /usr/local/pgsql/bin; do
        if [ -x "$p/psql" ]; then echo "$p"; exit 0; fi
    done
    find / -name psql -type f -executable 2>/dev/null | head -1 | xargs -r -n1 dirname
' | head -1)"
echo "    pg_bin=$PG_BIN"

PSQL_EXEC=(docker exec "$CONTAINER" "$PG_BIN/psql" -U postgres -d postgres -tAX)

echo
echo "==> SELECT pg_reload_conf();"
"${PSQL_EXEC[@]}" -c "SELECT pg_reload_conf();"

SETTINGS=(
    random_page_cost
    seq_page_cost
    shared_buffers
    effective_cache_size
    work_mem
    enable_bitmapscan
    max_parallel_workers_per_gather
)

echo
echo "==> Effective settings in the running cluster:"
printf '  %-35s %s\n' "SETTING" "VALUE"
printf '  %-35s %s\n' "-----------------------------------" "-----"
for s in "${SETTINGS[@]}"; do
    val="$("${PSQL_EXEC[@]}" -c "SHOW $s;" | tr -d '[:space:]')"
    printf '  %-35s %s\n' "$s" "$val"
done

# Expect what we injected; with nothing injected, expect postgres' own default.
EXPECTED="${RPC:-4}"
ACTUAL="$("${PSQL_EXEC[@]}" -c "SHOW random_page_cost;" | tr -d '[:space:]')"

echo
# Numeric compare: postgres normalizes e.g. "1.1" and "4" in its own way.
if awk -v a="$EXPECTED" -v b="$ACTUAL" 'BEGIN { exit !(a + 0 == b + 0) }'; then
    if [ -n "$RPC" ]; then
        echo "==> OK: random_page_cost = $ACTUAL (matches injected --set)"
    else
        echo "==> OK: random_page_cost = $ACTUAL (unset -> postgres default)"
    fi
else
    echo "==> MISMATCH: expected random_page_cost=$EXPECTED, server reports $ACTUAL"
    exit 1
fi
REMOTE
