#!/bin/bash
# Transfer duckdb_where_order.py to c06 and run it inside the existing venv.
# scp the file, ssh + activate venv, run.

set -euo pipefail

SERVER="${SERVER:-c06}"
REMOTE_PATH="${REMOTE_PATH:-/mnt/labstore/psiegler/single_query_comparison/}"
SCRIPT_NAME="duckdb_where_order.py"
DB_FILE="${DB_FILE:-imdb.duckdb}"
ATTEMPTS="${ATTEMPTS:-5}"

HERE="$(cd "$(dirname "$0")" && pwd)"

scp "$HERE/$SCRIPT_NAME" "$SERVER:$REMOTE_PATH"

ssh "$SERVER" << EOF
    set -e
    cd "$REMOTE_PATH"
    source .venv/bin/activate
    python3 $SCRIPT_NAME $DB_FILE $ATTEMPTS
    deactivate
EOF
