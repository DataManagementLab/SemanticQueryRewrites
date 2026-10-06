#!/bin/bash
set -euo pipefail

# Onboard one learned_db dataset into the project.
#
# Usage: ./add_dataset.sh <full_scaled_name> [--count N] [--skip-build]
#   e.g. ./add_dataset.sh financial_scaled4
#
# Does, for the given dataset:
#   1. (remote) build <short>.duckdb from datasets/<full>/*.csv into the c06 workspace,
#      emit an authoritative schema sidecar, then full-copy the db into the c07 workspace;
#      scp the schema sidecar back to prompts/<short>/schema.txt.
#   2. (local) split workloads/<full>/workload_200k_s1.sql -> sql/<short>_200k/ (first N),
#      and the complex/many-joins workload -> sql/<short>_complex/ (first N).
#   3. (local) render prompts/<short>/ and config/experiment_<short>_oracle.yaml(+_complex).
#
# --skip-build reuses an already-built remote db (re-does only local artifacts + schema fetch).
#
# Servers c06 and c07 share /mnt, so we ssh once (c06) and cp into the c07 folder directly.

FULL=""
COUNT=1000
SKIP_BUILD=false
while [ $# -gt 0 ]; do
    case "$1" in
        --count) COUNT="$2"; shift 2 ;;
        --count=*) COUNT="${1#--count=}"; shift ;;
        --skip-build) SKIP_BUILD=true; shift ;;
        -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        -*) echo "unknown arg: $1" >&2; exit 1 ;;
        *) FULL="$1"; shift ;;
    esac
done
[ -n "$FULL" ] || { echo "Error: need <full_scaled_name> (e.g. financial_scaled4)" >&2; exit 1; }

SHORT="$(echo "$FULL" | sed -E 's/_scaled[0-9]+$//')"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"      # systematic_eval/dataset_setup
SE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"                          # systematic_eval
REPO_ROOT="$(cd "$SE_DIR/.." && pwd)"

SERVER="c06"
LAB="/mnt/labstore/SIGs/ML/learned_db"
C06_PATH="/mnt/labstore/psiegler/c06_multi_query_comparison"
C07_PATH="/mnt/labstore/psiegler/c07_multi_query_comparison"
CSV_DIR="$LAB/datasets/$FULL"
WL_DIR="$LAB/workloads/$FULL"
DB_NAME="${SHORT}.duckdb"
SCHEMA_REMOTE="$C06_PATH/${SHORT}_schema.txt"
SCHEMA_LOCAL="$(mktemp -t ${SHORT}_schema.XXXX.txt)"

echo "=== onboarding $FULL (short: $SHORT) ==="

# --- 1. remote build -------------------------------------------------------
if [ "$SKIP_BUILD" = "false" ]; then
    echo "[1/3] remote build on $SERVER -> $C06_PATH/$DB_NAME"
    ssh "$SERVER" "test -d '$CSV_DIR'" || { echo "Error: $CSV_DIR not found on $SERVER" >&2; exit 1; }
    scp -q "$SCRIPT_DIR/build_duckdb.py" "$SERVER:$C06_PATH/build_duckdb.py"
    ssh "$SERVER" bash -s <<EOF
set -euo pipefail
cd "$C06_PATH"
source .venv/bin/activate
python3 -c 'import duckdb' 2>/dev/null || pip install -q duckdb
python3 build_duckdb.py --csv-dir "$CSV_DIR" --db-file "$C06_PATH/$DB_NAME" --schema-out "$SCHEMA_REMOTE"
mkdir -p "$C07_PATH"
cp -f "$C06_PATH/$DB_NAME" "$C07_PATH/$DB_NAME"
ls -lh "$C06_PATH/$DB_NAME" "$C07_PATH/$DB_NAME"
deactivate
EOF
else
    echo "[1/3] --skip-build: reusing remote db, fetching schema only"
    ssh "$SERVER" "test -f '$SCHEMA_REMOTE'" || { echo "Error: $SCHEMA_REMOTE missing; run without --skip-build first" >&2; exit 1; }
fi
scp -q "$SERVER:$SCHEMA_REMOTE" "$SCHEMA_LOCAL"

# --- 2. local query sets ---------------------------------------------------
echo "[2/3] splitting workloads -> sql/${SHORT}_200k, sql/${SHORT}_complex (first $COUNT)"
ssh "$SERVER" "test -f '$WL_DIR/workload_200k_s1.sql'" \
    || { echo "Error: $WL_DIR/workload_200k_s1.sql not found" >&2; exit 1; }
ssh "$SERVER" "head -n $COUNT '$WL_DIR/workload_200k_s1.sql'" \
    | python3 "$SCRIPT_DIR/split_workload.py" --workload-file - --count "$COUNT" \
        --out-dir "$REPO_ROOT/sql/${SHORT}_200k"

# Prefer the many-joins workload; fall back to complex_workload_200k_s1.sql.
if ssh "$SERVER" "test -f '$WL_DIR/complex_workload_many_joins_hints.sql'"; then
    COMPLEX_SRC="complex_workload_many_joins_hints.sql"
else
    COMPLEX_SRC="complex_workload_200k_s1.sql"
fi
echo "       complex source: $COMPLEX_SRC"
ssh "$SERVER" "head -n $COUNT '$WL_DIR/$COMPLEX_SRC'" \
    | python3 "$SCRIPT_DIR/split_workload.py" --workload-file - --count "$COUNT" \
        --out-dir "$REPO_ROOT/sql/${SHORT}_complex"

# --- 3. local scaffold (prompts + config) ----------------------------------
echo "[3/3] rendering prompts + configs"
cd "$REPO_ROOT"
python3 -m systematic_eval.dataset_setup.render_prompts --short "$SHORT" --schema-file "$SCHEMA_LOCAL"
python3 -m systematic_eval.dataset_setup.make_config --short "$SHORT"
rm -f "$SCHEMA_LOCAL"

echo "=== done: $SHORT ==="
echo "  sql/${SHORT}_200k:      $(ls "$REPO_ROOT/sql/${SHORT}_200k"/*.sql 2>/dev/null | wc -l) files"
echo "  sql/${SHORT}_complex:   $(ls "$REPO_ROOT/sql/${SHORT}_complex"/*.sql 2>/dev/null | wc -l) files"
echo "  prompts/${SHORT}/       (schema.txt + generation/prompt_wk01.txt + system + refinement + example)"
echo "  config/experiment_${SHORT}_oracle.yaml (+ _complex_oracle.yaml)"
echo "  remote db: $C06_PATH/$DB_NAME  and  $C07_PATH/$DB_NAME"
