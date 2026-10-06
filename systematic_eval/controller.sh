#!/bin/bash
set -euo pipefail

# Stage orchestration: runs the four pipeline stages locally and invokes the
# execution backend (stages/execution.py) on the lab server over SSH.
#
# Usage: ./controller.sh <experiment> [--version <label>] [--stage N]
# Example: ./controller.sh experiment_T_imdb_job_12_oracle_c07_4-2 --version v01
#
# <experiment> is the config file's base name and is the experiment's whole identity
# (the YAML carries no name field). It determines:
#   - Config file:      config/<experiment>.yaml
#   - Transfer data:    transfer_data/<experiment>[__<version>]/
#   - Saved results:    saved_results/<experiment>[__<version>]/
#
# When --version is supplied, per-run folders are suffixed with __<version>
# so multiple variants of the same experiment can coexist locally. The remote
# workspace stays shared (parallel runs of different versions are out of scope).

usage() {
    echo "Usage: $0 <experiment> [--version <label>] [--stage N | --baseline] [--no-plan-comparisons] [--full-plan-comparisons]"
    echo ""
    echo "  <experiment>       Config file base name, i.e. config/<experiment>.yaml."
    echo ""
    echo "  --version <label>  Optional free-text run label (alnum, '.', '-', '_'). When set,"
    echo "                     transfer_data/ and saved_results/ folders are suffixed __<label>."
    echo "  --stage 1   (default) Generation + initial base-table validation, then stages 2-4."
    echo "  --stage 2   Start at the refinement loop. Requires transfer.json + result.json in transfer_data/."
    echo "  --stage 3   Start at aggregation + the final timed execution. Requires final refine outputs (or transfer.json+result.json if refinement disabled)."
    echo "  --stage 4   Statistics only. Requires rule_summary_result.json. No remote work."
    echo "              (Resuming starts at a boundary where rule validity is already known.)"
    echo "  --baseline  Post-run backfill: measure original runtimes of the no-rule queries for the"
    echo "              runtime-weighted whole-workload metric, then re-run stats. Requires an existing"
    echo "              rule_summary_result.json; does NOT recompute it. Runs on remote (starts a container)."
    echo "  --no-plan-comparisons  Stats stage: skip costly per-query plan comparisons (keep runtime stats/plots/CSV)."
    echo "  --full-plan-comparisons  Stats stage: plot every rule subset. By default a query with more than"
    echo "                     50 subsets is reduced to its oracle pick and optimizer pick (2 plots)."
    echo ""
    echo "Example: $0 experiment_T_imdb_job_12_oracle_c07_4-2 --version v01 --stage 3"
}

if [ $# -lt 1 ]; then
    usage
    exit 1
fi

EXPERIMENT_NAME="$1"
shift

START_STAGE=1
VERSION=""
STATS_EXTRA_ARGS=""
BASELINE_ONLY=false
while [ $# -gt 0 ]; do
    case "$1" in
        --baseline)
            BASELINE_ONLY=true
            shift
            ;;
        --no-plan-comparisons)
            STATS_EXTRA_ARGS="$STATS_EXTRA_ARGS --no-plan-comparisons"
            shift
            ;;
        --full-plan-comparisons)
            STATS_EXTRA_ARGS="$STATS_EXTRA_ARGS --full-plan-comparisons"
            shift
            ;;
        --stage)
            START_STAGE="$2"
            shift 2
            ;;
        --stage=*)
            START_STAGE="${1#--stage=}"
            shift
            ;;
        --version)
            VERSION="$2"
            shift 2
            ;;
        --version=*)
            VERSION="${1#--version=}"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

case "$START_STAGE" in
    1|2|3|4) ;;
    *)
        echo "Error: --stage must be 1, 2, 3, or 4 (got: $START_STAGE)"
        exit 1
        ;;
esac

DIR_SUFFIX=""
if [ -n "$VERSION" ]; then
    if ! [[ "$VERSION" =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "Error: --version must match [A-Za-z0-9._-]+ (got: '$VERSION')"
        exit 1
    fi
    DIR_SUFFIX="__${VERSION}"
fi

if [ "$BASELINE_ONLY" = "true" ]; then
    START_MSG="baseline backfill (no-rule query runtimes) + stats"
else
    START_MSG="stage $START_STAGE"
fi
if [ -n "$VERSION" ]; then
    echo "Starting pipeline at $START_MSG for experiment '$EXPERIMENT_NAME' (version '$VERSION')"
else
    echo "Starting pipeline at $START_MSG for experiment '$EXPERIMENT_NAME'"
fi

# Resolve directories: script lives in systematic_eval/, repo root is one level up
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Derived paths
CONFIG="$SCRIPT_DIR/config/${EXPERIMENT_NAME}.yaml"
TRANSFER_DIR="$SCRIPT_DIR/transfer_data/${EXPERIMENT_NAME}${DIR_SUFFIX}"
RESULTS_DIR="$SCRIPT_DIR/saved_results/${EXPERIMENT_NAME}${DIR_SUFFIX}"

if [ ! -f "$CONFIG" ]; then
    echo "Error: config file not found: $CONFIG"
    exit 1
fi

# Create per-experiment directories
mkdir -p "$TRANSFER_DIR"
mkdir -p "$RESULTS_DIR"

# --baseline is a post-run backfill: it needs an existing final-execution result to
# know which queries already have runtimes (only the *rest* are measured). It skips
# stages 1-3 but still needs a remote container, so it reuses the stage<=3 remote path.
if [ "$BASELINE_ONLY" = "true" ]; then
    if [ ! -f "$TRANSFER_DIR/rule_summary_result.json" ]; then
        echo "Error: --baseline requires $TRANSFER_DIR/rule_summary_result.json from a prior run"
        exit 1
    fi
    START_STAGE=4  # skip generate/refine/aggregate/final-exec; remote forced on below
fi

# Parse config values from YAML
read_config() {
    python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print($1)"
}

SERVER=$(read_config "c['remote']['server']")
REMOTE_PATH=$(read_config "c['remote']['path']")
DB_FILE=$(read_config "c['dataset']['db_file']")
THRESHOLD=$(read_config "c['execution']['performance_threshold']")
ATTEMPTS=$(read_config "c['execution']['attempts']")
REFINE_ENABLED=$(read_config "c.get('refinement',{}).get('enabled', True)")
REFINE_ITERATIONS=$(read_config "c.get('refinement',{}).get('iterations', 1)")
VALIDATION_MODE=$(read_config "c.get('execution',{}).get('validation_mode', 'query')")
COST_ESTIMATION=$(read_config "c.get('execution',{}).get('cost_estimation', False)")
COST_SAVE_THRESHOLD=$(read_config "c.get('execution',{}).get('cost_save_threshold', 0.0)")
STATS_MODE=$(read_config "c.get('statistics',{}).get('mode', 'optimizer')")
ENGINE=$(read_config "c.get('execution',{}).get('engine','duckdb')")
UMBRA_PORT=$(read_config "c.get('execution',{}).get('umbra_port', 5432)")
UMBRA_MEMORY_GB=$(read_config "c.get('execution',{}).get('umbra_memory_gb', 0.0)")
DRIVER_MEMORY_GB=$(read_config "c.get('execution',{}).get('driver_memory_gb', 0.0)")
POSTGRES_HOST=$(read_config "c.get('execution',{}).get('postgres_host', '127.0.0.1')")
POSTGRES_PORT=$(read_config "c.get('execution',{}).get('postgres_port', 5432)")
POSTGRES_USER=$(read_config "c.get('execution',{}).get('postgres_user', 'postgres')")
POSTGRES_PASSWORD=$(read_config "c.get('execution',{}).get('postgres_password', 'postgres')")
POSTGRES_DBNAME=$(read_config "c.get('execution',{}).get('postgres_dbname', 'imdb')")
# Unset -> empty -> no ALTER SYSTEM SET -> postgres' own default (4.0).
POSTGRES_RANDOM_PAGE_COST=$(read_config "c.get('execution',{}).get('postgres_random_page_cost') or ''")
FIX_JOIN_ORDER=$(read_config "c.get('execution',{}).get('fix_join_order', False)")
COST_MODEL=$(read_config "c.get('execution',{}).get('cost_model', 'planner')")
ZS_MODEL_TYPE=$(read_config "c.get('execution',{}).get('zeroshot_model_type','') or ''")
ZS_MODEL_DIR=$(read_config "c.get('execution',{}).get('zeroshot_model_dir','') or ''")
ZS_SEED=$(read_config "c.get('execution',{}).get('zeroshot_seed', 9)")
ZS_STATS_FILE=$(read_config "c.get('execution',{}).get('zeroshot_statistics_file','') or ''")
ZS_DB_STATS=$(read_config "c.get('execution',{}).get('zeroshot_database_stats','') or ''")
MAX_SUBSETS=$(read_config "c.get('execution',{}).get('max_subsets', 4096)")
WARMUP_MODE=$(read_config "c.get('execution',{}).get('warmup_mode', 'per_query')")
PRIME_OS_CACHE=$(read_config "c.get('execution',{}).get('prime_os_cache', True)")
MAX_TEMP_SIZE=$(read_config "c.get('execution',{}).get('max_temp_size', '0')")
QUERY_TIMEOUT=$(read_config "c.get('execution',{}).get('query_timeout_s', 0.0)")
VALIDATION_WORKERS=$(read_config "c.get('execution',{}).get('validation_workers', 0)")
MEASURE_WORKLOAD_BASELINE=$(read_config "c.get('execution',{}).get('measure_workload_baseline', True)")
SQL_DIR=$(read_config "c['dataset']['sql_dir']")

# ------------------------------------------------
# Integrity guard: the remote workspace ($REMOTE_PATH) is shared across
# experiments. If a remote run is killed (e.g. OOM from a runaway query) it
# leaves no fresh output, and a stale result file from a *different* dataset
# can otherwise be scp'd back and silently propagated. After every copy-back we
# assert the returned result keys are a subset of the submitted input keys; on
# mismatch we abort instead of poisoning the pipeline with foreign data.
verify_keys_subset() {
    # $1 = submitted input json (transfer), $2 = returned result json
    local input_json="$1" result_json="$2"
    python3 - "$input_json" "$result_json" <<'PYEOF'
import json, sys
input_path, result_path = sys.argv[1], sys.argv[2]
with open(input_path) as f:
    input_keys = set(json.load(f).keys())
with open(result_path) as f:
    result_keys = set(json.load(f).keys())
# Result keys carry suffixes (e.g. "_no_change", "_rerun"); match on the query
# prefix before the first "_<digit>" rule marker, which is dataset-identifying.
def base(k):
    return k.split("_")[0]
input_bases = {base(k) for k in input_keys}
foreign = sorted({k for k in result_keys if base(k) not in input_bases})
if foreign:
    print(f"INTEGRITY CHECK FAILED: {result_path} contains {len(foreign)} key(s) "
          f"absent from {input_path} (foreign dataset?). Examples: {foreign[:5]}",
          file=sys.stderr)
    sys.exit(1)
PYEOF
}

# ------------------------------------------------
# Activate virtual environment
source "$REPO_ROOT/.venv/bin/activate"

# python3 -m calls need repo root as cwd
cd "$REPO_ROOT"

# ------------------------------------------------
# Remote setup (needed for stages 1, 2, 3 — anything that runs on remote)

NEEDS_REMOTE=true
if [ "$START_STAGE" -ge 4 ]; then
    NEEDS_REMOTE=false
fi
# --baseline forced START_STAGE=4 (to skip stages 1-3) but still measures on remote.
if [ "$BASELINE_ONLY" = "true" ]; then
    NEEDS_REMOTE=true
fi

# ------------------------------------------------
# Stage 1: Rule generation (local LLM calls) + initial remote execution

if [ "$START_STAGE" -le 1 ]; then
    python3 -m systematic_eval.run_pipeline --config "$CONFIG" --stage generate --transfer-dir "$TRANSFER_DIR"
fi

if [ "$NEEDS_REMOTE" = "true" ]; then
# Ensure remote directory exists
ssh "$SERVER" "mkdir -p $REMOTE_PATH"

# Copy execution script, rule engine, and transfer data to remote server
scp "$SCRIPT_DIR/stages/execution.py" "$SERVER:${REMOTE_PATH}run_sql.py"
scp "$SCRIPT_DIR/sql_predicate_converter.py" "$SERVER:${REMOTE_PATH}sql_predicate_converter.py"
scp "$SCRIPT_DIR/remote_requirements.txt" "$SERVER:${REMOTE_PATH}remote_requirements.txt"
# transfer.json only exists for stage<=1 runs; a --baseline backfill skips generation.
[ -f "$TRANSFER_DIR/transfer.json" ] && scp "$TRANSFER_DIR/transfer.json" "$SERVER:$REMOTE_PATH"
if [ "$ENGINE" = "umbra" ]; then
    scp "$SCRIPT_DIR/umbra_setup.py" "$SERVER:${REMOTE_PATH}umbra_setup.py"
    scp "$SCRIPT_DIR/resource_monitor.py" "$SERVER:${REMOTE_PATH}resource_monitor.py"
fi
if [ "$ENGINE" = "postgres" ]; then
    scp "$SCRIPT_DIR/pg_lab_setup.py" "$SERVER:${REMOTE_PATH}pg_lab_setup.py"
    ssh "$SERVER" "mkdir -p ${REMOTE_PATH}postgres"
    scp "$SCRIPT_DIR/postgres/postgresql16.conf" "$SERVER:${REMOTE_PATH}postgres/postgresql16.conf"
fi

# Ensure remote venv exists and dependencies are installed
ssh "$SERVER" << EOF
    cd "$REMOTE_PATH"
    [ -d .venv ] || python3 -m venv .venv
    source .venv/bin/activate
    pip install -q -r remote_requirements.txt
    if [ "$ENGINE" = "umbra" ] || [ "$ENGINE" = "postgres" ]; then
        pip install -q psycopg2-binary
    fi
    deactivate
EOF

# Interpreter used to run scorer/score_plans.py on the remote. Defaults to the venv
# built next to the scp'd scorer code; ZS_SCORER_PYTHON points at an already-built
# venv instead (e.g. a working one under another remote workspace on the shared
# labstore mount) and then skips the build entirely.
SCORER_PYTHON="${ZS_SCORER_PYTHON:-${REMOTE_PATH}scorer/.venv/bin/python}"
# Workspace root on the remote (…/psiegler); searched for a reusable scorer venv.
PARENT_DIR="$(dirname "${REMOTE_PATH%/}")"

# Set up the isolated ZeroShot scorer venv (py3.12 + torch/dgl + ldb_models) when
# cost_model=zeroshot. Heavy; built only for zeroshot runs. Model + feature_statistics
# are read straight from the labstore mount (no scp); only the scorer code is copied.
if [ "$COST_MODEL" = "zeroshot" ]; then
    echo "Setting up ZeroShot scorer on $SERVER..."
    ssh "$SERVER" "mkdir -p ${REMOTE_PATH}scorer"
    scp -r "$SCRIPT_DIR/scorer/." "$SERVER:${REMOTE_PATH}scorer/"
    ssh "$SERVER" << EOF
    cd "${REMOTE_PATH}scorer"
    SCORER_PY="$SCORER_PYTHON"
    # uv is the blessed path (handles dgl/torch indices); resolve it even when not on
    # the non-interactive PATH. Skip the (heavy) build if the venv already exists.
    UVBIN="\$(command -v uv 2>/dev/null || true)"
    [ -z "\$UVBIN" ] && [ -x "\$HOME/.local/bin/uv" ] && UVBIN="\$HOME/.local/bin/uv"
    if [ "\$SCORER_PY" != "${REMOTE_PATH}scorer/.venv/bin/python" ]; then
        echo "using external scorer venv: \$SCORER_PY"
    else
        # A venv missing ldb_models is a half-finished build (the pip fallback below
        # installs deps first, so torch can be present while the package is not).
        # Drop it, otherwise the "already exists" guard hides the breakage until the
        # aggregation stage dies with ModuleNotFoundError hours into the run.
        if [ -d .venv ] && ! ./.venv/bin/python -c "import ldb_models" 2>/dev/null; then
            echo "scorer venv incomplete (ldb_models missing) - rebuilding"
            rm -rf .venv
        fi
        if [ -d .venv ]; then
            echo "scorer venv already exists; skipping build (delete .venv to rebuild)"
        elif [ -n "\$UVBIN" ]; then
            "\$UVBIN" venv --python 3.12 .venv && "\$UVBIN" sync || exit 1
        elif DONOR=\$(for c in ${PARENT_DIR}/*/scorer/.venv; do
                         [ "\$c" = "${REMOTE_PATH}scorer/.venv" ] && continue
                         "\$c/bin/python" -c "import ldb_models" 2>/dev/null && { echo "\$c"; break; }
                     done) && [ -n "\$DONOR" ]; then
            # No uv here, and the pip fallback below cannot install the private
            # ldb_models repo reliably. Sibling workspaces sit on the same shared
            # labstore mount and pin the same /usr/bin/python3.12, so linking to a
            # known-good venv is instant and avoids a ~5G duplicate.
            echo "uv not found on $SERVER - linking scorer venv to \$DONOR"
            ln -s "\$DONOR" .venv || exit 1
        else
            echo "uv not found on $SERVER - falling back to pip"
            python3.12 -m venv .venv || exit 1
            ./.venv/bin/pip install -q -U pip || exit 1
            ./.venv/bin/pip install -q torch==2.3.0 psycopg2-binary orjson numpy tqdm || exit 1
            ./.venv/bin/pip install -q "https://data.dgl.ai/wheels/torch-2.3/cu121/dgl-2.4.0%2Bcu121-cp312-cp312-manylinux1_x86_64.whl" || exit 1
            # Not -q: the private-repo install is the step that fails in practice
            # (auth/build), and its error message is what you need to see.
            ./.venv/bin/pip install "git+https://${LDB_MODELS_TOKEN:+${LDB_MODELS_TOKEN}@}github.com/DataManagementLab/ldb_models.git@subplex" || exit 1
        fi
    fi
    # Fail here, not three stages later, if the scorer cannot import its model stack.
    "\$SCORER_PY" -c "import ldb_models, torch, dgl" || {
        echo "ERROR: scorer venv at \$SCORER_PY is unusable (see import error above)."
        echo "       Rebuild it (delete ${REMOTE_PATH}scorer/.venv and re-run, ideally"
        echo "       with uv installed on $SERVER) or set ZS_SCORER_PYTHON to a working venv."
        exit 1
    }
EOF
fi

# Start Umbra Docker container on remote server (if engine=umbra)
if [ "$ENGINE" = "umbra" ]; then
    echo "Starting Umbra container on $SERVER..."
    ssh "$SERVER" "cd $REMOTE_PATH && source .venv/bin/activate && python3 umbra_setup.py --start --port $UMBRA_PORT --memory-gb $UMBRA_MEMORY_GB && deactivate"
fi

# Start pg_lab Docker container on remote server (if engine=postgres)
if [ "$ENGINE" = "postgres" ]; then
    echo "Starting pg_lab container on $SERVER..."
    PG_SET_FLAGS=""
    if [ -n "$POSTGRES_RANDOM_PAGE_COST" ]; then
        PG_SET_FLAGS="--set random_page_cost=$POSTGRES_RANDOM_PAGE_COST"
        echo "  random_page_cost = $POSTGRES_RANDOM_PAGE_COST (from execution.postgres_random_page_cost)"
    fi
    ssh "$SERVER" "cd $REMOTE_PATH && source .venv/bin/activate && python3 pg_lab_setup.py --start --port $POSTGRES_PORT --conf postgres/postgresql16.conf $PG_SET_FLAGS && deactivate"
fi

# DuckDB: prime the OS page cache by sequentially reading the DB file once.
# Cheap (no parsing/decompression) and gives a deterministic starting cache
# state across measurement runs. Postgres/Umbra read from container-internal
# storage, so this would be a no-op for them.
if [ "$ENGINE" = "duckdb" ] && [ "$PRIME_OS_CACHE" = "True" ]; then
    echo "Priming OS page cache via cat $DB_FILE on $SERVER..."
    PRIME_START=$(date +%s)
    ssh "$SERVER" << EOF
        cd "$REMOTE_PATH"
        source .venv/bin/activate
        run_exp -m "eval" -n 1 --exclusive -- "cat $DB_FILE > /dev/null"
        deactivate
EOF
    PRIME_END=$(date +%s)
    echo "OS page cache prime done (took $((PRIME_END - PRIME_START))s)."
elif [ "$ENGINE" = "duckdb" ]; then
    echo "Skipping OS page cache prime (execution.prime_os_cache=false)."
fi

# One-time warmup before measurements. For postgres/umbra this primes the
# server's buffer pool + plan cache (state survives in the container). For
# DuckDB the engine-internal state is lost on connection close, but the OS
# page cache populated by the warmup queries persists, which still tightens
# runtime variance.
if [ "$WARMUP_MODE" = "startup_only" ] || [ "$WARMUP_MODE" = "startup_and_per_query" ]; then
    echo "Priming $ENGINE caches via one-time warmup over all queries in $SQL_DIR..."
    # Ship the full SQL dir so warmup can run every query (not just the experiment subset).
    ssh "$SERVER" "mkdir -p ${REMOTE_PATH}warmup_sql"
    scp -q "$REPO_ROOT/$SQL_DIR"/*.sql "$SERVER:${REMOTE_PATH}warmup_sql/"
    WARMUP_ENGINE_FLAGS=""
    if [ "$ENGINE" = "umbra" ]; then
        WARMUP_ENGINE_FLAGS="--engine umbra --umbra-port $UMBRA_PORT"
    elif [ "$ENGINE" = "postgres" ]; then
        WARMUP_ENGINE_FLAGS="--engine postgres --postgres-host $POSTGRES_HOST --postgres-port $POSTGRES_PORT --postgres-user $POSTGRES_USER --postgres-password $POSTGRES_PASSWORD --postgres-dbname $POSTGRES_DBNAME"
    fi
    ssh "$SERVER" << EOF
        cd "$REMOTE_PATH"
        source .venv/bin/activate
        run_exp -m "eval" -n 1 --exclusive -- "python3 run_sql.py --mode warmup --sql-dir warmup_sql --output /dev/null --db-file $DB_FILE $WARMUP_ENGINE_FLAGS"
        deactivate
EOF
fi

fi  # end NEEDS_REMOTE setup block

# Build engine flags for all execution invocations
ENGINE_FLAGS=""
if [ "$ENGINE" = "umbra" ]; then
    ENGINE_FLAGS="--engine umbra --umbra-port $UMBRA_PORT"
elif [ "$ENGINE" = "postgres" ]; then
    ENGINE_FLAGS="--engine postgres --postgres-host $POSTGRES_HOST --postgres-port $POSTGRES_PORT --postgres-user $POSTGRES_USER --postgres-password $POSTGRES_PASSWORD --postgres-dbname $POSTGRES_DBNAME"
fi

# Driver-side memory backstop. run_sql.py fetches results into host RAM, which
# for umbra/postgres lives *outside* the DB container's cgroup cap — a runaway
# client-side fetch could still exhaust the host (the container cap only bounds
# the server). Wrap run_sql.py in a transient systemd user scope so it is
# OOM-killed at DRIVER_MEMORY_GB instead of taking the host down. Skipped for
# duckdb (the engine runs inside the driver, so its RAM use is legitimate) and
# when the cap is 0. Degrades to no wrapper if systemd-run --user is unavailable.
DRIVER_WRAP=""
if [ "$NEEDS_REMOTE" = "true" ] && { [ "$ENGINE" = "umbra" ] || [ "$ENGINE" = "postgres" ]; } \
   && [ "$(python3 -c "print(int(round(float('$DRIVER_MEMORY_GB')*1024)))")" != "0" ]; then
    DRIVER_MB=$(python3 -c "print(int(round(float('$DRIVER_MEMORY_GB')*1024)))")
    if ssh "$SERVER" "systemd-run --user --scope --quiet -p MemoryMax=256M -p MemorySwapMax=0 true" >/dev/null 2>&1; then
        DRIVER_WRAP="systemd-run --user --scope --quiet -p MemoryMax=${DRIVER_MB}M -p MemorySwapMax=0 "
        echo "Driver memory backstop: run_sql.py capped at ${DRIVER_MEMORY_GB} GB (systemd-run --user --scope)."
    else
        echo "WARNING: systemd-run --user with MemoryMax/MemorySwapMax is unavailable on $SERVER;"
        echo "         running run_sql.py WITHOUT a driver memory cap (server-side cap still applies)."
    fi
fi

# Build fix-join-order flag (only used for final execution)
FIX_JOIN_ORDER_FLAG=""
if [ "$FIX_JOIN_ORDER" = "True" ]; then
    FIX_JOIN_ORDER_FLAG="--fix-join-order"
fi

# ------------------------------------------------
# Whole-workload baseline measurement (shared by the always-on stage-3 call and
# the manual --baseline backfill). Measures the ORIGINAL runtime of queries that
# had no rule applied — the gray 1.0x bars on the *_speedup_bar_all plots — so the
# runtime-weighted improvement can be computed over the full workload. It does NOT
# recompute the rule queries; their original+improved runtimes stay as measured.
# Requires a container (called only when NEEDS_REMOTE=true) and rule_summary_result.json.
run_baseline_measurement() {
    # Build baseline_input.json locally: workload queries minus those already measured.
    python3 -m systematic_eval.run_pipeline --config "$CONFIG" --stage baseline-prep --transfer-dir "$TRANSFER_DIR"
    if [ ! -f "$TRANSFER_DIR/baseline_input.json" ]; then
        echo "Baseline: no no-rule queries to measure; skipping."
        return 0
    fi
    scp "$TRANSFER_DIR/baseline_input.json" "$SERVER:$REMOTE_PATH"
    ssh "$SERVER" "rm -f ${REMOTE_PATH}baseline_runtimes.json"
    ssh "$SERVER" << EOF
    cd "$REMOTE_PATH"
    source .venv/bin/activate
    # Serial/exclusive, same timing path as the final measurement run.
    run_exp -m "eval" -n 1 --exclusive -- "${DRIVER_WRAP}python3 run_sql.py --mode baseline --input baseline_input.json --output baseline_runtimes.json --db-file $DB_FILE --attempts $ATTEMPTS --warmup-mode $WARMUP_MODE --max-temp-size $MAX_TEMP_SIZE --query-timeout $QUERY_TIMEOUT $ENGINE_FLAGS"
    deactivate
EOF
    scp "$SERVER:${REMOTE_PATH}baseline_runtimes.json" "$TRANSFER_DIR/"
    verify_keys_subset "$TRANSFER_DIR/baseline_input.json" "$TRANSFER_DIR/baseline_runtimes.json"
}

# Initial execution (part of stage 1) — runs original generated rules on remote
if [ "$START_STAGE" -le 1 ]; then
    # Delete any stale result.json on the shared remote path first, so a killed
    # run leaves no file and the scp below fails loudly (rather than copying a
    # previous experiment's leftover result back).
    ssh "$SERVER" "rm -f ${REMOTE_PATH}result.json"
    ssh "$SERVER" << EOF
    cd "$REMOTE_PATH"
    source .venv/bin/activate
    # Validation run: shared node (no --exclusive), parallel workers. Correctness
    # only — rule selection uses EXPLAIN cost, not these timings.
    run_exp -m "eval" -n 1 -- "${DRIVER_WRAP}python3 run_sql.py --input transfer.json --output result.json --db-file $DB_FILE --threshold $THRESHOLD --attempts $ATTEMPTS --validation-mode $VALIDATION_MODE --warmup-mode $WARMUP_MODE --max-temp-size $MAX_TEMP_SIZE --query-timeout $QUERY_TIMEOUT --workers $VALIDATION_WORKERS $ENGINE_FLAGS"
    deactivate
EOF

    # Copy results back (fails the script if the remote run produced no output)
    scp "$SERVER:${REMOTE_PATH}result.json" "$TRANSFER_DIR/"
    verify_keys_subset "$TRANSFER_DIR/transfer.json" "$TRANSFER_DIR/result.json"
fi

# ------------------------------------------------
# Stage 2: Refinement loop (configurable iterations)

if [ "$START_STAGE" -le 2 ] && [ "$REFINE_ENABLED" = "True" ]; then
    if [ ! -f "$TRANSFER_DIR/result.json" ]; then
        echo "Error: --stage $START_STAGE requires $TRANSFER_DIR/result.json from a prior run"
        exit 1
    fi
    for i in $(seq 0 $((REFINE_ITERATIONS - 1))); do
        ITER_NUM=$((i + 1))
        TRANSFER_FILE="transfer$((ITER_NUM + 1)).json"
        RESULT_FILE="result$((ITER_NUM + 1)).json"

        python3 -m systematic_eval.run_pipeline --config "$CONFIG" --stage refine --iteration "$i" --transfer-dir "$TRANSFER_DIR"

        scp "$TRANSFER_DIR/$TRANSFER_FILE" "$SERVER:$REMOTE_PATH"

        ssh "$SERVER" "rm -f ${REMOTE_PATH}${RESULT_FILE}"
        ssh "$SERVER" << EOF
            cd "$REMOTE_PATH"
            source .venv/bin/activate
            # Validation run: shared node (no --exclusive), parallel workers.
            run_exp -m "eval" -n $ITER_NUM -- "${DRIVER_WRAP}python3 run_sql.py --input $TRANSFER_FILE --output $RESULT_FILE --db-file $DB_FILE --threshold $THRESHOLD --attempts $ATTEMPTS --validation-mode $VALIDATION_MODE --warmup-mode $WARMUP_MODE --max-temp-size $MAX_TEMP_SIZE --query-timeout $QUERY_TIMEOUT --workers $VALIDATION_WORKERS $ENGINE_FLAGS"
            deactivate
EOF

        scp "$SERVER:${REMOTE_PATH}$RESULT_FILE" "$TRANSFER_DIR/"
        verify_keys_subset "$TRANSFER_DIR/$TRANSFER_FILE" "$TRANSFER_DIR/$RESULT_FILE"
    done
fi

# ------------------------------------------------
# Stage 3: Aggregation + final execution
# Tightly coupled: aggregation produces rule_summary_transfer.json which final
# execution consumes. Keeping them in one stage avoids exposing an internal seam.

if [ "$START_STAGE" -le 3 ]; then
    python3 -m systematic_eval.run_pipeline --config "$CONFIG" --stage aggregate --transfer-dir "$TRANSFER_DIR"

    if [ "$COST_ESTIMATION" = "True" ]; then
        # Cost-filtered aggregation: run rule matching + EXPLAIN cost filtering on remote.
        # For oracle/both runs, keep the full fired-rule pool per query (cost winner is
        # recorded only as metadata) so final execution still enumerates the full powerset
        # for the oracle upper bound; the optimizer view selects the winner in statistics.
        KEEP_FULL_POOL_FLAG=""
        if [ "$STATS_MODE" = "oracle" ] || [ "$STATS_MODE" = "both" ]; then
            KEEP_FULL_POOL_FLAG="--keep-full-pool"
        fi
        scp "$TRANSFER_DIR/cost_aggregate_input.json" "$SERVER:$REMOTE_PATH"

        if [ "$COST_MODEL" = "zeroshot" ]; then
            # Learned-model selection: EXPLAIN VERBOSE candidates + external scorer.
            ssh "$SERVER" << EOF
        cd "$REMOTE_PATH"
        source .venv/bin/activate
        run_exp -m "eval" -n 1 --exclusive -- "${DRIVER_WRAP}python3 run_sql.py --mode zeroshot-aggregate --input cost_aggregate_input.json --output rule_summary_transfer.json --db-file $DB_FILE --cost-save-threshold $COST_SAVE_THRESHOLD $ENGINE_FLAGS $FIX_JOIN_ORDER_FLAG $KEEP_FULL_POOL_FLAG --max-subsets $MAX_SUBSETS --scorer-python $SCORER_PYTHON --scorer-script ${REMOTE_PATH}scorer/score_plans.py --zeroshot-model-type '$ZS_MODEL_TYPE' --zeroshot-model-dir '$ZS_MODEL_DIR' --zeroshot-seed $ZS_SEED --zeroshot-statistics-file '$ZS_STATS_FILE' --zeroshot-database-stats '$ZS_DB_STATS'"
        deactivate
EOF
        else
            ssh "$SERVER" << EOF
        cd "$REMOTE_PATH"
        source .venv/bin/activate
        run_exp -m "eval" -n 1 --exclusive -- "${DRIVER_WRAP}python3 run_sql.py --mode cost-aggregate --input cost_aggregate_input.json --output rule_summary_transfer.json --db-file $DB_FILE --cost-save-threshold $COST_SAVE_THRESHOLD $ENGINE_FLAGS $FIX_JOIN_ORDER_FLAG $KEEP_FULL_POOL_FLAG"
        deactivate
EOF
        fi

        scp "$SERVER:${REMOTE_PATH}rule_summary_transfer.json" "$TRANSFER_DIR/"
    fi

    # Final execution: run original vs refined queries
    scp "$TRANSFER_DIR/rule_summary_transfer.json" "$SERVER:$REMOTE_PATH"

    ssh "$SERVER" "rm -f ${REMOTE_PATH}rule_summary_result.json"
    ssh "$SERVER" << EOF
    cd "$REMOTE_PATH"
    source .venv/bin/activate
    run_exp -m "eval" -n 1 --exclusive -- "${DRIVER_WRAP}python3 run_sql.py --input rule_summary_transfer.json --output rule_summary_result.json --db-file $DB_FILE --threshold $THRESHOLD --attempts $ATTEMPTS --validation-mode query --warmup-mode $WARMUP_MODE --max-temp-size $MAX_TEMP_SIZE --query-timeout $QUERY_TIMEOUT $ENGINE_FLAGS $FIX_JOIN_ORDER_FLAG"
    deactivate
EOF

    scp "$SERVER:${REMOTE_PATH}rule_summary_result.json" "$TRANSFER_DIR/"
    verify_keys_subset "$TRANSFER_DIR/rule_summary_transfer.json" "$TRANSFER_DIR/rule_summary_result.json"

    # Always-on: measure the no-rule query baselines in the SAME session (container
    # still up), so the runtime-weighted whole-workload metric appears on the *_all
    # plots. Disable with execution.measure_workload_baseline: false.
    if [ "$MEASURE_WORKLOAD_BASELINE" = "True" ]; then
        echo "Measuring whole-workload baseline (no-rule query runtimes)..."
        run_baseline_measurement
    fi
fi

# Manual --baseline backfill: stages 1-3 were skipped; measure the no-rule query
# runtimes now (container is up because NEEDS_REMOTE was forced on).
if [ "$BASELINE_ONLY" = "true" ]; then
    echo "Baseline backfill: measuring no-rule query runtimes for '$EXPERIMENT_NAME'..."
    run_baseline_measurement
fi

# Container teardown — only if we started one (i.e. needed remote)
if [ "$NEEDS_REMOTE" = "true" ]; then
    if [ "$ENGINE" = "umbra" ]; then
        echo "Stopping Umbra container on $SERVER..."
        ssh "$SERVER" "cd $REMOTE_PATH && source .venv/bin/activate && python3 umbra_setup.py --teardown --port $UMBRA_PORT && deactivate"
    fi

    if [ "$ENGINE" = "postgres" ]; then
        echo "Stopping pg_lab container on $SERVER..."
        ssh "$SERVER" "cd $REMOTE_PATH && source .venv/bin/activate && python3 pg_lab_setup.py --teardown --port $POSTGRES_PORT && deactivate"
    fi
fi

# ------------------------------------------------
# Stage 4: Statistics

python3 -m systematic_eval.run_pipeline --config "$CONFIG" --stage stats --transfer-dir "$TRANSFER_DIR"$STATS_EXTRA_ARGS

deactivate

echo "Done! Results saved to: saved_results/${EXPERIMENT_NAME}${DIR_SUFFIX}/"
