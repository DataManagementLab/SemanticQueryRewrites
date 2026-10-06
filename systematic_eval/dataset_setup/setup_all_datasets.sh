#!/bin/bash
set -euo pipefail

# Onboard all 18 learned_db datasets (everything except imdb + basketball, which are
# already wired up). Runs add_dataset.sh sequentially; on a per-dataset failure it logs
# and continues, then reports the summary at the end.
#
# Usage: ./setup_all_datasets.sh [--count N] [--skip-build] [dataset ...]
#   With no dataset args, runs the full manifest below. Pass explicit <full_scaled_name>s
#   to onboard only a subset.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MANIFEST=(
    accidents_scaled1
    airline_scaled1
    baseball_scaled10
    carcinogenesis_scaled674
    consumer_scaled6
    credit_scaled5
    employee_scaled3
    fhnk_scaled2
    financial_scaled4
    geneea_scaled23
    genome_scaled6
    hepatitis_scaled2000
    movielens_scaled8
    seznam_scaled2
    ssb_scaled1
    tournament_scaled50
    tpc_h_scaled1
    walmart_scaled1
)

PASSTHRU=()
DATASETS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --count|--count=*|--skip-build) PASSTHRU+=("$1"); [ "$1" = "--count" ] && { PASSTHRU+=("$2"); shift; }; shift ;;
        *) DATASETS+=("$1"); shift ;;
    esac
done
[ ${#DATASETS[@]} -eq 0 ] && DATASETS=("${MANIFEST[@]}")

OK=()
FAILED=()
for ds in "${DATASETS[@]}"; do
    echo
    echo "############################################################"
    echo "# $ds"
    echo "############################################################"
    if "$SCRIPT_DIR/add_dataset.sh" "$ds" "${PASSTHRU[@]}"; then
        OK+=("$ds")
    else
        echo "!!! FAILED: $ds" >&2
        FAILED+=("$ds")
    fi
done

echo
echo "=== summary: ${#OK[@]} ok, ${#FAILED[@]} failed ==="
[ ${#OK[@]} -gt 0 ] && printf '  ok:     %s\n' "${OK[@]}"
[ ${#FAILED[@]} -gt 0 ] && printf '  FAILED: %s\n' "${FAILED[@]}"
[ ${#FAILED[@]} -eq 0 ]
