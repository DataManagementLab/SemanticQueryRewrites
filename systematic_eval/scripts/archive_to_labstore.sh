#!/bin/bash
set -euo pipefail

# Archive the data behind the reported thesis runs to the labstore.
#
# Usage: ./archive_to_labstore.sh [--execute] [--copy-dbs] [--bwlimit KBPS]
#
# Selection (one run = one config in config/, template excluded):
#   saved_results/<run>/                 complete
#   transfer_data/<run>/                 only files NOT byte-identical in saved_results/<run>/
#   llm_cache/                           complete (raw LLM responses, read by thesis_numbers.py)
#   sql/<sql_dir>/                       every sql_dir referenced by a config
#   repo.bundle, ARCHIVE_README.md       git history + restore notes
#
# Default is a local dry run: checks paths, compares duplicates, prints the plan and its
# size. Nothing leaves this machine. --execute uploads via rsync (resumable).
# --copy-dbs copies the *.duckdb files server-side on the remote (no local traffic),
# with or without --execute.

SERVER="c07"
DEST="/mnt/labstore/psiegler/thesis_archive"
DB_SRC="/mnt/labstore/psiegler/c07_multi_query_comparison"

EXECUTE=false
COPY_DBS=false
BWLIMIT=""
while [ $# -gt 0 ]; do
    case "$1" in
        --execute) EXECUTE=true; shift ;;
        --copy-dbs) COPY_DBS=true; shift ;;
        --bwlimit) BWLIMIT="$2"; shift 2 ;;
        -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

SE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(dirname "$SE")"
cd "$ROOT"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
LIST="$STAGE/files.txt"
: > "$LIST"
missing=0

need() {
    if [ -e "$1" ]; then echo "$1" >> "$LIST"; else echo "MISSING: $1" >&2; missing=$((missing + 1)); fi
}

# --- runs --------------------------------------------------------------------
runs=()
for c in "$SE"/config/experiment_*.yaml; do
    n="$(basename "$c" .yaml)"
    [ "$n" = "experiment_template" ] || runs+=("$n")
done
echo "runs: ${#runs[@]}"

dup_bytes=0
for n in "${runs[@]}"; do
    sr="systematic_eval/saved_results/$n"
    td="systematic_eval/transfer_data/$n"
    need "$sr"
    [ -d "$td" ] || { echo "MISSING: $td" >&2; missing=$((missing + 1)); continue; }
    for f in "$td"/*; do
        [ -f "$f" ] || continue
        twin="$sr/$(basename "$f")"
        if [ -f "$twin" ] && cmp -s "$f" "$twin"; then
            dup_bytes=$((dup_bytes + $(stat -c %s "$f")))
        else
            echo "$f" >> "$LIST"
        fi
    done
done

# --- llm cache + workloads ---------------------------------------------------
need "llm_cache"
while read -r d; do
    need "$d"
done < <(grep -h '^[[:space:]]*sql_dir:' "$SE"/config/experiment_*.yaml \
         | grep -v experiment_template | sed -E 's/.*sql_dir:[[:space:]]*"?([^"#]*)"?.*/\1/' \
         | sed 's/[[:space:]]*$//' | sort -u)

if [ "$missing" -gt 0 ]; then
    echo "$missing path(s) missing, aborting." >&2
    exit 1
fi

# --- git bundle + readme -----------------------------------------------------
git -C "$ROOT" bundle create "$STAGE/repo.bundle" --all 2>/dev/null
COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
cat > "$STAGE/ARCHIVE_README.md" <<EOF
# Thesis archive: Semantic Query Rewrites (Philipp Siegler, TU Darmstadt)

Created $(date -I) from commit \`$COMMIT\`.

Data behind the ${#runs[@]} runs reported in the thesis (one per config in
\`systematic_eval/config/\`). Paths mirror the repository layout.

| Path | Content |
|---|---|
| \`repo.bundle\` | Full git history (\`git clone repo.bundle code\`) |
| \`systematic_eval/saved_results/<run>/\` | Final outputs of each reported run, complete |
| \`systematic_eval/transfer_data/<run>/\` | Stage I/O **without** files byte-identical in \`saved_results/<run>/\` |
| \`llm_cache/\` | Raw LLM requests + responses (reproduction without API cost; read by \`thesis_numbers.py\`) |
| \`sql/\` | Query workloads used by the configs |
| \`databases/\` | DuckDB files (if copied; source: \`$DB_SRC\`) |

## Restore

\`\`\`bash
git clone repo.bundle code && cd code
rsync -a <archive>/{llm_cache,sql} .
rsync -a <archive>/systematic_eval/ systematic_eval/
# transfer_data duplicates were dropped; copy them back from saved_results:
for d in systematic_eval/transfer_data/*/; do
  n=\$(basename "\$d"); cp -n systematic_eval/saved_results/"\$n"/*.json "\$d"
done
\`\`\`

\`scripts/thesis_numbers.py\` and \`scripts/thesis_figures.py\` read both
\`saved_results/\` and \`transfer_data/\`, so the restore step above is required.
EOF

# --- plan --------------------------------------------------------------------
gib() { awk -v b="$1" 'BEGIN { printf "%.2f GiB", b / 1024^3 }'; }
total=$(tr '\n' '\0' < "$LIST" | du -scb --files0-from=- | tail -1 | cut -f1)
extra=$(du -scb "$STAGE/repo.bundle" "$STAGE/ARCHIVE_README.md" | tail -1 | cut -f1)
echo "entries:          $(wc -l < "$LIST")"
echo "upload size:      $(gib $((total + extra)))"
echo "skipped (dups):   $(gib "$dup_bytes")"
echo "destination:      $SERVER:$DEST"

if $COPY_DBS; then
    ssh "$SERVER" "mkdir -p '$DEST/databases' && cp -v --update '$DB_SRC'/*.duckdb '$DEST/databases/'"
fi

if ! $EXECUTE; then
    echo "dry run only, nothing uploaded (pass --execute to upload)."
    exit 0
fi

# --- upload ------------------------------------------------------------------
RSYNC=(rsync -a -r --partial --info=progress2 --human-readable)
[ -n "$BWLIMIT" ] && RSYNC+=(--bwlimit="$BWLIMIT")

ssh "$SERVER" "mkdir -p '$DEST'"
"${RSYNC[@]}" "$STAGE/repo.bundle" "$STAGE/ARCHIVE_README.md" "$SERVER:$DEST/"
"${RSYNC[@]}" --files-from="$LIST" "$ROOT/" "$SERVER:$DEST/"

echo "done."
