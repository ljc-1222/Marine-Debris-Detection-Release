#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'Usage: %s [--ref COMMIT]\n' "${0##*/}" >&2
    exit 2
}

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$repo_root"
git rev-parse --is-inside-work-tree >/dev/null

ref=
if (( $# )); then
    [[ $# == 2 && $1 == --ref ]] || usage
    ref=$2
    git rev-parse --verify "$ref^{commit}" >/dev/null
fi

status=0

list_paths() {
    if [[ -n "$ref" ]]; then
        git ls-tree -r --name-only "$ref"
    else
        git ls-files
    fi
}

bad_paths=$(list_paths | awk '
        /^[Aa][Gg][Ee][Nn][Tt][Ss]?\.md$/ ||
        /^slurm\// ||
        /^scripts\/run_copy_paste_experiments\.sh$/ ||
        /^configs\/dcr_codetr_epoch19_provenance\.json$/ ||
        /^configs\/train\/rfdetr_(m|2xl)\.yaml$/ ||
        /^(data|datasets|Marine_Debris_Image_Dataset|runs|runs-smoke|runs-analysis|submissions|artifacts|data_derived|logs|wandb|\.cache|outputs|checkpoints|reports)\// ||
        /^docs\// && ! /^docs\/(README|method|inference|training)\.md$/ ||
        /(^|\/)[^\/]*(experiment|search|ablation|sweep|final_sprint)[^\/]*$/ ||
        /^reports\/dataset_eda\/tables\// ||
        /^docs\/prompts\// ||
        /(^|\/)marine_debris_dcr_agent_prompt\.md$/ ||
        ! /^tests\// && /(^|[\/_-])test([\/_.-]|$)|_submission([.]|$)/ ||
        /(^|\/)\.env($|\.)/ && ! /(^|\/)\.env\.example$/ ||
        /(^|\/)(id_rsa|id_ed25519)$/ ||
        /\.(pt|pth|ckpt|safetensors|onnx|engine|npy|npz|csv|ipynb|log|tar|gz|zip)$/ { print }
    ')
if [[ -n "$bad_paths" ]]; then
    printf 'Repository hygiene: prohibited tracked paths:\n%s\n' "$bad_paths" >&2
    status=1
fi

if [[ -n "$ref" ]]; then
    while IFS= read -r -d '' entry; do
        metadata=${entry%%$'\t'*}
        path=${entry#*$'\t'}
        read -r mode type object <<<"$metadata"
        [[ "$mode" == 160000 || "$type" != blob ]] && continue
        size=$(git cat-file -s "$object")
        if (( size > 5 * 1024 * 1024 )); then
            printf 'Repository hygiene: tracked file exceeds 5 MiB: %s (%s bytes)\n' \
                "$path" "$size" >&2
            status=1
        fi
    done < <(git ls-tree -r -z "$ref")
else
    while IFS= read -r -d '' entry; do
        metadata=${entry%%$'\t'*}
        path=${entry#*$'\t'}
        read -r mode object stage <<<"$metadata"
        [[ "$mode" == 160000 || "$stage" != 0 ]] && continue
        size=$(git cat-file -s "$object")
        if (( size > 5 * 1024 * 1024 )); then
            printf 'Repository hygiene: tracked file exceeds 5 MiB: %s (%s bytes)\n' \
                "$path" "$size" >&2
            status=1
        fi
    done < <(git ls-files --stage -z)
fi

scan_exclusions=(':(exclude)scripts/check_repo_hygiene.sh')
secret_pattern='AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}|github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|xox[baprs]-[A-Za-z0-9-]{10,}|Policy=eyJ|Signature=[A-Za-z0-9%~_-]{20,}|Key-Pair-Id=|Download-Request-ID=|Bearer[[:space:]]+[A-Za-z0-9._~-]{16,}|http://[0-9]{1,3}(\.[0-9]{1,3}){3}:[0-9]+/sharing/'
if [[ -n "$ref" ]]; then
    grep_command=(git grep -n -I -E "$secret_pattern" "$ref" -- . "${scan_exclusions[@]}")
else
    grep_command=(git grep --cached -n -I -E "$secret_pattern" -- . "${scan_exclusions[@]}")
fi
grep_status=0
matches=$("${grep_command[@]}" 2>&1) || grep_status=$?
if (( grep_status == 0 )); then
    printf 'Repository hygiene: possible credential or signed URL:\n%s\n' "$matches" >&2
    status=1
elif (( grep_status != 1 )); then
    printf 'Repository hygiene: credential scan failed:\n%s\n' "$matches" >&2
    status=1
fi

personal_pattern='/home/[^/[:space:]$<{]+/|/work/[0-9][^/[:space:]]*/|/work/HPC_software/|MST[0-9]{6,}|TRI[0-9]{6,}|cbi-lgn[0-9]*|hgpn[0-9]+'
if [[ -n "$ref" ]]; then
    grep_command=(git grep -n -I -E "$personal_pattern" "$ref" -- . "${scan_exclusions[@]}")
else
    grep_command=(git grep --cached -n -I -E "$personal_pattern" -- . "${scan_exclusions[@]}")
fi
grep_status=0
matches=$("${grep_command[@]}" 2>&1) || grep_status=$?
if (( grep_status == 0 )); then
    printf 'Repository hygiene: personal path or cluster account is tracked:\n%s\n' "$matches" >&2
    status=1
elif (( grep_status != 1 )); then
    printf 'Repository hygiene: personal-path scan failed:\n%s\n' "$matches" >&2
    status=1
fi

if [[ -n "$ref" ]]; then
    if ! git diff-tree --check --root --no-commit-id -r "$ref"; then
        status=1
    fi
else
    if ! git diff --cached --check; then
        status=1
    fi
fi

if (( status )); then
    exit "$status"
fi
printf 'Repository hygiene: OK\n'
