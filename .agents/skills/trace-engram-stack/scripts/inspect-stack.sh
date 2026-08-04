#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'Usage: %s [ENGRAMS_ROOT TORCHTITAN_ROOT VLLM_ROOT]\n' "$0"
}

if (( $# != 0 && $# != 3 )); then
    usage >&2
    exit 2
fi

current_root=$(git rev-parse --show-toplevel)
checkout_parent=$(dirname "$current_root")

if (( $# == 3 )); then
    engrams_root=$1
    torchtitan_root=$2
    vllm_root=$3
else
    engrams_root="$checkout_parent/engrams"
    torchtitan_root="$checkout_parent/torchtitan"
    vllm_root="$checkout_parent/vllm"
    case "$(basename "$current_root")" in
        engrams) engrams_root=$current_root ;;
        torchtitan) torchtitan_root=$current_root ;;
        vllm) vllm_root=$current_root ;;
    esac
fi

require_repo() {
    local name=$1
    local root=$2
    if ! git -C "$root" rev-parse --git-dir >/dev/null 2>&1; then
        printf '%s is not a Git checkout: %s\n' "$name" "$root" >&2
        printf 'Pass all three repository roots explicitly.\n' >&2
        exit 1
    fi
}

for repo_spec in \
    "Engrams:$engrams_root" \
    "TorchTitan:$torchtitan_root" \
    "vLLM:$vllm_root"
do
    require_repo "${repo_spec%%:*}" "${repo_spec#*:}"
done

show_repo() {
    local name=$1
    local root=$2
    local expected_ref=$3
    local branch
    local head
    local origin_url
    local dirty_count

    branch=$(git -C "$root" symbolic-ref --quiet --short HEAD || printf 'detached')
    head=$(git -C "$root" rev-parse HEAD)
    origin_url=$(git -C "$root" remote get-url origin 2>/dev/null || printf 'missing')
    dirty_count=$(git -C "$root" status --porcelain | wc -l | tr -d ' ')

    printf '\n%s\n' "$name"
    printf '  root: %s\n' "$root"
    printf '  origin: %s\n' "$origin_url"
    printf '  checkout: %s @ %s\n' "$branch" "$head"
    printf '  working-tree entries: %s\n' "$dirty_count"
    if git -C "$root" rev-parse --verify --quiet "${expected_ref}^{commit}" >/dev/null; then
        printf '  integration ref: %s @ %s\n' \
            "$expected_ref" \
            "$(git -C "$root" rev-parse "${expected_ref}^{commit}")"
    else
        printf '  integration ref: %s is missing from local refs\n' "$expected_ref"
    fi
}

show_divergence() {
    local name=$1
    local root=$2
    local fork_ref=$3
    local upstream_ref=$4
    local counts
    local fork_only
    local upstream_only
    local upstream_remote
    local upstream_url

    if ! git -C "$root" rev-parse --verify --quiet "${fork_ref}^{commit}" >/dev/null \
        || ! git -C "$root" rev-parse --verify --quiet "${upstream_ref}^{commit}" >/dev/null
    then
        printf '\n%s divergence unavailable: fetch %s and %s\n' \
            "$name" "$fork_ref" "$upstream_ref"
        return
    fi

    counts=$(git -C "$root" rev-list --left-right --count "$fork_ref...$upstream_ref")
    fork_only=${counts%%[[:space:]]*}
    upstream_only=${counts##*[[:space:]]}
    upstream_remote=${upstream_ref%%/*}
    upstream_url=$(
        git -C "$root" remote get-url "$upstream_remote" 2>/dev/null \
            || printf 'not configured; ref may be stale'
    )
    printf '\n%s divergence\n' "$name"
    printf '  fork ref: %s @ %s\n' "$fork_ref" "$(git -C "$root" rev-parse "$fork_ref")"
    printf '  upstream ref: %s @ %s\n' \
        "$upstream_ref" "$(git -C "$root" rev-parse "$upstream_ref")"
    printf '  upstream remote: %s\n' "$upstream_url"
    printf '  merge base: %s\n' "$(git -C "$root" merge-base "$fork_ref" "$upstream_ref")"
    printf '  fork-only commits: %s\n' "$fork_only"
    printf '  upstream-only commits: %s\n' "$upstream_only"
}

show_embedded_torchtitan() {
    local engrams_ref=origin/dev
    local tree_entry
    local mode
    local import_commit
    local import_split

    if ! git -C "$engrams_root" rev-parse --verify --quiet "${engrams_ref}^{commit}" >/dev/null; then
        printf '\nEmbedded TorchTitan unavailable: fetch %s\n' "$engrams_ref"
        return
    fi

    tree_entry=$(git -C "$engrams_root" ls-tree "$engrams_ref" libs/torchtitan)
    mode=${tree_entry%% *}
    import_commit=$(
        git -C "$engrams_root" log "$engrams_ref" -1 \
            --format='%H' --grep='git-subtree-dir: libs/torchtitan'
    )
    import_split=$(
        git -C "$engrams_root" log "$engrams_ref" -1 \
            --format='%B' --grep='git-subtree-dir: libs/torchtitan' \
            | sed -n 's/^git-subtree-split: //p'
    )

    printf '\nEngrams embedded TorchTitan\n'
    printf '  tree entry: %s\n' "$tree_entry"
    if [[ $mode == 040000 ]]; then
        printf '  storage: ordinary directory tree, not a live submodule\n'
    elif [[ $mode == 160000 ]]; then
        printf '  storage: submodule gitlink\n'
    else
        printf '  storage: unexpected tree mode %s\n' "$mode"
    fi
    printf '  latest subtree import commit: %s\n' "${import_commit:-not found}"
    printf '  imported TorchTitan split: %s\n' "${import_split:-not found}"
}

show_repo Engrams "$engrams_root" origin/dev
show_repo TorchTitan "$torchtitan_root" origin/engram/torchtitan_apr_28_26
show_repo vLLM "$vllm_root" origin/main

show_divergence TorchTitan "$torchtitan_root" \
    origin/engram/torchtitan_apr_28_26 upstream/main
show_divergence vLLM "$vllm_root" origin/main upstream/main
show_embedded_torchtitan

printf '\nRefs above are local remote-tracking refs. Fetch before making a current-state claim.\n'
