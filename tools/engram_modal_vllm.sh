#!/usr/bin/env bash
set -euo pipefail

remote="${VLLM_MODAL_SSH_HOST:-modal-ssh}"
remote_dir="${VLLM_MODAL_REMOTE_DIR:-/root/vllm}"
repo_root="$(git rev-parse --show-toplevel)"
local_commit="$(git rev-parse HEAD)"

usage() {
  echo "usage: $0 {sync|run|shell|status} [command ...]" >&2
  exit 2
}

case "${1:-}" in
  sync)
    ssh "$remote" "mkdir -p '$remote_dir'"
    rsync -az --delete \
      --filter=':- .gitignore' \
      --exclude .git \
      --exclude .venv \
      --exclude __pycache__ \
      --exclude .ruff_cache \
      "$repo_root/" "$remote:$remote_dir/"
    ssh "$remote" "git config --global --add safe.directory '$remote_dir' && \
      cd '$remote_dir' && \
      if [[ ! -d .git ]]; then git init -b modal-sync; fi && \
      git config user.name 'Engram Modal Sync' && \
      git config user.email 'modal-sync@engram.local' && \
      git add -A && \
      (git diff --cached --quiet || git commit -m 'Sync $local_commit')"
    ;;
  run)
    shift
    (($#)) || usage
    printf -v command '%q ' "$@"
    ssh "$remote" "cd '$remote_dir' && \
      export VIRTUAL_ENV='$remote_dir/.venv' && \
      export PATH='$remote_dir/.venv/bin':\$PATH && \
      $command"
    ;;
  shell)
    ssh -t "$remote" "cd '$remote_dir' && \
      export VIRTUAL_ENV='$remote_dir/.venv' && \
      export PATH='$remote_dir/.venv/bin':\$PATH && \
      exec \${SHELL:-/bin/bash} -l"
    ;;
  status)
    ssh "$remote" "nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader && test -d '$remote_dir' && du -sh '$remote_dir' || true"
    ;;
  *) usage ;;
esac
