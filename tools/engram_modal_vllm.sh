#!/usr/bin/env bash
set -euo pipefail

remote="${VLLM_MODAL_SSH_HOST:-modal-ssh}"
remote_dir="${VLLM_MODAL_REMOTE_DIR:-/root/vllm}"
repo_root="$(git rev-parse --show-toplevel)"

usage() {
  echo "usage: $0 {sync|run|shell|status} [command ...]" >&2
  exit 2
}

case "${1:-}" in
  sync)
    ssh "$remote" "mkdir -p '$remote_dir'"
    rsync -az --delete \
      --exclude .git \
      --exclude .venv \
      --exclude __pycache__ \
      --exclude .ruff_cache \
      "$repo_root/" "$remote:$remote_dir/"
    ;;
  run)
    shift
    (($#)) || usage
    printf -v command '%q ' "$@"
    ssh "$remote" "cd '$remote_dir' && $command"
    ;;
  shell)
    ssh -t "$remote" "cd '$remote_dir' && exec \${SHELL:-/bin/bash} -l"
    ;;
  status)
    ssh "$remote" "nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader && test -d '$remote_dir' && du -sh '$remote_dir' || true"
    ;;
  *) usage ;;
esac
