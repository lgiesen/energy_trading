#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Pull challenger training artifacts from remote server to local machine.

Usage:
  scripts/pull_challenger_artifacts.sh --ssh-host <host-or-alias> [options]

Options:
  --ssh-host <value>     Required. SSH host/IP or local ~/.ssh/config alias.
  --ssh-user <value>     Optional. Defaults to current local user.
  --remote-root <path>   Optional. Remote repo root.
                         Default: /home/l/<ssh-user>/energy_trading
  --dest <path>          Optional. Local destination directory.
                         Default: artifacts/downloads/challenger

Examples:
  scripts/pull_challenger_artifacts.sh --ssh-host uni-gpu --ssh-user l_gies10
  scripts/pull_challenger_artifacts.sh --ssh-host 164.92.233.7 --ssh-user l_gies10
EOF
}

ssh_host=""
ssh_user="$(whoami)"
dest="artifacts/downloads/challenger"
remote_root=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ssh-host)
      ssh_host="${2:-}"
      shift 2
      ;;
    --ssh-user)
      ssh_user="${2:-}"
      shift 2
      ;;
    --remote-root)
      remote_root="${2:-}"
      shift 2
      ;;
    --dest)
      dest="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$ssh_host" ]]; then
  echo "Error: --ssh-host is required." >&2
  usage >&2
  exit 2
fi

if [[ -z "$remote_root" ]]; then
  remote_root="/home/l/${ssh_user}/energy_trading"
fi

mkdir -p "$dest"

remote_prefix="${ssh_user}@${ssh_host}:${remote_root}"

echo "Pulling challenger artifacts from ${remote_prefix}"
rsync -avh --progress \
  "${remote_prefix}/models/checkpoints/xgboost_afrr_challenger.joblib" \
  "${dest}/"
rsync -avh --progress \
  "${remote_prefix}/data/reports/xgboost_challenger_report.json" \
  "${dest}/"
rsync -avh --progress \
  "${remote_prefix}/data/reports/xgboost_optuna_trials.csv" \
  "${dest}/"

echo
echo "[OK] Download complete."
echo "Local destination: ${dest}"
