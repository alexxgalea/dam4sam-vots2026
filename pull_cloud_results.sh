#!/bin/bash
# Incremental rescue puller — RUNS LOCALLY (this WSL2 box), NOT on the cloud instance.
# The cloud box can't push to WSL2 (no inbound route), so durability lives here:
# we mirror the cloud results dir down every INT seconds. A GPU fault / instance
# death then costs at most one interval of work instead of the whole run.
#
# This is the safety net that was MISSING on 2026-06-12, when a GPU Xid fault
# stranded 20 sequences on a box whose disk was still reachable (rescued by hand).
#
# Usage:   bash pull_cloud_results.sh <ssh_port> [host] [out_tracker] [interval_s]
# Example: bash pull_cloud_results.sh 40913 38.117.87.43 dam4sam_l_cloud 180
#
# Launch it in the background right after `cloud_run_l.sh full` starts, e.g.:
#   nohup bash pull_cloud_results.sh 40913 > pull_l_cloud.log 2>&1 &
set -uo pipefail
PORT="${1:?usage: pull_cloud_results.sh <ssh_port> [host] [out_tracker] [interval_s]}"
HOST="${2:-38.117.87.43}"
OUT="${3:-dam4sam_l_cloud}"
INT="${4:-180}"
BASE=/home/alexx/Projects/VOTSTracker
REMOTE="$BASE/workspace2026/results/$OUT/baseline/"
LOCAL="$BASE/workspace2026/results/$OUT/baseline/"
mkdir -p "$LOCAL"
SSH="ssh -p $PORT -o ConnectTimeout=15 -o BatchMode=yes -o ServerAliveInterval=10"

echo "[pull] mirroring root@$HOST:$PORT ($OUT) every ${INT}s -> $LOCAL"
# Do NOT `set -e` around the loop: a transient unreachable box must log + retry,
# never kill the puller. --partial keeps half-transferred files resumable.
while true; do
  ts=$(date '+%m-%d %H:%M:%S')
  if rsync -az --partial -e "$SSH" "root@$HOST:$REMOTE" "$LOCAL" 2>/tmp/pull_cloud.err; then
    n=$(ls "$LOCAL" 2>/dev/null | wc -l)
    echo "[$ts] synced — $n dirs local"
  else
    echo "[$ts] sync FAILED (box unreachable / GPU fault?) — $(tail -1 /tmp/pull_cloud.err 2>/dev/null)"
  fi
  sleep "$INT"
done
