#!/usr/bin/env bash

# Simple bidirectional git sync: pulls remote changes and pushes local ones.
# - Stashes local changes before pulling to reduce conflicts
# - Re-applies local changes and commits if needed
# - Pushes resulting state to origin/main

set -uo pipefail

PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
REPO_DIR="/root/pinp-8448"
BRANCH="main"
LOG_FILE="$REPO_DIR/sync.log"

exec 9>"$REPO_DIR/.git-sync.lock" || exit 0
flock -n 9 || exit 0

timestamp() { date -Iseconds; }
log() { echo "[$(timestamp)] $*" | tee -a "$LOG_FILE"; }

cd "$REPO_DIR" || exit 1

# Safety for running as root inside containers
git config --global --add safe.directory "$REPO_DIR" >/dev/null 2>&1 || true

# Ensure identity (only if not set)
if ! git config user.name >/dev/null; then git config user.name "Auto Sync"; fi
if ! git config user.email >/dev/null; then git config user.email "autosync@local"; fi

log "Starting git sync in $REPO_DIR on branch $BRANCH"

# Ensure branch exists and checked out
current_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")
if [ "$current_branch" != "$BRANCH" ]; then
  git checkout -B "$BRANCH" >/dev/null 2>&1 || git checkout "$BRANCH" || true
fi

# Detect dirty state and stash (including untracked)
dirty=0
git diff --quiet || dirty=1
git diff --cached --quiet || dirty=1
if [ "$dirty" -ne 0 ]; then
  log "Working tree dirty, stashing local changes"
  git stash push -u -m "auto-sync $(timestamp)" >/dev/null 2>&1 || true
fi

# Fetch and integrate remote changes
git fetch origin "$BRANCH" >/dev/null 2>&1 || true
if git rev-parse --verify origin/"$BRANCH" >/dev/null 2>&1; then
  if ! git pull --rebase origin "$BRANCH" -q; then
    log "Rebase failed, trying merge with 'theirs' preference"
    git rebase --abort >/dev/null 2>&1 || true
    git pull --no-rebase --no-edit -s ort -X theirs origin "$BRANCH" -q || true
  fi
fi

# Re-apply stashed changes if any
if git stash list | grep -q "auto-sync"; then
  log "Re-applying stashed local changes"
  if ! git stash pop -q; then
    # Conflicts: accept local by default, then commit
    log "Conflicts on stash pop; staging all and committing"
    git add -A
    git commit -m "chore: auto-merge local changes after pop $(timestamp)" >/dev/null 2>&1 || true
  fi
fi

# Stage and commit any remaining updates
if ! git diff --quiet || ! git diff --cached --quiet; then
  git add -A
  if ! git diff --cached --quiet; then
    log "Committing local changes"
    git commit -m "chore: auto-sync $(hostname -f 2>/dev/null || hostname) $(timestamp)" >/dev/null 2>&1 || true
  fi
fi

# Push to remote
if git rev-parse --verify origin/"$BRANCH" >/dev/null 2>&1; then
  git push origin "$BRANCH" -q || log "Push failed"
else
  git push -u origin "$BRANCH" -q || log "Initial push failed"
fi

log "Sync complete"


