#!/usr/bin/env bash
#
# push.sh — stage, commit, and push all local changes to the current branch.
#
# Usage:
#   ./push.sh                      # commit with a timestamped default message
#   ./push.sh "your message here"  # commit with a custom message
#
set -euo pipefail

# Run from the repo root regardless of where the script is called from.
cd "$(dirname "$0")"

# Make sure we're in a git repo.
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Error: not inside a git repository." >&2
    exit 1
fi

branch="$(git rev-parse --abbrev-ref HEAD)"

# Bail out early if there's nothing to commit.
if [ -z "$(git status --porcelain)" ]; then
    echo "Nothing to commit — working tree clean. Pushing anything unpushed..."
    git push origin "$branch"
    exit 0
fi

# Commit message: first argument, or a timestamped default.
msg="${1:-Update $(date '+%Y-%m-%d %H:%M:%S')}"

echo "Branch:  $branch"
echo "Message: $msg"
echo

git add -A
git commit -m "$msg"
git push origin "$branch"

echo
echo "Done — pushed to origin/$branch."
