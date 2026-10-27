#!/usr/bin/env bash
# Pull the page's data from the server, build the GitHub Pages site with Live mode, commit and push:
#   bash scripts/publish_site.sh "commit message"
# (run scripts/refresh_web.sh on the server first; the claude.ai artifact is published separately)
set -eu
HOST=${HOST:-scottlee@130.126.138.27}; REPO=${REPO:-labloop2}  # relative to the server home (a local ~ would expand here)
PROXY=${PROXY:-https://labloop-live.scottlee.workers.dev}
cd "$(dirname "$0")/.."
scp -q "$HOST:$REPO/web/data/replay_console.json" "$HOST:$REPO/web/data/baselines.json" web/data/
scp -q "$HOST:$REPO/web/data/replay_combo_*.json" web/data/
rm -f web/std.html web/std_live.html
python scripts/build_site.py --proxy "$PROXY"
git -c core.longpaths=true add -A
git -c core.longpaths=true commit -q -m "${1:-Refresh the web page data}

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git -c core.longpaths=true push -q origin main
git log --oneline | head -1
