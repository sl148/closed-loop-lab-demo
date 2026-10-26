#!/usr/bin/env bash
# Bundle the archive and deploy the live-mode proxy:  bash worker/deploy.sh
set -eu
cd "$(dirname "$0")"
cp ../web/data/kb.json src/kb.txt
npx -y wrangler@4 deploy
