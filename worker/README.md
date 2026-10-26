# Live-mode proxy (Cloudflare Worker)

Lets visitors of the GitHub Pages demo press **Live**: the agent runs in their browser, and each turn goes
through this Worker to Claude Haiku. The API key stays in a Worker secret.

Guards: allowed origins only; a signed 30-minute session per run, at most 3 runs per visitor and 60 per
day (`wrangler.toml` vars); about 40 turns per minute per session; only the page's own prompts and tools;
model and `max_tokens` fixed in the Worker. The hard cap is the spend limit on the Anthropic workspace.

## One-time setup

1. Anthropic Console: create a workspace for the demo, set its monthly spend limit, and create an API key in it.
2. `cd worker && npx wrangler@4 login`
3. `npx wrangler@4 kv namespace create COUNTS` and put the printed id into `wrangler.toml`.
4. `npx wrangler@4 secret put ANTHROPIC_API_KEY` (paste the workspace key)
5. `node -e "console.log(crypto.randomBytes(32).toString('hex'))" | npx wrangler@4 secret put TOKEN_SECRET`
6. `bash deploy.sh`, then build the site with the Worker's URL:
   `python scripts/build_site.py --proxy https://labloop-live.<your-subdomain>.workers.dev`

## Local test (no key in the Worker)

`.dev.vars` with `UPSTREAM=http://127.0.0.1:8777/messages`, `TOKEN_SECRET=dev`,
`ALLOWED_ORIGINS=http://127.0.0.1:8765`; run `python scripts/live_proxy.py` (reads the key from `.env`),
`npx wrangler@4 dev`, and serve `docs/` built with `--proxy http://127.0.0.1:8787` on port 8765.
