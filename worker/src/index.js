// Live-mode proxy for the GitHub Pages demo (Cloudflare Worker). The page's agent runs in the browser; each
// turn comes here, and the Worker calls Claude Haiku with a key the browser never sees. Guards, cheapest first:
// allowed origins; a signed session per run (KV counts runs per visitor and per day); a turn rate per session;
// only this page's prompts and tools; model and max_tokens fixed here. The workspace spend limit in the
// Anthropic Console is the hard cap behind all of them.
import KB from "./kb.txt";  // the document archive (web/data/kb.json), bundled at deploy, served only to sessions

const MODEL = "claude-haiku-5-5";
const MAX_TOKENS = 4000;
const TOOLS = new Set(["list_documents", "search_corpus", "read_pages", "propose", "abstain", "propose_data", "no_change"]);
const PREFIX = "You extract scheduling limits for an automated cell-viability screening lab.";  // the page's prompt head
const MAX_BODY = 400_000;
const enc = new TextEncoder();

export default {
  async fetch(req, env) {
    const origin = req.headers.get("Origin") || "";
    if (!(env.ALLOWED_ORIGINS || "").split(",").map((s) => s.trim()).includes(origin)) return new Response("forbidden", { status: 403 });
    const cors = { "Access-Control-Allow-Origin": origin, "Access-Control-Allow-Headers": "content-type, authorization",
      "Access-Control-Allow-Methods": "GET, POST, OPTIONS", "Access-Control-Max-Age": "86400", Vary: "Origin" };
    if (req.method === "OPTIONS") return new Response(null, { status: 204, headers: cors });
    const json = (obj, status = 200) => new Response(JSON.stringify(obj), { status, headers: { ...cors, "content-type": "application/json" } });
    const path = new URL(req.url).pathname;
    if (path === "/session" && req.method === "POST") return session(req, env, json);
    const id = await verify(env, (req.headers.get("authorization") || "").replace(/^Bearer /, ""));
    if (!id) return json({ error: "session expired", code: "session" }, 401);
    if (path === "/kb" && req.method === "GET")
      return new Response(KB, { headers: { ...cors, "content-type": "application/json", "cache-control": "private, max-age=3600" } });
    if (path === "/messages" && req.method === "POST") return messages(req, env, json, id, cors);
    return json({ error: "not found" }, 404);
  },
};

async function session(req, env, json) {
  const day = new Date().toISOString().slice(0, 10), ip = req.headers.get("CF-Connecting-IP") || "unknown";
  const kIp = `ip:${day}:${ip}`, kDay = `day:${day}`;
  const [nIp, nDay] = (await Promise.all([env.COUNTS.get(kIp), env.COUNTS.get(kDay)])).map((v) => Number(v) || 0);
  if (nIp >= Number(env.RUNS_PER_IP)) return json({ error: "You have used today's live runs on this site.", code: "rate_limited" }, 429);
  if (nDay >= Number(env.RUNS_PER_DAY)) return json({ error: "This site's live runs for today are used up.", code: "rate_limited" }, 429);
  const ttl = { expirationTtl: 2 * 86400 };
  await Promise.all([env.COUNTS.put(kIp, String(nIp + 1), ttl), env.COUNTS.put(kDay, String(nDay + 1), ttl)]);
  const exp = Date.now() + Number(env.SESSION_MIN || 30) * 60_000;
  return json({ token: await sign(env, `${exp}.${crypto.randomUUID()}`), expires: exp, runs_left: Number(env.RUNS_PER_IP) - nIp - 1 });
}

async function messages(req, env, json, id, cors) {
  if (env.TURNS) {  // rate-limit binding: turns per minute per session (approximate, per location)
    const { success } = await env.TURNS.limit({ key: id });
    if (!success) return json({ error: "Too many turns too fast.", code: "rate_limited" }, 429);
  }
  const raw = await req.text();
  const reject = (error, status) => { console.log("rejected", status, error, raw.length); return json({ error }, status); };  // visible in `wrangler tail`
  if (raw.length > MAX_BODY) return reject("request too large", 413);
  let b;
  try { b = JSON.parse(raw); } catch { return reject("bad json", 400); }
  const msgs = Array.isArray(b.messages) ? b.messages : [], tools = Array.isArray(b.tools) ? b.tools : [];
  const first = msgs[0] && msgs[0].content, text = typeof first === "string" ? first : Array.isArray(first) ? first.map((c) => c.text || "").join("") : "";
  if (!msgs.length || msgs.length > 120 || msgs[0].role !== "user" || !text.startsWith(PREFIX)) return reject("not a labloop request", 400);
  if (tools.length > 8 || tools.some((t) => !TOOLS.has(t && t.name))) return reject(`tool not allowed: ${tools.map((t) => t && t.name).join(",")}`, 400);
  const body = { model: MODEL, max_tokens: Math.min(Number(b.max_tokens) || 2000, MAX_TOKENS), messages: msgs, ...(tools.length ? { tools } : {}) };
  const headers = { "content-type": "application/json", "anthropic-version": "2023-06-01" };
  if (env.ANTHROPIC_API_KEY) headers["x-api-key"] = env.ANTHROPIC_API_KEY;
  const up = await fetch(env.UPSTREAM || "https://api.anthropic.com/v1/messages", { method: "POST", headers, body: JSON.stringify(body) });
  if (!up.ok) console.log("upstream", up.status, (await up.clone().text()).slice(0, 300));  // the API's error type and message only
  return new Response(up.body, { status: up.status, headers: { ...cors, "content-type": "application/json" } });
}


// session tokens: "<expiry ms>.<run id>.<HMAC-SHA256 of the two, base64url>", signed with the TOKEN_SECRET secret
async function key(env) {
  return crypto.subtle.importKey("raw", enc.encode(env.TOKEN_SECRET), { name: "HMAC", hash: "SHA-256" }, false, ["sign", "verify"]);
}
const b64 = (buf) => btoa(String.fromCharCode(...new Uint8Array(buf))).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
const unb64 = (s) => Uint8Array.from(atob(s.replace(/-/g, "+").replace(/_/g, "/")), (c) => c.charCodeAt(0));

async function sign(env, payload) {
  return `${payload}.${b64(await crypto.subtle.sign("HMAC", await key(env), enc.encode(payload)))}`;
}

async function verify(env, token) {
  const i = token.lastIndexOf(".");
  if (i < 0) return null;
  const payload = token.slice(0, i), [exp, id] = payload.split(".");
  if (!(Number(exp) > Date.now()) || !id) return null;
  try {
    return (await crypto.subtle.verify("HMAC", await key(env), unb64(token.slice(i + 1)), enc.encode(payload))) ? id : null;
  } catch { return null; }
}
