// Live mode outside claude.ai (the GitHub Pages build): a stand-in for claude.ai's `sample` capability that
// sends each turn to the site's proxy (worker/), which calls Claude Haiku. Tool calls run here in the page,
// the way claude.ai runs them. build_site.py sets window.LABLOOP_LIVE = { proxy } and inlines this file; the
// artifact never carries it (claude.ai provides `sample` there).
(() => {
  const cfg = window.LABLOOP_LIVE;
  if (!cfg || !cfg.proxy || window.claude) return;
  const base = cfg.proxy.replace(/\/$/, "");
  let sess = null;  // one session per live run: {token, expires, runs_left}
  const fail = async (r) => {
    let e = {};
    try { e = await r.json(); } catch (_) { /* not json */ }
    const up = e.error && typeof e.error === "object" ? e.error : null;  // the Claude API's own error, passed through by the proxy
    throw { code: e.code || (up && up.type) || (r.status === 429 ? "rate_limited" : "error"), status: r.status,
            message: up ? `${up.type}: ${up.message}` : e.error || `proxy ${r.status}` };
  };
  const TRANSIENT = [429, 500, 502, 503, 504, 529], WAIT_MS = [2000, 6000, 15000];  // overloaded or busy: wait and send the turn again
  async function token() {
    if (sess && sess.expires - Date.now() > 60000) return sess.token;
    const r = await fetch(`${base}/session`, { method: "POST" });
    if (!r.ok) await fail(r);
    sess = await r.json();
    cfg.runsLeft = sess.runs_left;
    return sess.token;
  }
  async function call(path, init = {}, retry = true) {
    for (let k = 0; ; k++) {
      const r = await fetch(base + path, { ...init, headers: { ...(init.headers || {}), authorization: `Bearer ${await token()}` } });
      if (r.status === 401 && retry) { sess = null; return call(path, init, false); }
      if (r.ok) return r.json();
      if (k < WAIT_MS.length && TRANSIENT.includes(r.status)) { await new Promise((ok) => setTimeout(ok, WAIT_MS[k])); continue; }
      await fail(r);
    }
  }
  cfg.kb = () => call("/kb");

  const sample = async (input, opts = {}) => {
    const msgs = typeof input === "string" ? [{ role: "user", content: input }] : input.slice();
    const tools = opts.tools || [];
    const spec = tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.inputSchema || { type: "object", properties: {} } }));
    let text = "";
    for (let i = 0; i < 30; i++) {
      if (opts.signal && opts.signal.aborted) throw { code: "cancelled", message: "aborted" };
      const r = await call("/messages", { method: "POST", headers: { "content-type": "application/json" }, signal: opts.signal,
        body: JSON.stringify({ messages: msgs, max_tokens: 4000, ...(spec.length ? { tools: spec } : {}) }) });
      msgs.push({ role: "assistant", content: r.content });
      text = r.content.filter((b) => b.type === "text").map((b) => b.text).join("");
      if (opts.onText) opts.onText({ text, delta: text });
      const uses = r.content.filter((b) => b.type === "tool_use");
      if (!uses.length) break;
      const results = [];
      for (const u of uses) {
        const t = tools.find((x) => x.name === u.name);
        let out;
        try { out = t ? await t.execute(u.input) : `error: no tool ${u.name}`; } catch (e) { out = `error: ${e.message || e}`; }
        results.push({ type: "tool_result", tool_use_id: u.id, content: typeof out === "string" ? out : JSON.stringify(out) });
      }
      msgs.push({ role: "user", content: results });
    }
    return { text, truncated: false };
  };
  sample.json = async (input, opts = {}) => {
    const r = await sample(typeof input === "string" ? `${input}\nReply with the JSON only.` : input, opts);
    const m = r.text.match(/[\[{][\s\S]*[\]}]/);
    return JSON.parse(m ? m[0] : r.text);
  };
  sample.limits = async () => ({ tools: true, images: false });
  window.claude = { use: async (name) => (name === "sample" ? sample : null) };
})();
