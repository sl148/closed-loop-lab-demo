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
  let pending = null;  // one session request at a time: parallel calls share it (each session counts as a run)
  async function token() {
    if (sess && sess.expires - Date.now() > 60000) return sess.token;
    if (!pending) pending = (async () => {
      try {
        const r = await fetch(`${base}/session`, { method: "POST" });
        if (!r.ok) await fail(r);
        sess = await r.json();
        cfg.runsLeft = sess.runs_left;
        return sess.token;
      } finally { pending = null; }
    })();
    return pending;
  }
  async function send(path, init = {}, retry = true) {  // an ok Response, after retrying transient errors
    for (let k = 0; ; k++) {
      const r = await fetch(base + path, { ...init, headers: { ...(init.headers || {}), authorization: `Bearer ${await token()}` } });
      if (r.status === 401 && retry) { sess = null; return send(path, init, false); }
      if (r.ok) return r;
      if (k < WAIT_MS.length && TRANSIENT.includes(r.status)) { await new Promise((ok) => setTimeout(ok, WAIT_MS[k])); continue; }
      await fail(r);
    }
  }
  const call = async (path, init) => (await send(path, init)).json();
  cfg.kb = () => call("/kb");
  cfg.quota = async () => { try { const r = await fetch(`${base}/quota`); return r.ok ? r.json() : null; } catch (_) { return null; } };
  // one streamed turn (server-sent events of the Messages API), assembled into the message's content blocks
  async function turn(body, signal, onDelta) {
    const r = await send("/messages", { method: "POST", headers: { "content-type": "application/json" }, signal, body: JSON.stringify({ ...body, stream: true }) });
    const rd = r.body.getReader(), dec = new TextDecoder(), blocks = [];
    let buf = "";
    const on = (ev) => {
      if (ev.type === "content_block_start") blocks[ev.index] = { ...ev.content_block, ...(ev.content_block.type === "tool_use" ? { _json: "" } : {}) };
      else if (ev.type === "content_block_delta") {
        const b = blocks[ev.index], d = ev.delta;
        if (d.type === "text_delta") { b.text = (b.text || "") + d.text; onDelta(); }
        else if (d.type === "input_json_delta") b._json += d.partial_json;
        else if (d.type === "thinking_delta") { b.thinking = (b.thinking || "") + d.thinking; onDelta(); }
        else if (d.type === "signature_delta") b.signature = d.signature;  // a thinking block goes back to the API with its signature
      } else if (ev.type === "content_block_stop") {
        const b = blocks[ev.index];
        if (b && b.type === "tool_use") { b.input = b._json ? JSON.parse(b._json) : {}; delete b._json; }
      } else if (ev.type === "error") throw { code: ev.error.type, message: `${ev.error.type}: ${ev.error.message}` };
    };
    for (;;) {
      const { value, done } = await rd.read();
      if (done) break;
      buf += dec.decode(value, { stream: true }).replace(/\r\n/g, "\n");
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const data = buf.slice(0, i).split("\n").filter((l) => l.startsWith("data:")).map((l) => l.slice(5).trim()).join("");
        buf = buf.slice(i + 2);
        if (data) on(JSON.parse(data));
      }
      if (blocks.length) onDelta(blocks);
    }
    return blocks.filter(Boolean);
  }

  const sample = async (input, opts = {}) => {
    const msgs = typeof input === "string" ? [{ role: "user", content: input }] : input.slice();
    const tools = opts.tools || [];
    const spec = tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.inputSchema || { type: "object", properties: {} } }));
    let text = "";
    for (let i = 0; i < 30; i++) {
      if (opts.signal && opts.signal.aborted) throw { code: "cancelled", message: "aborted" };
      let content = [];
      const t0 = Date.now();
      const shown = () => {  // before any text: how long it has been thinking, so a long think does not look stuck
        text = content.filter((b) => b && b.type === "text").map((b) => b.text || "").join("");
        const words = content.reduce((a, b) => a + (b && b.thinking ? b.thinking.split(/\s+/).length : 0), 0);
        if (opts.onText) opts.onText({ text: text || `Thinking… ${Math.round((Date.now() - t0) / 1000)} s${words ? `, ${words} words` : ""}`, delta: text });
      };
      const tick = setInterval(shown, 1000);
      try {
        for (let k = 0; ; k++) {  // an overload that arrives mid-stream: send the turn again
          try {
            content = await turn({ messages: msgs, max_tokens: 4000, ...(spec.length ? { tools: spec } : {}) }, opts.signal, (b) => { if (b) content = b; shown(); });
            break;
          } catch (e) {
            if (e && e.name === "AbortError") throw { code: "cancelled", message: "stopped" };
            if (k < WAIT_MS.length && e && ["overloaded_error", "api_error", "rate_limit_error"].includes(e.code)) { await new Promise((ok) => setTimeout(ok, WAIT_MS[k])); continue; }
            throw e;
          }
        }
      } finally { clearInterval(tick); }
      shown();
      msgs.push({ role: "assistant", content });
      const uses = content.filter((b) => b.type === "tool_use");
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
