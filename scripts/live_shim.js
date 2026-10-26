// Test-only stand-in for claude.ai's `sample` capability (see scripts/live_proxy.py). Injected into a local
// copy of the page before its own scripts; never part of the published page. Each turn goes to the local
// proxy; tool calls run here in the page, the way claude.ai runs them.
(() => {
  const PROXY = "http://127.0.0.1:8777/messages";
  const turn = async (messages, tools, system) => {
    const r = await fetch(PROXY, { method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ messages, system, max_tokens: 4000,
        tools: tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.inputSchema || { type: "object", properties: {} } })) }) });
    return r.json();
  };
  const sample = async (input, opts = {}) => {
    const msgs = typeof input === "string" ? [{ role: "user", content: input }] : input.slice();
    const tools = opts.tools || [];
    let text = "";
    for (let i = 0; i < 30; i++) {
      if (opts.signal && opts.signal.aborted) throw { code: "cancelled", message: "aborted" };
      const r = await turn(msgs, tools);
      if (r.error) throw { code: "error", message: r.error };
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
    window.__liveTurns = (window.__liveTurns || 0) + 1;
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
