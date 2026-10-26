"""One agent conversation loop, shared by the constraint agent's phases and the in-loop reviews.

A phase is one Tool Runner conversation: a slot (what the trace calls this piece of work), a prompt,
and the subset of tools it may call. The session keeps token usage and the model's text per turn.
"""

from __future__ import annotations

PRICES = {  # $ per MTok
    "claude-opus-5-5": {"input": 4.0, "output": 20.0, "cache_read": 0.20, "cache_write": 5.0},
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "cache_read": 0.01, "cache_write": 0.125},  # prompts <= 100K
}


class AgentSession:
    def __init__(self, box, tools: dict, system: str, model: str, effort: str = "medium", client=None, max_tokens: int = 16000):
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.box, self.tools, self.system, self.model, self.effort, self.client = box, tools, system, model, effort, client
        self.max_tokens = max_tokens
        self.usage = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "turns": 0}
        self.turns: list[dict] = []

    def phase(self, slot: str, prompt: str, names: list[str], turns_max: int, decide: bool = True) -> None:
        """Run one conversation. With `decide`, a slot that ends without propose/abstain is abstained."""
        box = self.box
        box.slot = slot
        # a declined Opus turn reruns on a fallback model; Haiku has no server-side fallback
        extra = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"} if "opus" in self.model else {}
        runner = self.client.beta.messages.tool_runner(
            model=self.model, max_tokens=self.max_tokens, system=self.system, tools=[self.tools[n] for n in names],
            max_iterations=turns_max, output_config={"effort": self.effort}, cache_control={"type": "ephemeral"},
            **extra, messages=[{"role": "user", "content": prompt}])
        for i, msg in enumerate(runner):
            u = msg.usage
            self.usage["input"] += u.input_tokens
            self.usage["output"] += u.output_tokens
            self.usage["cache_read"] += getattr(u, "cache_read_input_tokens", 0) or 0
            self.usage["cache_write"] += getattr(u, "cache_creation_input_tokens", 0) or 0
            self.usage["turns"] += 1
            self.turns.append({"slot": slot, "stop": msg.stop_reason,
                               "text": " ".join(b.text for b in msg.content if b.type == "text")[:2000]})
            if msg.stop_reason == "refusal" or i + 1 >= turns_max:
                if decide:
                    box.abstained.setdefault(slot, f"stopped ({msg.stop_reason}, turn {i + 1})")
                break
        if decide and slot not in box.proposals and slot not in box.abstained:
            box.abstained[slot] = "no proposal"

    def priced_usage(self) -> dict:
        price = PRICES.get(self.model, PRICES["claude-opus-5-5"])
        return {**self.usage, "usd": round(sum(self.usage[k] * price[k] for k in price) / 1e6, 4)}
