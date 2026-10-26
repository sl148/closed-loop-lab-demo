"""Document archive for the constraint agent (M9): ingest, search, read, verify.

The archive is a folder of documents (manuals, datasheets, protocols, papers) listed in a
manifest. Ingest splits each into page-aware chunks; search is BM25 over the chunks (exact terms
and numbers matter in manuals: "37 °C", "30 min"); the agent reads whole pages around a hit and
every value it proposes must quote a page verbatim (`verify_quote`).

    data/kb/manifest.yaml   - id, title, url, version, date, file (committed)
    data/kb/docs/*          - the documents (not committed)
    data/kb/chunks.jsonl    - built by `ingest`
"""

from __future__ import annotations

import html
import json
import math
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

KB = Path("data/kb")
WORDS_PER_CHUNK, OVERLAP = 220, 40
STOP = set("a an and are as at be by for from has have in is it its of on or that the this to was were will with".split())


@dataclass
class Chunk:
    id: str
    doc: str
    page: int
    text: str


def pages_of(path: Path) -> list[str]:
    """Text per page. PDFs page by page; text and HTML files split on form feeds."""
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader  # only needed when the archive holds PDFs

        return [p.extract_text() or "" for p in PdfReader(str(path)).pages]
    raw = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() in (".html", ".htm"):
        raw = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw)
        raw = html.unescape(re.sub(r"<[^>]+>", " ", raw))
    return raw.split("\f")


def chunks_of(doc: str, pages: list[str]) -> list[Chunk]:
    out = []
    for pno, text in enumerate(pages, 1):
        words = text.split()
        step = WORDS_PER_CHUNK - OVERLAP
        for i in range(0, max(1, len(words)), step):
            piece = " ".join(words[i:i + WORDS_PER_CHUNK])
            if piece:
                out.append(Chunk(f"{doc}:p{pno}:{i // step}", doc, pno, piece))
            if i + WORDS_PER_CHUNK >= len(words):
                break
    return out


def manifest(kb: Path = KB) -> list[dict]:
    return yaml.safe_load((kb / "manifest.yaml").read_text())["docs"]


def ingest(kb: Path = KB) -> int:
    """(Re)builds chunks.jsonl and pages.json from the manifest's documents."""
    chunks, pages = [], {}
    for d in manifest(kb):
        ps = pages_of(kb / "docs" / d["file"])
        pages[d["id"]] = ps
        chunks += chunks_of(d["id"], ps)
    (kb / "chunks.jsonl").write_text("".join(json.dumps(asdict(c)) + "\n" for c in chunks))
    (kb / "pages.json").write_text(json.dumps(pages))
    return len(chunks)


def ingest_one(doc_id: str, kb: Path = KB) -> int:
    """Adds (or refreshes) one manifest document in an existing chunks.jsonl / pages.json."""
    d = next(x for x in manifest(kb) if x["id"] == doc_id)
    ps = pages_of(kb / "docs" / d["file"])
    pages = json.loads((kb / "pages.json").read_text())
    pages[doc_id] = ps
    rows = [json.loads(line) for line in (kb / "chunks.jsonl").read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r["doc"] != doc_id] + [asdict(c) for c in chunks_of(doc_id, ps)]
    (kb / "chunks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (kb / "pages.json").write_text(json.dumps(pages))
    return len(rows)


def snippet(text: str, query: str, n: int = 700) -> str:
    """The n-char stretch of a chunk or page that holds the most distinct query words (not its first n).
    LABLOOP_SNIPPET=head restores the old snippet, the chunk's first n chars (for comparing agent runs)."""
    text = re.sub(r"\s+", " ", text)
    if os.environ.get("LABLOOP_SNIPPET") == "head":
        return text[:n].strip() + ("…" if len(text) > n else "")
    words = {w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 2}
    low, best = text.lower(), (0, 0)
    for w in words:
        for m in re.finditer(re.escape(w), low):
            a = max(0, m.start() - n // 3)
            best = max(best, (sum(x in low[a:a + n] for x in words), -a))
    a = -best[1]
    return ("…" if a else "") + text[a:a + n].strip() + ("…" if a + n < len(text) else "")


def tokens(text: str) -> list[str]:
    text = text.lower().replace("°c", " degc ").replace("µ", "u")
    return [t for t in re.findall(r"[a-z]+|\d+(?:\.\d+)?", text) if t not in STOP]


class Archive:
    def __init__(self, kb: Path = KB, k1: float = 1.5, b: float = 0.75):
        self.kb = Path(kb)
        self.chunks = [Chunk(**json.loads(l)) for l in (self.kb / "chunks.jsonl").read_text().splitlines()]
        self.pages: dict[str, list[str]] = json.loads((self.kb / "pages.json").read_text())
        self.docs = {d["id"]: d for d in manifest(self.kb)}
        self.k1, self.b = k1, b
        self.tf = [Counter(tokens(c.text)) for c in self.chunks]
        self.len = [sum(t.values()) for t in self.tf]
        self.avg = sum(self.len) / max(1, len(self.len))
        df = Counter(w for t in self.tf for w in t)
        n = len(self.chunks)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}

    def search(self, query: str, k: int = 5, doc: str | None = None) -> list[dict]:
        q = tokens(query)
        scored = []
        for i, (c, tf) in enumerate(zip(self.chunks, self.tf)):
            if doc and c.doc != doc:
                continue
            s = 0.0
            for w in q:
                f = tf.get(w, 0)
                if f:
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
            if s > 0:
                scored.append((s, c))
        scored.sort(key=lambda x: -x[0])
        return [{"chunk": c.id, "doc": c.doc, "title": self.docs[c.doc].get("title", ""), "page": c.page,
                 "score": round(s, 3), "text": c.text} for s, c in scored[:k]]

    def page_counts(self) -> dict[str, int]:
        return {d: len(ps) for d, ps in self.pages.items()}

    def read_pages(self, doc: str, first: int, last: int | None = None) -> list[dict]:
        ps = self.pages[doc]
        last = min(len(ps), last or first)
        return [{"doc": doc, "page": p, "text": ps[p - 1]} for p in range(max(1, first), last + 1)]

    def verify_quote(self, doc: str, page: int, quote: str) -> bool:
        """The quote appears on that page, ignoring case, whitespace and hyphenated line breaks."""
        norm = lambda s: re.sub(r"\s+", " ", re.sub(r"-\s*\n\s*", "", s)).strip().lower()
        if doc not in self.pages or not 1 <= page <= len(self.pages[doc]) or len(quote.strip()) < 12:
            return False
        return norm(quote) in norm(self.pages[doc][page - 1])
