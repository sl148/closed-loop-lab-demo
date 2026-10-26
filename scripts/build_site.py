"""Build the standalone site (GitHub Pages, served from docs/) from the artifact page in web/.

    python scripts/build_site.py

The artifact host wraps web/index.html in a document skeleton; here the same skeleton is written
out, so the page runs as a plain static site. Live mode needs Claude inside claude.ai (the `sample`
capability), so the archive text it searches (data/kb.json, extracted from the source documents) is
left out: on this site the switch stays on Recorded.
"""

from __future__ import annotations

import shutil
from pathlib import Path

SRC, OUT = Path("web"), Path("docs")
SKIP = {"kb.json", "agent_tool_cases.json", "lab_data.xlsx"}  # archive text and snippets (third-party documents), used only by Live mode; the page builds its own workbook
HEAD = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
        '<style>:root{color-scheme:light;padding:env(safe-area-inset-top,0px) 0 env(safe-area-inset-bottom,0px)}'
        'body{margin:0;font:14px system-ui,sans-serif}img{max-width:100%}[hidden]{display:none!important}</style></head><body>')


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "data").mkdir(parents=True)
    (OUT / "index.html").write_text(HEAD + (SRC / "index.html").read_text(encoding="utf-8") + "</body></html>", encoding="utf-8")
    n = 0
    for f in sorted((SRC / "data").iterdir()):
        if f.is_file() and f.name not in SKIP:
            shutil.copy2(f, OUT / "data" / f.name)
            n += 1
    (OUT / ".nojekyll").write_text("")
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e6
    print(f"docs/: index.html + {n} data files, {size:.1f} MB")


if __name__ == "__main__":
    main()
