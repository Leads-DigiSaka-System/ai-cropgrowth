"""Method docs search for the agent's search_method_docs tool.

Indexes this repository's own documentation (README sections, module /
class / function docstrings, notebook markdown) and ranks chunks with BM25 —
keyword search, no embeddings or network, so it builds in well under a second.
Each result cites where it came from (file + section / function).
"""
from __future__ import annotations

import ast
import json
import math
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_TOKEN = re.compile(r"[a-z0-9_]+")
_STOP = set("a an and are as at be by can do does for from how in is it of on or that the this "
            "to what when where which who why with you your i me my we our".split())
CHUNK_CHARS = 1200


def tokenize(text: str) -> list[str]:
    out = []
    for t in _TOKEN.findall(text.lower()):               # NDVI_15 -> ndvi_15, ndvi, 15
        if t in _STOP:
            continue
        out.append(t)
        if "_" in t:
            out += [p for p in t.split("_") if p and p not in _STOP]
    return out


def _split(chunk: dict) -> list[dict]:
    """Long texts -> ~CHUNK_CHARS pieces on paragraph (blank-line) boundaries,
    so one long docstring doesn't drown its own relevant section."""
    text = chunk["text"]
    if len(text) <= CHUNK_CHARS:
        return [chunk]
    out, buf = [], ""
    for para in re.split(r"\n\s*\n", text):
        if buf and len(buf) + len(para) > CHUNK_CHARS:
            out.append(buf); buf = ""
        buf = f"{buf}\n\n{para}" if buf else para
    if buf:
        out.append(buf)
    return [{"source": chunk["source"] + (f" [{i + 1}/{len(out)}]" if len(out) > 1 else ""), "text": t}
            for i, t in enumerate(out)]


def _markdown_chunks(path: Path, text: str):
    title, buf, out = path.name, [], []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            out.append({"source": f"{path.relative_to(ROOT)} › {title}", "text": body})
    for line in text.splitlines():
        m = re.match(r"^#{1,4}\s+(.*)", line)
        if m:
            flush(); title, buf = m.group(1).strip(), []
        else:
            buf.append(line)
    flush()
    return out


def _python_chunks(path: Path, text: str):
    rel = path.relative_to(ROOT)
    tree = ast.parse(text)
    out = []
    doc = ast.get_docstring(tree)
    if doc:
        out.append({"source": f"{rel} (module)", "text": doc})
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith("_"):
            d = ast.get_docstring(node)
            if d:
                out.append({"source": f"{rel}:{node.lineno} {node.name}", "text": d})
    # module-level config blocks with their comments (phenology.DEFAULTS, QC codes, stage classes)
    for m in re.finditer(r"^([A-Z][A-Z0-9_]{3,}\s*=\s*(?:dict\(|\{|\[)(?:.|\n)*?^[)\]}])", text, re.M):
        out.append({"source": f"{rel} {m.group(1).split('=')[0].strip()}", "text": m.group(1)})
    qc = [l for l in text.splitlines() if re.match(r"^QC_[A-Z_]+\s*=\s*\d+", l)]
    if qc:
        out.append({"source": f"{rel} QC codes", "text": "\n".join(qc)})
    return out


def _notebook_chunks(path: Path, text: str):
    nb = json.loads(text)
    out = []
    for i, c in enumerate(nb.get("cells", [])):
        if c.get("cell_type") == "markdown":
            body = "".join(c.get("source", [])).strip()
            if body:
                out.append({"source": f"{path.relative_to(ROOT)} cell {i}", "text": body})
    return out


class DocsIndex:
    def __init__(self, chunks: list[dict], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.docs = [tokenize(c["source"] + " " + c["text"]) for c in chunks]
        self.k1, self.b = k1, b
        self.avg = sum(map(len, self.docs)) / max(1, len(self.docs))
        self.tf = [Counter(d) for d in self.docs]
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    @classmethod
    def build(cls, root: Path | str = ROOT) -> "DocsIndex":
        root = Path(root)
        chunks = []
        for p in sorted(root.glob("*.md")) + sorted(root.glob("*/*.md")):
            chunks += _markdown_chunks(p, p.read_text(encoding="utf-8"))
        for p in sorted((root / "cropgrowth_agent").glob("*.py")):
            chunks += _python_chunks(p, p.read_text(encoding="utf-8"))
        for p in sorted((root / "notebooks").glob("*.ipynb")):
            try:
                chunks += _notebook_chunks(p, p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return cls([piece for c in chunks for piece in _split(c)])

    def search(self, query: str, k: int = 5, max_chars: int = 2500) -> list[dict]:
        q = tokenize(query)
        scores = []
        for i, (d, tf) in enumerate(zip(self.docs, self.tf)):
            s = 0.0
            for t in q:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(d) / self.avg))
            if s > 0:
                scores.append((s, i))
        scores.sort(reverse=True)
        return [{"source": self.chunks[i]["source"], "score": round(s, 2),
                 "text": self.chunks[i]["text"][:max_chars]} for s, i in scores[:max(1, int(k))]]
