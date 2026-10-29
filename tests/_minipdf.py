"""Dependency-free PDF writer for tests and the sample corpus.

A page is a str (one line of text) or a dict {"lines": [...], "ops": "<raw PDF drawing ops>"}.
Standard Helvetica only, so text is Latin-script (no CJK font is embedded).
"""

from __future__ import annotations


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def label(x: int, y: int, text: str, size: int = 10) -> str:
    return f"BT /F1 {size} Tf {x} {y} Td ({_esc(text)}) Tj ET\n"


def make_pdf(pages: list) -> bytes:
    objs: list[bytes] = [b"<< /Type /Catalog /Pages 2 0 R >>", b"", b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for i, page in enumerate(pages):
        spec = {"lines": [page]} if isinstance(page, str) else page
        page_id, content_id = 4 + 2 * i, 5 + 2 * i
        kids.append(f"{page_id} 0 R")
        ops = ""
        for n, line in enumerate(spec.get("lines", [])):
            ops += label(72, 720 - 18 * n, line, 12)
        ops += spec.get("ops", "")
        stream = ops.encode("latin-1")
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_id} 0 R "
            f"/Resources << /Font << /F1 3 0 R >> >> >>".encode()
        )
        objs.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(pages)} >>".encode()

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for n, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def bar_chart(title: str, bars: list[tuple[str, int]], x0: int = 100, y0: int = 400, unit: int = 3) -> str:
    """Vector bar chart: axes, coloured bars, value + category labels. Returns drawing ops."""
    ops = label(x0, y0 + 260, title, 14)
    ops += f"0.2 w {x0} {y0} m {x0 + 60 * len(bars) + 20} {y0} l S {x0} {y0} m {x0} {y0 + 240} l S\n"
    colours = ["0.20 0.45 0.80", "0.85 0.35 0.25", "0.30 0.65 0.40", "0.90 0.70 0.20"]
    for i, (name, value) in enumerate(bars):
        x = x0 + 20 + 60 * i
        ops += f"{colours[i % len(colours)]} rg {x} {y0} 40 {value * unit} re f 0 g\n"
        ops += label(x + 5, y0 + value * unit + 5, str(value)) + label(x, y0 - 14, name)
    return ops
