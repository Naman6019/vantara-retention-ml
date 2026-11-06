"""Export docs/final_report.md to docs/final_report.pdf.

Uses python-markdown + xhtml2pdf (both in requirements) so the PDF can be
regenerated on any machine without a LaTeX toolchain. Display maths blocks are
rendered as monospace so equations remain visible even without a MathJax engine.
"""

import re
import sys
from pathlib import Path

import markdown
from xhtml2pdf import pisa

ROOT = Path(__file__).resolve().parent.parent


def _protect_math(text: str) -> str:
    # Display maths $$...$$ and inline $...$ -> fenced code so they survive as text.
    text = re.sub(r"\$\$(.+?)\$\$", lambda m: f"\n```\n{m.group(1).strip()}\n```\n", text, flags=re.S)
    text = re.sub(r"\$([^$\n]+?)\$", lambda m: f"`{m.group(1)}`", text)
    return text


def _resolve_images(text: str) -> str:
    """Point Markdown image links at real files so xhtml2pdf can embed them."""

    def replace(match: re.Match) -> str:
        alt, raw_path = match.group(1), match.group(2)
        if re.match(r"^[a-z]+://", raw_path, flags=re.I):
            return match.group(0)
        candidate = ROOT / raw_path
        if not candidate.exists():
            candidate = ROOT / "docs" / raw_path
        if not candidate.exists():
            return match.group(0)
        return f"![{alt}]({candidate.as_posix()})"

    return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", replace, text)


def main() -> int:
    src = ROOT / "docs" / "final_report.md"
    out = ROOT / "docs" / "final_report.pdf"
    text = _protect_math(src.read_text(encoding="utf-8"))
    text = _resolve_images(text)

    html_body = markdown.markdown(text, extensions=["tables", "fenced_code", "toc"])
    html = f"""<html><head><meta charset="utf-8"><style>
    @page {{ size: A4; margin: 1.8cm; }}
    body {{ font-family: Helvetica, Arial, sans-serif; font-size: 10.5px; color: #0f172a; }}
    h1 {{ font-size: 20px; }} h2 {{ font-size: 15px; margin-top: 16px; }} h3 {{ font-size: 12.5px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 8px 0; }}
    th, td {{ border: 1px solid #cbd5e1; padding: 4px 6px; font-size: 9.5px; }}
    th {{ background: #f1f5f9; }}
    pre {{ background: #f8fafc; border: 1px solid #e2e8f0; padding: 6px; font-size: 9px; }}
    code {{ background: #f1f5f9; }}
    img {{ max-width: 100%; }}
    </style></head><body>{html_body}</body></html>"""

    with open(out, "wb") as fh:
        status = pisa.CreatePDF(html, dest=fh, encoding="utf-8")
    if status.err:
        print("PDF generation reported errors", file=sys.stderr)
        return 1
    print(f"Wrote {out} ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
