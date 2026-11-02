import pathlib
import markdown
from xhtml2pdf import pisa

base = pathlib.Path(__file__).parent
md_text = (base / "combined_report.md").read_text(encoding="utf-8")

body_html = markdown.markdown(
    md_text, extensions=["tables", "fenced_code", "sane_lists", "toc"]
)

html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<style>
  @page {{ size: A4; margin: 2cm 1.8cm; }}
  body {{ font-family: Helvetica, Arial, sans-serif; font-size: 9.5pt; line-height: 1.45; color: #1a1a1a; }}
  h1 {{ font-size: 19pt; color: #0b3d63; border-bottom: 2px solid #0b3d63; padding-bottom: 6px; margin-top: 0; }}
  h2 {{ font-size: 15pt; color: #0b3d63; margin-top: 22px; border-bottom: 1px solid #ccc; padding-bottom: 3px; }}
  h3 {{ font-size: 12.5pt; color: #164a72; margin-top: 16px; }}
  h4 {{ font-size: 11pt; color: #333; margin-top: 12px; }}
  hr {{ border: none; border-top: 1px solid #bbb; margin: 18px 0; }}
  p {{ margin: 6px 0; text-align: justify; }}
  strong {{ color: #0b3d63; }}
  code {{ font-family: Courier, monospace; font-size: 8.5pt; background: #f0f0f0; padding: 1px 3px; }}
  pre {{ font-family: Courier, monospace; font-size: 8pt; background: #f4f4f4; padding: 8px; border: 0.5px solid #ccc; }}
  table {{ border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 8.5pt; }}
  th {{ background: #0b3d63; color: white; padding: 5px 6px; text-align: left; border: 0.5px solid #0b3d63; }}
  td {{ padding: 4px 6px; border: 0.5px solid #ccc; }}
  tr:nth-child(even) td {{ background: #f7f9fb; }}
  ul, ol {{ margin: 4px 0 8px 0; padding-left: 20px; }}
  li {{ margin: 3px 0; }}
  em {{ color: #444; }}
</style>
</head>
<body>
{body_html}
</body></html>
"""

(base / "combined_report.html").write_text(html, encoding="utf-8")

out_path = base.parent.parent / "REPORT_FULL.pdf"
with open(out_path, "wb") as f:
    result = pisa.CreatePDF(html, dest=f)

print("PDF error:", result.err, "-> ", out_path if result.err == 0 else "FAILED")
