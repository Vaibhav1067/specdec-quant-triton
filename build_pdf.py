"""Render REPORT.md (+ figures) to REPORT.pdf via HTML and headless Edge/Chrome."""
import pathlib, re, shutil, subprocess, sys
import markdown

root = pathlib.Path(__file__).resolve().parent
md = (root / "REPORT.md").read_text(encoding="utf-8")
md = md.replace("# INT8/INT4 Triton Kernels under Speculative-Decoding Verification Shapes",
                "# INT8/INT4 Triton Kernels under Speculative-Decoding Verification Shapes\n\n"
                "<p class='author'>Vaibhav Kushwaha &middot; github.com/Vaibhav1067/specdec-quant-triton</p>", 1)
body = markdown.markdown(md, extensions=["tables", "fenced_code"])
css = """
@page { size: A4; margin: 14mm 13mm; }
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 9.6pt; line-height: 1.42; color: #1a1a1a; }
h1 { font-size: 17pt; margin: 0 0 2px; }
.author { color: #555; margin: 0 0 10px; }
h2 { font-size: 12.5pt; border-bottom: 1px solid #ccc; padding-bottom: 2px; margin-top: 16px; page-break-after: avoid; }
table { border-collapse: collapse; margin: 6px 0 8px; font-size: 8.4pt; page-break-inside: avoid; }
th, td { border: 1px solid #c8c8c8; padding: 2px 6px; text-align: left; }
th { background: #f0f2f5; }
img { width: 100%; page-break-inside: avoid; margin: 4px 0; }
p:has(img) { page-break-inside: avoid; }
code { font-family: Consolas, monospace; font-size: 8.6pt; background: #f4f4f4; padding: 0 2px; }
pre { background: #f4f4f4; padding: 6px; font-size: 8.4pt; }
hr { border: 0; border-top: 1px solid #ddd; }
li { margin: 2px 0; }
"""
html = f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{css}</style></head><body>{body}</body></html>"
out_html = root / "REPORT.html"
out_html.write_text(html, encoding="utf-8")

browsers = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            shutil.which("chromium") or "", shutil.which("google-chrome") or ""]
exe = next((b for b in browsers if b and pathlib.Path(b).exists()), None)
if not exe:
    sys.exit("No Edge/Chrome found")
subprocess.run([exe, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={root / 'REPORT.pdf'}", out_html.as_uri()], check=True, timeout=120)
out_html.unlink()
print("wrote", root / "REPORT.pdf")
