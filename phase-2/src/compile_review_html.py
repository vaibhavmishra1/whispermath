from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def group_by_latex(rows: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["latex"]].append(row)
    return list(grouped.items())


def mathjax_escape(latex: str) -> str:
    return latex.replace("\\", "\\\\")


def build_html(rows: list[dict[str, Any]], title: str) -> str:
    groups = group_by_latex(rows)
    cards = []

    for index, (latex, variants) in enumerate(groups, start=1):
        source_row = variants[0].get("source_row_index", "")
        variant_rows = []
        for variant in variants:
            variant_rows.append(
                f"""
                <tr>
                  <td><span class="style">{html.escape(str(variant.get("style", "")))}</span></td>
                  <td>{html.escape(str(variant.get("spoken", "")))}</td>
                  <td class="notes">{html.escape(str(variant.get("notes", "")))}</td>
                </tr>
                """
            )

        cards.append(
            f"""
            <section class="card">
              <div class="card-header">
                <div>
                  <p class="eyebrow">Formula {index}</p>
                  <h2>Source row {html.escape(str(source_row))}</h2>
                </div>
              </div>

              <div class="rendered">
                \\[
                {mathjax_escape(latex)}
                \\]
              </div>

              <details>
                <summary>Raw LaTeX</summary>
                <pre>{html.escape(latex)}</pre>
              </details>

              <table>
                <thead>
                  <tr>
                    <th>Style</th>
                    <th>Spoken variant</th>
                    <th>Notes</th>
                  </tr>
                </thead>
                <tbody>
                  {''.join(variant_rows)}
                </tbody>
              </table>
            </section>
            """
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <script>
    window.MathJax = {{
      tex: {{
        inlineMath: [['\\\\(', '\\\\)']],
        displayMath: [['\\\\[', '\\\\]']],
        processEscapes: true
      }},
      svg: {{ fontCache: 'global' }}
    }};
  </script>
  <script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f8fb;
      --panel: #ffffff;
      --text: #172033;
      --muted: #667085;
      --line: #d9e0ea;
      --accent: #0f766e;
      --accent-soft: #e6f4f1;
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.5;
    }}

    main {{
      width: min(1120px, calc(100% - 32px));
      margin: 32px auto 56px;
    }}

    header {{
      margin-bottom: 24px;
    }}

    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
      line-height: 1.15;
      letter-spacing: 0;
    }}

    .summary {{
      color: var(--muted);
      margin: 0;
      max-width: 760px;
    }}

    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 20px;
      margin: 18px 0;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
    }}

    .card-header {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
    }}

    .eyebrow {{
      margin: 0 0 2px;
      color: var(--accent);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0;
    }}

    h2 {{
      margin: 0;
      font-size: 16px;
      font-weight: 650;
    }}

    .rendered {{
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfe;
      padding: 16px;
      margin: 16px 0;
      min-height: 72px;
    }}

    details {{
      margin: 12px 0 16px;
    }}

    summary {{
      cursor: pointer;
      color: var(--muted);
      font-size: 14px;
    }}

    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      background: #101828;
      color: #f8fafc;
      border-radius: 8px;
      padding: 12px;
      overflow-x: auto;
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
    }}

    th, td {{
      text-align: left;
      vertical-align: top;
      border-top: 1px solid var(--line);
      padding: 10px 8px;
      font-size: 14px;
      overflow-wrap: anywhere;
    }}

    th {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0;
    }}

    th:first-child,
    td:first-child {{
      width: 120px;
    }}

    th:last-child,
    td:last-child {{
      width: 260px;
    }}

    .style {{
      display: inline-block;
      background: var(--accent-soft);
      color: var(--accent);
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 12px;
      font-weight: 650;
    }}

    .notes {{
      color: var(--muted);
    }}

    @media (max-width: 760px) {{
      main {{
        width: min(100% - 20px, 1120px);
        margin-top: 20px;
      }}

      table, thead, tbody, tr, th, td {{
        display: block;
      }}

      thead {{
        display: none;
      }}

      tr {{
        border-top: 1px solid var(--line);
        padding: 10px 0;
      }}

      td {{
        border-top: 0;
        padding: 4px 0;
      }}

      th:first-child,
      td:first-child,
      th:last-child,
      td:last-child {{
        width: auto;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>{html.escape(title)}</h1>
      <p class="summary">Rendered review of {len(groups)} unique LaTeX formulas and {len(rows)} spoken variants. Use this to decide whether a formula is actually suitable for spoken-math training.</p>
    </header>
    {''.join(cards)}
  </main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile generated spoken/LaTeX pairs into a MathJax HTML review page.")
    parser.add_argument("--input", type=Path, default=Path("data/generated/spoken_latex_pairs.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/generated/review.html"))
    parser.add_argument("--title", default="WhisperMath Phase 2 Pair Review")
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    html_text = build_html(rows, args.title)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_text, encoding="utf-8")
    print(f"Wrote review page: {args.output}")


if __name__ == "__main__":
    main()
