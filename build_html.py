"""Convert markdown article to self-contained HTML with base64-embedded images."""

import re
import base64
import os
from pathlib import Path

BASE_DIR = Path(r"d:\Coding_Workspace\Blog_automation")
MD_PATH = BASE_DIR / "published" / "sliding-window-attention.md"
HTML_PATH = BASE_DIR / "published" / "sliding-window-attention.html"
FIGURES_DIR = BASE_DIR / "figures" / "output"


def read_image_base64(img_path: str) -> str:
    """Read an image file and return base64-encoded string."""
    # Resolve relative path from the markdown file location
    if img_path.startswith("../"):
        full_path = BASE_DIR / img_path.replace("../", "")
    elif img_path.startswith("figures/"):
        full_path = BASE_DIR / img_path
    else:
        full_path = Path(img_path)

    if not full_path.exists():
        print(f"  WARNING: Image not found: {full_path}")
        return ""

    with open(full_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    size_mb = full_path.stat().st_size / (1024 * 1024)
    print(f"  Embedded: {full_path.name} ({size_mb:.2f} MB)")
    return data


def escape_html(text: str) -> str:
    """Escape HTML special characters, but preserve already-converted HTML."""
    return text


def convert_inline(text: str) -> str:
    """Convert inline markdown formatting to HTML."""
    # Bold: **text** -> <strong>text</strong>
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    # Italic: *text* -> <em>text</em>
    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<em>\1</em>', text)
    # Inline code: `code` -> <code>code</code>
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    # Links: [text](url) -> <a href="url">text</a>
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', text)
    return text


def convert_markdown_to_html(md_text: str) -> str:
    """Convert markdown text to HTML body content."""
    lines = md_text.split('\n')
    html_parts = []
    i = 0
    image_count = 0

    while i < len(lines):
        line = lines[i]

        # Skip empty lines
        if line.strip() == '':
            i += 1
            continue

        # Horizontal rule
        if line.strip() == '---':
            html_parts.append('<hr>')
            i += 1
            continue

        # H1 title
        if line.startswith('# ') and not line.startswith('## '):
            title = line[2:].strip()
            html_parts.append(f'<h1>{convert_inline(title)}</h1>')
            i += 1
            continue

        # H2 heading
        if line.startswith('## ') and not line.startswith('### '):
            heading = line[3:].strip()
            # Special class for Further Reading
            if 'Further Reading' in heading:
                html_parts.append(f'<div class="further-reading">')
                html_parts.append(f'<h2>{convert_inline(heading)}</h2>')
                # Collect content until next --- or end
                i += 1
                while i < len(lines) and lines[i].strip() != '---':
                    subline = lines[i]
                    if subline.strip() == '':
                        i += 1
                        continue
                    if subline.startswith('- '):
                        html_parts.append(f'<p>{convert_inline(subline[2:].strip())}</p>')
                    else:
                        html_parts.append(f'<p>{convert_inline(subline.strip())}</p>')
                    i += 1
                html_parts.append('</div>')
                continue
            # Special class for Summary
            html_parts.append(f'<h2>{convert_inline(heading)}</h2>')
            i += 1
            continue

        # H3 heading
        if line.startswith('### '):
            heading = line[4:].strip()
            html_parts.append(f'<h3>{convert_inline(heading)}</h3>')
            i += 1
            continue

        # Code block
        if line.strip().startswith('```'):
            lang = line.strip().replace('```', '').strip()
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                # Escape HTML in code
                code_line = lines[i].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                code_lines.append(code_line)
                i += 1
            i += 1  # skip closing ```
            code_content = '\n'.join(code_lines)
            html_parts.append(f'<pre><code>{code_content}</code></pre>')
            continue

        # Image: ![caption](path)
        img_match = re.match(r'^!\[(.+?)\]\((.+?)\)\s*$', line.strip())
        if img_match:
            caption = img_match.group(1)
            img_path = img_match.group(2)
            b64 = read_image_base64(img_path)
            if b64:
                image_count += 1
                html_parts.append(f'<figure>')
                html_parts.append(f'    <img src="data:image/png;base64,{b64}" alt="{caption}">')
                html_parts.append(f'    <figcaption>{caption}</figcaption>')
                html_parts.append(f'</figure>')
            else:
                html_parts.append(f'<p><em>[Missing image: {img_path}]</em></p>')
            i += 1
            continue

        # Blockquote
        if line.startswith('> '):
            bq_lines = []
            while i < len(lines) and (lines[i].startswith('> ') or lines[i].strip() == ''):
                if lines[i].strip() == '':
                    # Check if next line continues the blockquote
                    if i + 1 < len(lines) and lines[i + 1].startswith('> '):
                        bq_lines.append('')
                        i += 1
                        continue
                    else:
                        break
                bq_lines.append(lines[i][2:])
                i += 1
            bq_text = '\n'.join(bq_lines)
            # Split into paragraphs
            bq_paras = [p.strip() for p in bq_text.split('\n\n') if p.strip()]
            html_parts.append('<blockquote>')
            for bp in bq_paras:
                html_parts.append(f'<p>{convert_inline(bp)}</p>')
            html_parts.append('</blockquote>')
            continue

        # Unordered list
        if line.startswith('- '):
            html_parts.append('<ul>')
            while i < len(lines) and lines[i].startswith('- '):
                item = lines[i][2:].strip()
                html_parts.append(f'<li>{convert_inline(item)}</li>')
                i += 1
            html_parts.append('</ul>')
            continue

        # Ordered list
        ol_match = re.match(r'^(\d+)\.\s+', line)
        if ol_match:
            html_parts.append('<ol>')
            while i < len(lines) and re.match(r'^\d+\.\s+', lines[i]):
                item = re.sub(r'^\d+\.\s+', '', lines[i]).strip()
                html_parts.append(f'<li>{convert_inline(item)}</li>')
                i += 1
            html_parts.append('</ol>')
            continue

        # CTA (italic line at the very end)
        if line.startswith('*') and line.endswith('*') and 'subscribing' in line.lower():
            cta_text = line.strip('*').strip()
            html_parts.append(f'<div class="cta"><p><em>{cta_text}</em></p></div>')
            i += 1
            continue

        # Subtitle (italic line right after title)
        if line.startswith('*') and line.endswith('*') and i < 5:
            subtitle = line.strip('*').strip()
            html_parts.append(f'<p class="subtitle">{subtitle}</p>')
            i += 1
            continue

        # Regular paragraph
        para_lines = []
        while i < len(lines) and lines[i].strip() != '' and not lines[i].startswith('#') \
                and not lines[i].startswith('```') and not lines[i].startswith('- ') \
                and not lines[i].startswith('> ') and not re.match(r'^\d+\.\s+', lines[i]) \
                and not re.match(r'^!\[', lines[i].strip()) \
                and lines[i].strip() != '---':
            para_lines.append(lines[i])
            i += 1
        if para_lines:
            para_text = ' '.join(para_lines)
            html_parts.append(f'<p>{convert_inline(para_text)}</p>')
            continue

        i += 1

    print(f"\n  Total images embedded: {image_count}")
    return '\n'.join(html_parts), image_count


def build_html(body_content: str, title: str) -> str:
    """Wrap content in the full HTML template."""
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 18px;
            line-height: 1.75;
            color: #1a1a1a;
            background: #ffffff;
            max-width: 760px;
            margin: 0 auto;
            padding: 40px 24px 80px;
        }}

        h1 {{
            font-size: 2.2em;
            font-weight: 700;
            line-height: 1.2;
            margin-bottom: 8px;
            color: #111111;
            letter-spacing: -0.02em;
        }}

        .subtitle {{
            font-size: 1.2em;
            color: #555555;
            margin-bottom: 32px;
            line-height: 1.5;
        }}

        h2 {{
            font-size: 1.6em;
            font-weight: 700;
            margin-top: 48px;
            margin-bottom: 16px;
            color: #111111;
            letter-spacing: -0.01em;
        }}

        h3 {{
            font-size: 1.25em;
            font-weight: 600;
            margin-top: 32px;
            margin-bottom: 12px;
            color: #222222;
        }}

        p {{
            margin-bottom: 16px;
        }}

        strong {{
            font-weight: 600;
            color: #111111;
        }}

        a {{
            color: #2563eb;
            text-decoration: underline;
            text-underline-offset: 2px;
        }}

        ul, ol {{
            margin-bottom: 16px;
            padding-left: 28px;
        }}

        li {{
            margin-bottom: 8px;
        }}

        figure {{
            margin: 32px 0;
            text-align: center;
        }}

        figure img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #e5e5e5;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }}

        figcaption {{
            font-size: 0.88em;
            color: #666666;
            margin-top: 12px;
            line-height: 1.5;
            text-align: center;
            font-style: italic;
        }}

        blockquote {{
            background: #f0f7ff;
            border-left: 4px solid #2563eb;
            padding: 20px 24px;
            margin: 24px 0;
            border-radius: 0 8px 8px 0;
            font-size: 0.95em;
        }}

        blockquote p {{
            margin-bottom: 8px;
        }}

        blockquote p:last-child {{
            margin-bottom: 0;
        }}

        pre {{
            background: #1e1e2e;
            color: #cdd6f4;
            padding: 24px;
            border-radius: 8px;
            overflow-x: auto;
            margin: 24px 0;
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
            font-size: 0.85em;
            line-height: 1.6;
        }}

        code {{
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
            font-size: 0.88em;
        }}

        p code, li code {{
            background: #f3f4f6;
            padding: 2px 6px;
            border-radius: 4px;
            color: #d63384;
            font-size: 0.85em;
        }}

        hr {{
            border: none;
            border-top: 1px solid #e5e5e5;
            margin: 48px 0;
        }}

        .covers-box {{
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 24px;
            margin: 24px 0;
        }}

        .covers-box strong {{
            font-size: 1.05em;
        }}

        .cta {{
            background: #f0f7ff;
            border-radius: 12px;
            padding: 32px;
            margin-top: 48px;
            text-align: center;
        }}

        .cta p {{
            margin-bottom: 0;
            font-size: 1.05em;
        }}

        .further-reading {{
            background: #fafafa;
            border-radius: 8px;
            padding: 24px;
            margin-top: 32px;
        }}

        .further-reading h2 {{
            margin-top: 0;
            font-size: 1.3em;
        }}

        @media print {{
            body {{
                max-width: 100%;
                padding: 0;
                font-size: 14px;
            }}
            pre {{
                white-space: pre-wrap;
            }}
        }}
    </style>
</head>
<body>
{body_content}
</body>
</html>'''


def main():
    print("Reading markdown...")
    md_text = MD_PATH.read_text(encoding="utf-8")

    print("Converting markdown to HTML with embedded images...")
    body_html, image_count = convert_markdown_to_html(md_text)

    title = "Sliding Window Attention: How Modern LLMs See the World Through a Narrow Lens"
    full_html = build_html(body_html, title)

    print(f"\nWriting HTML to {HTML_PATH}...")
    HTML_PATH.write_text(full_html, encoding="utf-8")

    size_mb = HTML_PATH.stat().st_size / (1024 * 1024)
    word_count = len(re.findall(r'\b\w+\b', md_text))

    print(f"\n{'='*40}")
    print(f"HTML Export Complete")
    print(f"{'='*40}")
    print(f"File: {HTML_PATH}")
    print(f"Size: {size_mb:.1f} MB")
    print(f"Images embedded: {image_count} of 35")
    print(f"Word count: {word_count}")


if __name__ == "__main__":
    main()
