# Blog Automation Pipeline

An automated pipeline for creating deeply pedagogical technical blog posts with professional diagrams. Built on Claude Code skills and PaperBanana for academic-quality diagram generation. The output follows a rigorous teaching style: intuition first, then visuals, then math, then code.

## What It Does

You give it a topic. It researches, plans, generates 25-35 diagrams, writes a 7,000+ word article, and exports a self-contained HTML file ready for Substack or Medium. The entire process is broken into five skills that can be run individually or as a full pipeline.

## Sample Output

The first article produced by this pipeline is on **Sliding Window Attention**, included in the `published/` directory. It contains 35 PaperBanana-generated diagrams, 5 code blocks, and covers the full mechanism from intuition through implementation.

---

## Setup

### Prerequisites

- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) CLI installed
- [PaperBanana](https://pypi.org/project/paperbanana/) CLI installed (`pip install paperbanana`)
- A Google API key for Gemini (used by PaperBanana for image generation)
- Python 3.10+

### Installation

1. Clone the repository:

```bash
git clone https://github.com/<your-username>/blog-automation.git
cd blog-automation
```

2. Install PaperBanana:

```bash
pip install paperbanana
```

3. Create a `.env` file in the project root with your Google API key:

```
GOOGLE_API_KEY=your-google-api-key-here
```

4. Open the project in Claude Code:

```bash
claude
```

Claude Code will automatically load the `CLAUDE.md` file (writing style guide and project rules) and all skills from `.claude/skills/`.

### PaperBanana MCP Setup (Optional)

For direct MCP integration, add this to your Claude Code MCP config (`~/.claude/mcp.json`):

```json
{
  "mcpServers": {
    "paperbanana": {
      "command": "uvx",
      "args": ["--from", "paperbanana[mcp]", "paperbanana-mcp"],
      "env": {
        "GOOGLE_API_KEY": "your-google-api-key"
      }
    }
  }
}
```

---

## Project Structure

```
blog-automation/
├── CLAUDE.md                          # Writing style guide + project rules
├── .claude/
│   └── skills/
│       ├── research-topic/SKILL.md    # Deep web research on a topic
│       ├── plan-article/SKILL.md      # Article outline with diagram specs
│       ├── generate-diagrams/SKILL.md # PaperBanana diagram generation
│       ├── write-article/SKILL.md     # Full article draft
│       ├── publish/SKILL.md           # Final formatting + approval gate
│       ├── export-html/SKILL.md       # Self-contained HTML export
│       └── write-blog/SKILL.md        # Full pipeline (all steps)
├── research/                          # Research notes (markdown)
├── plan/                              # Article outlines with diagram specs
├── figures/
│   ├── descriptions/                  # .txt description files for PaperBanana
│   └── output/                        # Generated .png diagram files
├── drafts/                            # Article drafts (markdown)
├── published/                         # Final articles (markdown + HTML)
└── build_html.py                      # HTML export script
```

---

## Usage

### The Full Pipeline

To write a complete article from scratch, run each skill in order inside Claude Code:

#### Step 1: Research

```
/research-topic sliding window attention
```

Performs deep web research on the topic. Searches multiple sources, reads papers and blog posts, and saves structured research notes to `research/<topic-slug>.md`. The notes include core concepts, mathematical foundations, comparisons, implementation details, and visual opportunities for diagrams.

#### Step 2: Plan

```
/plan-article sliding window attention
```

Creates a detailed article outline based on the research notes. Produces a section-by-section plan with teaching flow, transition sentences, and specifications for every diagram (typically 25-35 per article). Saved to `plan/<topic-slug>_outline.md`.

#### Step 3: Generate Diagrams

```
/generate-diagrams sliding window attention
```

Writes `.txt` description files for each planned diagram, then runs PaperBanana CLI to generate them. Each diagram goes through an iterative generation and critique cycle (PaperBanana uses Gemini VLM for feedback). Every generated diagram is visually verified. Output goes to `figures/output/`.

Note: This step can take a while depending on the number of diagrams and API rate limits. The skill generates 2 diagrams at a time by default.

#### Step 4: Write

```
/write-article sliding window attention
```

Writes the full article following the outline, referencing all generated diagrams. The skill visually inspects every diagram before writing to ensure accurate descriptions. Applies all style rules from `CLAUDE.md` (no em dashes, "we" perspective, short paragraphs, bold lead-ins, figure references before and after each image). Saved to `drafts/<topic-slug>.md`.

#### Step 5: Publish

```
/publish sliding window attention
```

Adds a subtitle, Further Reading section with source links, and a subscriber call-to-action. Runs a final quality check (em dashes, figure references, formatting). Presents a summary for manual approval before copying to `published/<topic-slug>.md`.

#### Step 6: Export HTML

```
/export-html sliding window attention
```

Converts the published markdown into a single self-contained HTML file with all images embedded as base64 data URIs. The HTML file can be opened in any browser without an internet connection. Useful for copy-pasting into Substack or Medium editors. Saved to `published/<topic-slug>.html`.

### One-Shot Pipeline

To run all steps automatically with approval checkpoints:

```
/write-blog sliding window attention
```

---

## Writing Style

The `CLAUDE.md` file contains a comprehensive writing style guide. Key rules enforced across all articles:

- **No em dashes**. Use commas, semicolons, or separate sentences instead.
- **"We" perspective** throughout. The writer and reader learn together.
- **Short paragraphs** of 2-4 sentences maximum. Each makes exactly one point.
- **25-35 diagrams** per article. Every section has at least one figure.
- **Figures referenced before and after**. Introduce with "as shown in figure X", explain after with "As illustrated in figure X, ...".
- **Bold lead-ins** on bullet points: "**Component Name**: explanation..."
- **Intuition before math, math before code**. Build understanding in layers.
- **Concrete running examples** traced through every step with exact numbers.
- **Callout boxes** for key definitions (1-2 per article).
- **Transition sentences** between every section.

## Diagram Style (PaperBanana)

All diagrams follow a consistent academic illustration style:

- Pure white background
- Sans-serif labels
- Pastel color palette (blue, green, lavender, pink, gray, peach)
- Solid black arrows for data flow
- Thin gray arrows for residual connections
- Dashed arrows for tied/shared weights
- Matrix shapes always labeled with dimensions
- Every component labeled

---

## Target Audience

ML engineers, researchers, and students who want to deeply understand technical concepts. The style matches the pedagogical depth of textbook chapters, not blog post summaries.

## License

MIT
