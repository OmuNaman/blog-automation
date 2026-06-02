---
name: generate-diagrams
description: "Generate publication-quality academic diagrams and statistical plots for blog posts using PaperBanana CLI. Handles the full pipeline: writing detailed diagram descriptions, generating images via PaperBanana (Nano Banana Pro for diagrams, Matplotlib for plots), and visually verifying every output. Trigger when user mentions diagrams, figures, illustrations, plots, charts, or PaperBanana."
---

# Diagram & Plot Generation Skill (PaperBanana)

## Overview

1. **Write** precise description/data files for every figure
2. **Generate** using the correct PaperBanana CLI subcommand
3. **VISUALLY VERIFY every single image yourself**
4. **Fix or regenerate** anything that fails verification
5. **Report** final status to the user

---

## PaperBanana Version & Installation

- **Installed version:** 0.1.2
- **CLI path:** `C:\Users\naman\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\Scripts\paperbanana.exe`

Set this in your scripts:
```bash
PB="C:\Users\naman\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\Scripts\paperbanana.exe"
```

---

## Available Commands (v0.1.2)

PaperBanana has **4 CLI commands**. There is NO `batch` or `batch-report` command.

| Command | Purpose |
|---|---|
| `paperbanana generate` | Methodology diagrams (architectures, flowcharts, pipelines) |
| `paperbanana plot` | Statistical plots from CSV/JSON data (code-based Matplotlib) |
| `paperbanana evaluate` | Compare generated vs human reference image |
| `paperbanana setup` | First-time API key configuration |

### IMPORTANT: No batch command exists

The skill previously documented `batch` and `batch-report` commands. **These do not exist in v0.1.2.** Generate diagrams one at a time, or loop in bash.

---

## Command 1: `paperbanana generate` — Diagrams

Use for: architecture diagrams, flowcharts, system pipelines, concept illustrations, any visual with components + connections.

```bash
"$PB" generate \
  -i <path/to/description.txt> \
  -c "<figure caption>" \
  -o <path/to/output.png> \
  -n 3 \
  --vlm-model gemini-2.5-pro \
  --image-model gemini-3-pro-image-preview
```

| Flag | Short | Required | Description |
|---|---|---|---|
| `--input` | `-i` | Yes | Path to description `.txt` file |
| `--caption` | `-c` | Yes | Figure caption / communicative intent |
| `--output` | `-o` | No | Output image path (default: auto in `outputs/`) |
| `--iterations` | `-n` | No | Refinement iterations (default: 3) |
| `--vlm-provider` | | No | VLM provider (default: `gemini`) |
| `--vlm-model` | | No | VLM model name |
| `--image-provider` | | No | Image gen provider (default: `google_imagen`) |
| `--image-model` | | No | Image gen model name |
| `--config` | | No | Path to config YAML file |

### IMPORTANT: No `--optimize` flag

The `--optimize` flag was documented before but **does not exist** in `generate`. Do not pass it.

---

## Command 2: `paperbanana plot` — Statistical Plots

Use for: bar charts, line graphs, scatter plots, comparison charts. This is **code-based** (generates and executes Matplotlib Python code), so numbers are exact with zero hallucination.

```bash
"$PB" plot \
  -d <path/to/data.csv> \
  --intent "<what the chart should communicate>" \
  -o <path/to/output.png> \
  -n 3 \
  --vlm-model gemini-2.5-pro
```

| Flag | Short | Required | Description |
|---|---|---|---|
| `--data` | `-d` | Yes | Path to CSV or JSON data file |
| `--intent` | | Yes | Communicative intent for the plot |
| `--output` | `-o` | No | Output image path |
| `--vlm-provider` | | No | VLM provider (default: `gemini`) |
| `--vlm-model` | | No | VLM model name |
| `--iterations` | `-n` | No | Refinement iterations (default: 3) |

### IMPORTANT: No `--image-model` or `--optimize` flags for `plot`

The `plot` command does NOT accept `--image-model` or `--optimize`. It only uses VLM to generate Matplotlib code, then executes it. Do not pass image-related flags.

---

## Command 3: `paperbanana evaluate` — Quality Assessment

```bash
"$PB" evaluate \
  -g <generated.png> \
  -r <reference.png> \
  --context <method.txt> \
  -c "<caption>"
```

| Flag | Short | Required | Description |
|---|---|---|---|
| `--generated` | `-g` | Yes | Path to generated image |
| `--reference` | `-r` | Yes | Path to human reference image |
| `--context` | | Yes | Path to source context text file |
| `--caption` | `-c` | Yes | Figure caption |
| `--vlm-provider` | | No | VLM provider (default: `gemini`) |

Note: `evaluate` does NOT have a `--vlm-model` flag.

---

## Known Bugs & Workarounds (v0.1.2)

These are bugs we encountered and fixed locally. If PaperBanana gets reinstalled/upgraded, these fixes will be lost and need to be reapplied.

### Bug 1: Default VLM model is `gemini-2.0-flash` (deprecated)

**File:** `paperbanana/core/config.py`
**Problem:** Default `vlm_model` is `"gemini-2.0-flash"` which no longer exists.
**Fix:** Change default to `"gemini-2.5-flash"`:
```python
# In class Settings:
vlm_model: str = "gemini-2.5-flash"  # was "gemini-2.0-flash"
```

### Bug 2: `plot` command ignores `--vlm-model` flag

**File:** `paperbanana/cli.py` (in the `plot` function)
**Problem:** The `plot` command creates `Settings()` without passing `vlm_model`, so the CLI flag is silently ignored.
**Fix:** Add conditional override:
```python
# Replace:
settings = Settings(
    vlm_provider=vlm_provider,
    refinement_iterations=iterations,
)
# With:
overrides = dict(
    vlm_provider=vlm_provider,
    refinement_iterations=iterations,
)
if vlm_model:
    overrides["vlm_model"] = vlm_model
settings = Settings(**overrides)
```

### Bug 3: `_extract_code` crashes when VLM response has no closing backticks

**File:** `paperbanana/agents/visualizer.py`
**Problem:** `response.index("```", start)` throws `ValueError` if the VLM response contains opening ` ```python ` but no closing ` ``` `.
**Fix:** Use `response.find()` instead which returns -1 on failure:
```python
def _extract_code(self, response: str) -> str:
    if "```python" in response:
        start = response.index("```python") + len("```python")
        end = response.find("```", start)
        if end == -1:
            return response[start:].strip()
        return response[start:end].strip()
    elif "```" in response:
        start = response.index("```") + 3
        end = response.find("```", start)
        if end == -1:
            return response[start:].strip()
        return response[start:end].strip()
    return response.strip()
```

### Bug 4: Windows backslash in OUTPUT_PATH breaks generated plot code

**File:** `paperbanana/agents/visualizer.py` (in `_execute_plot_code`)
**Problem:** Injecting `OUTPUT_PATH = "D:\path\to\file.png"` creates invalid escape sequences.
**Fix:** Normalize to forward slashes and use raw string:
```python
# Replace:
full_code = f'OUTPUT_PATH = "{output_path}"\n{code}'
# With:
safe_path = output_path.replace("\\", "/")
full_code = f'OUTPUT_PATH = r"{safe_path}"\n{code}'
```

Also strip any VLM-generated OUTPUT_PATH before injecting:
```python
code = re.sub(r'^OUTPUT_PATH\s*=\s*["\'].*["\']\s*$', "", code, flags=re.MULTILINE)
```

---

## Model Configuration

| Role | Model ID | Override flag |
|---|---|---|
| **VLM** (planning, critique) | `gemini-2.5-pro` | `--vlm-model gemini-2.5-pro` |
| **Image generation** (diagrams) | `gemini-3-pro-image-preview` | `--image-model gemini-3-pro-image-preview` |
| **Plots** | Matplotlib (code-based) | No image model needed |

Always pass `--vlm-model gemini-2.5-pro` on every command. The default has been changed to `gemini-2.5-flash` locally, but `2.5-pro` gives better results.

---

## Routing: When to Use What

| You need... | Command | Input file |
|---|---|---|
| Architecture diagram | `generate` | `.txt` description |
| Flowchart / pipeline | `generate` | `.txt` description |
| Concept illustration | `generate` | `.txt` description |
| Bar/line/scatter chart | `plot` | `.csv` or `.json` data |
| Comparison table with metrics | `plot` | `.csv` data |
| Quality check vs reference | `evaluate` | generated + reference `.png` |

**Rule of thumb:** If it has numbers/data, use `plot`. If it has boxes/arrows/flow, use `generate`.

---

## Input: `$ARGUMENTS` = topic name or slug

Expects ONE of:
- Existing outline at `plan/<topic-slug>_outline.md` with a **Diagram Master List**
- OR a topic name, and we build the diagram list from scratch

---

## File Organization

```
figures/
  descriptions/<topic-slug>/     # .txt files for `generate` command
    fig_overview.txt
    fig_architecture.txt
  data/<topic-slug>/             # .csv/.json files for `plot` command
    fig_benchmark.csv
    fig_scaling.csv
  output/<topic-slug>/           # Generated .png files
    fig_overview.png
    fig_benchmark.png
```

---

## Process

### Step 1: Write ALL Description/Data Files FIRST

Write ALL files BEFORE running any generation. Consistency matters.

#### For Diagrams (`.txt` → `generate`)

Create at `figures/descriptions/<topic-slug>/<fig_id>.txt`:

```
<Diagram Title>

IMPORTANT: Pure white background (#FFFFFF). No gray tint, no gradient.

== CONTEXT ==
<1-2 sentences explaining what this shows>

== LAYOUT ==
- Direction: top-to-bottom / left-to-right / center-outward
- Sections: <how many groups, relative positions>

== COMPONENTS ==
1. Component Name
   - Shape: rounded rectangle / circle / diamond
   - Color: light blue #B3D9FF / soft green #B8E6B8 / lavender #D4B8E6 /
     light pink #FFB8D4 / light gray #E0E0E0 / peach #FFD4B8
   - Label: "<exact text>"
   - Position: top-left / center / etc.

== CONNECTIONS ==
- Solid black arrows (→) for main data flow
- Dashed arrows (-->) for optional/shared paths
- Thin gray arrows for residual/skip connections

1. ComponentA → ComponentB: solid black, label "embeddings"
2. ComponentB → ComponentC: solid black

== STYLE ==
Clean academic illustration. Sans-serif font. Pastel palette.
White background (#FFFFFF).
```

#### For Plots (`.csv` → `plot`)

Create data at `figures/data/<topic-slug>/<fig_id>.csv`:

```csv
model,accuracy,f1_score,latency_ms
Baseline,78.2,76.1,45
Ours (small),85.4,83.2,52
```

The `--intent` flag describes what the chart should communicate.

### Step 2: Generate Each Figure

#### Diagrams (one at a time, no batch):
```bash
"$PB" generate \
  -i figures/descriptions/<topic>/<fig_id>.txt \
  -c "<caption>" \
  -o figures/output/<topic>/<fig_id>.png \
  -n 3 \
  --vlm-model gemini-2.5-pro \
  --image-model gemini-3-pro-image-preview
```

#### Plots:
```bash
"$PB" plot \
  -d figures/data/<topic>/<fig_id>.csv \
  --intent "<communicative intent>" \
  -o figures/output/<topic>/<fig_id>.png \
  -n 3 \
  --vlm-model gemini-2.5-pro
```

#### Generating multiple diagrams in sequence:

Since there's no batch command, loop in bash:
```bash
PB="C:\Users\naman\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\Scripts\paperbanana.exe"
DESC="d:/Coding_Workspace/Blog_automation/figures/descriptions/<topic>"
OUT="d:/Coding_Workspace/Blog_automation/figures/output/<topic>"

echo "=== 1/N: fig_name ==="
"$PB" generate -i "$DESC/fig_name.txt" -c "Caption" -o "$OUT/fig_name.png" -n 3 --vlm-model gemini-2.5-pro --image-model gemini-3-pro-image-preview 2>&1

echo "=== 2/N: fig_name2 ==="
"$PB" generate -i "$DESC/fig_name2.txt" -c "Caption" -o "$OUT/fig_name2.png" -n 3 --vlm-model gemini-2.5-pro --image-model gemini-3-pro-image-preview 2>&1
```

**Rate limits:** Each `generate` call takes 2-5 minutes and makes multiple Gemini API calls. If you hit rate limits (429 errors), wait 120 seconds between calls.

---

### Step 3: VISUALLY VERIFY EVERY SINGLE IMAGE

> **Open and look at EVERY generated .png using the Read tool.**

For each image, check:

1. **Components** — Are ALL components from the description present?
2. **Labels** — Spelled correctly? No garbled text?
3. **Layout** — Matches the described arrangement?
4. **Connections** — Arrows present, correct direction, solid vs dashed?
5. **Colors** — Pastel palette? White background?
6. **Readability** — All text readable? Nothing cut off?

For plots also check:
- Axis labels correct and readable
- Plotted values match the CSV data
- Legend present
- Chart type correct

#### Rating:
- **PASS** — All correct. Move on.
- **FIXABLE** — 1-3 specific issues. Add `CRITICAL:` markers to description, regenerate.
- **FAIL** — Fundamentally broken after 3 attempts. Flag for user.

---

### Step 4: Handle Failures

#### FIXABLE (max 3 retries):
1. Add `CRITICAL:` lines to the description file
2. Regenerate only that figure
3. Verify again

#### FAIL (max 3 retries):
1. Simplify the description (fewer components, shorter labels)
2. Consider splitting into 2 simpler figures
3. After 3 failures, flag for user with best attempt shown

---

### Step 5: Matplotlib Fallback for Plots

If `paperbanana plot` fails repeatedly, generate plots directly with matplotlib:

```python
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("figures/data/<topic>/<fig_id>.csv")
fig, ax = plt.subplots(figsize=(10, 6))
# ... plot code ...
fig.savefig("figures/output/<topic>/<fig_id>.png", dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
```

This is a reliable fallback since `paperbanana plot` ultimately generates and runs matplotlib code anyway.

---

### Step 6: Final Report

```markdown
# Figure Generation Report

## Summary
- Total: X figures (D diagrams + P plots)
- PASSED first try: Y
- PASSED after fixes: Z
- FAILED: W

## Status
| # | ID | Type | Status | Attempts | Notes |
|---|---|---|---|---|---|
| 1 | fig_overview | diagram | PASS | 1 | Clean |
| 2 | fig_results | plot | PASS | 1 | Data accurate |
| 3 | fig_arch | diagram | FIXED | 3 | Label typo fixed |
```

---

## Quick Reference

```bash
# Diagram
"$PB" generate -i desc.txt -c "Caption" -o out.png -n 3 --vlm-model gemini-2.5-pro --image-model gemini-3-pro-image-preview

# Plot
"$PB" plot -d data.csv --intent "Show X vs Y" -o out.png -n 3 --vlm-model gemini-2.5-pro

# Evaluate
"$PB" evaluate -g generated.png -r reference.png --context method.txt -c "Caption"
```

---

## Important Rules

1. **Write ALL descriptions BEFORE any generation**
2. **Always pass `--vlm-model gemini-2.5-pro`** for best quality
3. **NO `--optimize` flag** (does not exist)
4. **NO `batch` command** (does not exist)
5. **VERIFY EVERY IMAGE YOURSELF**
6. **Route correctly:** data → `plot`, diagram → `generate`
7. **Max 3 retries** per figure before flagging
8. **White background (#FFFFFF)** in every description
9. **Rate limit awareness:** wait between calls if hitting 429s

## What Works Well
- Methodology diagrams (architectures, pipelines, workflows)
- Statistical plots via Matplotlib (bar, line, scatter)
- Concept illustrations with labeled components
- Educational infographics

## Known Limitations
- Complex math/equations render as garbled text
- Very dense diagrams (15+ components) may be unreliable (split them)
- Arrow alignment can be imperfect
- Raster output only (PNG, no SVG)
- No batch mode; must generate one at a time
- Rate limits on Gemini API (wait between calls)
- Windows path backslashes cause issues in plot code (fixed locally)
