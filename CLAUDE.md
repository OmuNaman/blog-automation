# Blog Automation Project

## What This Is
An automated pipeline for creating high-quality, deeply pedagogical technical blog posts
with professional diagrams generated via PaperBanana. The output should match the style
and quality of the DeepSeek book chapters (Chapters 3, 4, 5) exactly.

## Tech Stack
- PaperBanana (CLI + MCP) for academic-quality diagram generation
- Markdown for article drafts
- Substack / Medium for publishing

## Target Audience
ML engineers, researchers, and students who want to deeply understand technical concepts.

---

## PaperBanana CLI Path (Windows)
PaperBanana is not on PATH. Always use the full executable path:
- **paperbanana CLI**: `C:\Users\naman\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\Scripts\paperbanana.exe`
- **paperbanana MCP**: `C:\Users\naman\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\Scripts\paperbanana-mcp.exe`

---

## WRITING STYLE GUIDE (CRITICAL: Follow this exactly)

### Overall Philosophy
The writing style is that of an exceptional teacher who walks beside the reader.
Every concept is built from intuition first, then visualized, then formalized with math,
then implemented in code. The reader should never feel lost. Every paragraph should feel
like a conversation with a brilliant, patient mentor.

### Tone and Voice
- Use "we" perspective throughout. The writer and reader are learning together.
  Example: "Let's trace the journey of a new token as it enters the system."
  Example: "Now that we have our compressed matrix, we can reconstruct..."
  Example: "We have successfully built a functional MoE layer."
- Conversational but technically precise. Never sacrifice accuracy for simplicity.
- Enthusiastic about clever ideas. Use phrases like:
  "This is the magic of MLA"
  "This is the beauty of the MoE design"
  "The answer lies in a beautiful trick"
- Direct and confident. State things clearly, do not hedge unnecessarily.

### STRICT FORMATTING RULES

#### NEVER use em dashes (the long dash: —)
This is absolutely critical. Use commas, semicolons, or separate sentences instead.
- WRONG: "The model — which uses attention — processes tokens"
- RIGHT: "The model, which uses attention, processes tokens"
- RIGHT: "The model uses attention. It processes tokens."

#### NEVER use emojis

#### Paragraph length
- Keep paragraphs to 2 to 4 sentences maximum.
- Each paragraph should make exactly one point.
- Start new paragraphs frequently. Dense walls of text are forbidden.

#### Sentence starters (use these frequently)
- "The..." (most common, 20% of sentences)
- "This..." (second most common)
- "As shown in figure X.Y, ..."
- "As illustrated in figure X.Y, ..."
- "Let's..." (to invite the reader into action)
- "We..." (collaborative)
- "For..." (when giving examples)
- "Notice..." (drawing attention)
- "However, ..." (for counterpoints)
- "Now that we have..., let's..." (transitions)

#### Section titles
- Head 1 titles are descriptive and engaging:
  "The intuition behind mixture of experts"
  "The mechanics of MoE: A hands-on mathematical walkthrough"
  "Quantifying the gains"
- Head 2 titles are specific and action-oriented:
  "The query path (unchanged)"
  "The key/value path (the innovation)"
  "From scores to weights: Top-K selection and softmax normalization"
  "The absorption trick: How attention scores are calculated"
  "Attempt #1: The auxiliary loss"
  "Attempt #2: The load balancing loss"

#### Bullet points structure
Bullet points use a bold lead-in term followed by a colon and explanation:
- **Input Matrix**: Shape (4, 8), four tokens, each with an 8-dimensional embedding.
- **Expert 1 Importance**: 0.9 + 0.5 = 1.4
- **Punctuation Experts**: The router first encounters the question mark...

#### Numbered lists for sequential steps
Use numbered lists for step-by-step processes:
1. Calculate Attention Weights: We multiply the Query Vector...
2. Compute Context Vector: We multiply these Attention Weights...

### PEDAGOGICAL STRUCTURE (Follow this for every article)

#### Article opening
1. Article title (descriptive, not clickbait)
2. "This article covers" followed by 3 to 5 bullet points
3. Bridge paragraph connecting to prerequisite knowledge
4. Roadmap figure (Figure X.1) showing where this topic fits in the bigger picture

#### Section structure (repeat for each major concept)
1. **Intuition first**: Explain WHY this exists. What problem does it solve?
   Use the pattern: "The problem with X" followed by "The solution: Y"
2. **Visual walkthrough**: Introduce the architecture/process with a figure.
   Say "Let's examine the complete workflow, as illustrated in figure X.Y."
3. **Step-by-step breakdown**: Walk through the figure component by component.
   Use a running example with concrete numbers (e.g., 4 tokens, shape (4, 8)).
4. **Mathematical formalization**: Show the equations, derive them step by step.
   Use "The absorption trick" style naming for key insights.
5. **Quantify the gains**: Give exact numbers comparing old vs new approach.
6. **Code implementation**: Show working code with annotations.

#### How to use figures (EXTREMELY IMPORTANT, 25 to 35 figures per article)
Figures are the backbone of the teaching. Every concept gets a figure.

Before showing a figure, introduce it:
"Let's examine the complete workflow, as illustrated in figure 3.2."
"The full process is shown in figure 4.7."

After showing a figure, explain it:
"As shown in figure 3.2, the data flows through two main paths..."
"As illustrated in figure 4.11, the calculation proceeds as follows:"

Figure caption format:
"Figure 3.2 The full architectural data flow of Multi-Head Latent Attention (MLA)."
"Figure 4.5 The initial challenge of MoE. The input matrix is passed through each
of the three expert networks in parallel, resulting in three separate output matrices."

Captions are descriptive and self-contained. A reader should understand the figure
from the caption alone.

#### Running examples
Use a consistent, simple running example throughout the article:
- 4 tokens: "The", "next", "day", "is"
- Small dimensions: (4, 8) input matrix
- Trace concrete values through every step
- Show exact matrix shapes at every transformation

#### Callout boxes
Use sparingly (1 to 2 per article) for key definitions:
"Note: In our example, the latent dimension happens to be the same as..."
"What is a Hidden State? A hidden state is another name for..."
"Expert Capacity: A fixed limit on the maximum number of tokens..."

#### Transitions between sections
Use explicit bridging sentences:
"Now that we have the core intuition, it's time to open the black box."
"We have successfully built X. But this raises a new question..."
"This sets up the central challenge of this section."
"Having explored the theory, let's put our knowledge to the test."

#### Article ending
- Summary section with bullet points recapping key takeaways
- Each bullet is 2 to 3 sentences, not just a phrase
- Optional: link to code implementation

### WHAT TO AVOID
- Em dashes (use commas or periods instead)
- Vague hand-waving ("it somehow works better")
- Skipping steps in derivations
- Figures without explanation
- Sections without figures
- Overly long paragraphs (max 4 sentences)
- Passive voice when active is clearer
- Jargon without definition on first use
- Clickbait titles or sensationalism

---

## DIAGRAM STYLE (PaperBanana)
- Pure white background (#FFFFFF) always
- Clean academic illustration style
- Sans-serif labels (clear, readable)
- Pastel color palette: blue, green, lavender, pink, gray, peach/orange
- Solid black arrows for data flow
- Thin gray arrows for residual/skip connections
- Dashed arrows for tied/shared weights
- Colored bordered rectangles to group related components
- Matrix shapes always labeled: (rows, columns)
- Every component in a diagram must be labeled

---

## WORKFLOW
1. /research-topic <topic> -> deep research, save to research/
2. /plan-article <topic> -> outline with diagram specs, save to plan/
3. /generate-diagrams <topic> -> write .txt descriptions, run PaperBanana, verify
4. /write-article <topic> -> draft using research + outline + figures
5. /publish <topic> -> format for Substack, manual approval required

## RULES
- NEVER publish without explicit user approval
- NEVER use em dashes
- Always save research notes before writing
- Every .txt diagram description must be saved BEFORE generating
- Aim for 25 to 35 diagrams per article
- Every section must have at least one diagram
- Use concrete running examples with exact numbers
- Build intuition BEFORE math, math BEFORE code
