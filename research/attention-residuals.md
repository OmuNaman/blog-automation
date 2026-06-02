# Research: Attention Residuals (AttnRes)

## Quick Summary

Attention Residuals (AttnRes), proposed by Moonshot AI's Kimi team (March 2026, arXiv:2603.15031), replaces standard residual connections in transformers with learned, input-dependent softmax attention over preceding layer outputs. This addresses the fundamental "PreNorm dilution" problem where hidden states grow as O(L) with depth, progressively drowning out individual layer contributions. Block AttnRes, the practical variant, partitions layers into ~8 blocks to reduce memory from O(Ld) to O(Nd), achieving a 1.25x compute advantage with <2% inference overhead. Integrated into Kimi Linear (48B total / 3B activated, 1.4T tokens), it improved GPQA-Diamond by +7.5 points and Math by +3.6 points.

---

## Core Concepts

### What is it?

AttnRes is a drop-in replacement for standard residual connections in transformers. Instead of each layer receiving a fixed, equal-weighted sum of all previous layer outputs (h_l = h_{l-1} + f_l(h_{l-1})), AttnRes allows each layer to **selectively choose** how much weight to give each previous layer's output using softmax attention computed via a learned pseudo-query vector.

### Why does it exist? The PreNorm Dilution Problem

Standard residual connections with PreNorm (the dominant architecture in modern LLMs) have a fundamental flaw:

1. **Uniform accumulation**: Every layer output is added with a fixed weight of 1.0. The hidden state at layer l is literally the sum of all previous layer outputs: h_l = x_0 + f_1(h_0) + f_2(h_1) + ... + f_l(h_{l-1})

2. **O(L) hidden state growth**: Since we keep adding vectors, the magnitude ||h_l|| grows linearly with depth L. In a 50-layer model, the hidden state is the sum of ~50 vectors.

3. **Signal dilution**: Each individual layer's contribution becomes a smaller and smaller fraction of the total. A layer's output f_l might be ~1/L of the total hidden state. Deeper layers must produce increasingly large outputs just to be "heard."

4. **Lack of selectivity**: The attention sublayer and the FFN sublayer receive the same blended signal, even though they may benefit from different mixtures of earlier information.

5. **Many layers are nearly redundant**: Research has shown entire layers can be removed from deep LLMs with minimal performance impact, suggesting the uniform accumulation wastes capacity.

This is the "PreNorm dilution" problem. The name comes from the fact that PreNorm (applying LayerNorm before each sublayer) is the standard in modern LLMs, and while it stabilizes training, it does nothing to address the uniform accumulation issue.

### The Depth-Time Duality Insight

The key conceptual breakthrough: the problem of information dilution across network **depth** is structurally identical to the problem of memory loss across a **sequence** of tokens.

- In sequence processing, early tokens get "forgotten" as more tokens are added. The Transformer solved this with attention, allowing any position to selectively access any previous position.
- In depth processing, early layer outputs get "diluted" as more layers are added. AttnRes solves this the same way: by applying attention over depth.

The authors describe this as "rotating" the attention mechanism 90 degrees: from horizontal (across sequence positions) to vertical (across network depth).

### What did it replace/improve upon?

AttnRes improves upon:
- **Standard residual connections** (He et al., 2016): Fixed unit-weight accumulation
- **DenseFormer** (EPFL, 2024): Uses Depth-Weighted Averaging (DWA) with learned but input-independent scalar weights per layer pair
- **DeepCrossAttention** (Google, 2025): Uses learnable input-dependent weights for dynamic layer combination, claims 3x training speedup
- **Value Residual Learning / ResFormer** (2024): Adds residual connections specifically to value vectors from first layer to all subsequent layers

AttnRes is distinguished by using full softmax attention with input-dependent, content-aware weights over depth, making it the most expressive approach.

---

## How It Works (Technical Depth)

### Step 1: Standard Residual Recap

In a standard PreNorm transformer with L layers:

```
h_0 = x  (token embeddings)
h_l = h_{l-1} + f_l(RMSNorm(h_{l-1}))   for l = 1, ..., L
```

Where f_l is the sublayer (attention or FFN). This means:

```
h_L = x + f_1(RMSNorm(h_0)) + f_2(RMSNorm(h_1)) + ... + f_L(RMSNorm(h_{L-1}))
    = x + Σ_{l=1}^{L} f_l(RMSNorm(h_{l-1}))
```

Every layer output gets weight 1.0. No selectivity.

### Step 2: Full AttnRes Formulation

AttnRes replaces the fixed accumulation with learned attention weights:

```
h_l = Σ_{i=0}^{l-1} α_{i→l} · v_i
```

Where:
- v_0 = x (token embeddings)
- v_i = f_i(RMSNorm(h_{i-1})) for i >= 1 (each layer's output)
- α_{i→l} are attention weights computed via softmax

The attention weights are computed as:

```
α_{i→l} = softmax_i( w_l^T · RMSNorm(k_i) )
         = exp(w_l^T · RMSNorm(k_i)) / Σ_{j=0}^{l-1} exp(w_l^T · RMSNorm(k_j))
```

Where:
- w_l ∈ R^d is a **learned pseudo-query vector** for layer l (one per layer)
- k_i is the "key" representation for layer i's output (derived from v_i)
- RMSNorm normalizes the keys before scoring

**Key insight**: Each layer has its own query vector w_l that determines which previous layers are most relevant. Different tokens can retrieve different layer representations based on what is actually useful (input-dependent).

### Step 3: Block AttnRes (The Practical Variant)

Full AttnRes requires O(Ld) memory (storing all L layer outputs of dimension d). For large models with pipeline parallelism, this is impractical.

**Solution: Block AttnRes**

1. **Partition** L layers into N blocks (typically N ≈ 8) of roughly equal size
2. **Within each block**: Use standard residual connections (simple addition)
3. **At block boundaries**: Use attention-based aggregation across block representations

This reduces memory from O(Ld) to O(Nd).

**Block-level mechanism**:
- Let B_1, B_2, ..., B_N be the N blocks
- After each block completes, its cumulative output becomes one "value" in the attention
- The next block's input is a softmax-weighted combination of all previous block outputs plus the original token embedding

**PyTorch pseudocode** (from official repo):

```python
def block_attn_res(block_reprs, current_partial_sum, proj):
    # block_reprs: list of previous block outputs + token embeddings
    # current_partial_sum: accumulated hidden state within current block

    # Stack as values: [N_prev, batch, seq_len, d_model]
    V = stack(block_reprs + [current_partial_sum])
    V = RMSNorm(V)

    # Compute attention logits using learned projection
    # proj.weight has shape [1, d_model]
    logits = einsum('d, n b t d -> n b t', proj.weight.squeeze(), V)

    # Softmax over the block dimension (dim=0)
    weights = logits.softmax(dim=0)

    # Weighted aggregation
    output = einsum('n b t, n b t d -> b t d', weights, V)

    return output
```

**Checkpointing**: Block boundaries are detected when `layer_number % (block_size // 2) == 0`. At these points, the block representation is cached for future attention.

### Step 4: System Implementation for Pipeline Parallelism

**Challenge**: In pipeline-parallel training, different stages hold different layers. Cross-block attention requires representations from blocks on different GPUs.

**Solution: Cache-based pipeline communication**
- Each pipeline stage caches its block representations
- When a new block starts, it receives cached representations from previous stages via point-to-point communication
- This avoids redundant recomputation

**Two-phase computation strategy**:
- Phase 1: Compute standard attention on the input within the current block
- Phase 2: Apply cross-block attention using cached block representations
- Online softmax used to amortize computation during inference

---

## Mathematical Foundation

### Hidden State Growth Analysis

**Standard residuals:**
```
||h_L||² ≈ ||x||² + Σ_{l=1}^{L} ||f_l||² + cross_terms
```
The magnitude grows as O(L), assuming layer outputs are roughly uncorrelated with mean magnitude μ:
```
||h_L|| ≈ √L · μ  (for uncorrelated) or L · μ (for correlated)
```

**AttnRes:**
```
||h_L||² = ||Σ α_{i→L} v_i||²
```
Since α are softmax weights (sum to 1), the output is a convex combination:
```
||h_L|| ≤ max_i ||v_i||
```
The hidden state magnitude is **bounded** regardless of depth, completely eliminating the O(L) growth problem.

### Gradient Flow Analysis

**Standard residuals:**
```
∂L/∂h_l = ∂L/∂h_L · Π_{k=l+1}^{L} (I + ∂f_k/∂h_{k-1})
```
The gradient must flow through a product of (L-l) terms, which can lead to vanishing/exploding gradients despite residual connections helping with the identity term.

**AttnRes:**
```
∂L/∂v_l = Σ_{k=l+1}^{L} α_{l→k} · ∂L/∂h_k + (terms from attention weight gradients)
```
Every layer receives a direct gradient signal weighted by its attention weight. Layers that contribute more (higher α) get stronger gradients. This creates a more uniform gradient distribution across depth.

### Formal Equivalence to Linear Attention

The paper establishes that standard residual connections are equivalent to **low-rank linear attention over depth**. Specifically, standard residuals compute:

```
h_l = Σ_{i=0}^{l-1} 1 · v_i  (all weights = 1)
```

This is linear attention with constant attention weights. AttnRes generalizes this to full-rank softmax attention over depth, providing strictly more expressive depth-wise information routing.

---

## Comparisons and Alternatives

### vs. Standard Residual Connections
| Aspect | Standard | AttnRes |
|--------|----------|---------|
| Weights | Fixed (all 1.0) | Learned, input-dependent |
| Hidden state growth | O(L) | Bounded (convex combination) |
| Layer selectivity | None | Full softmax attention |
| Overhead | Zero | <2% inference, <4% training |
| Depth preference | Favors wider, shallower | Enables narrower, deeper |

### vs. DenseFormer (EPFL, Feb 2024)
- **DenseFormer**: Uses Depth-Weighted Averaging (DWA) with **learned scalar weights per layer pair** (input-independent)
- **AttnRes**: Uses **input-dependent** attention weights via pseudo-query vectors
- DenseFormer weights are fixed after training; AttnRes weights change per token
- AttnRes is strictly more expressive but slightly more expensive

### vs. DeepCrossAttention (Google, Feb 2025)
- **DCA**: Uses learnable, input-dependent weights for dynamic layer combination
- Claims up to 3x training speedup for equivalent quality
- Both address the same fundamental problem but with different mechanisms
- AttnRes uses a single pseudo-query vector per layer; DCA uses full cross-attention

### vs. Value Residual Learning / ResFormer (Oct 2024)
- **ResFormer**: Adds residual connection from first layer's values to all subsequent layers' values
- Targets attention concentration specifically (not hidden state dilution broadly)
- ResFormer saves ~10-14% parameters for equivalent loss
- SVFormer variant reduces KV cache by ~50% by sharing values
- AttnRes is more general (operates on full hidden states, not just values)

### Quantitative Results (Kimi Linear 48B/3B, 1.4T tokens)

| Task | Baseline | AttnRes | Delta |
|------|----------|---------|-------|
| MMLU | 73.5 | 74.6 | +1.1 |
| GPQA-Diamond | 36.9 | 44.4 | **+7.5** |
| BBH | 76.3 | 78.0 | +1.7 |
| TriviaQA | 69.9 | 71.8 | +1.9 |
| Math | 53.5 | 57.1 | **+3.6** |
| HumanEval | 59.1 | 62.2 | **+3.1** |
| MBPP | 72.0 | 73.9 | +1.9 |
| CMMLU | 82.0 | 82.9 | +0.9 |
| C-Eval | 79.6 | 82.5 | +2.9 |

**Scaling law result**: Block AttnRes matches baseline performance trained with 1.25x more compute.

---

## Historical Context

### Timeline of Depth-wise Aggregation Innovations
- **2016**: Residual connections (He et al.) - fixed unit-weight addition
- **2016**: DenseNet (Huang et al.) - concatenate all previous outputs (vision)
- **2017**: Transformer (Vaswani et al.) - adopts residual connections
- **2020**: PreNorm becomes standard in LLMs (GPT-3 era)
- **Feb 2024**: DenseFormer (EPFL) - learned input-independent scalar weights per layer
- **Oct 2024**: Value Residual Learning / ResFormer - residual connections on value vectors
- **Feb 2025**: DeepCrossAttention (Google) - learnable input-dependent cross-layer weights
- **Mar 2026**: Attention Residuals (Moonshot/Kimi) - full softmax attention over depth

### Key Papers
- He et al. (2016) "Deep Residual Learning for Image Recognition" - original residual connections
- Vaswani et al. (2017) "Attention Is All You Need" - transformer architecture
- Bai et al. (2024) "DenseFormer: Enhancing Information Flow via DWA" - depth-weighted averaging
- Qiu et al. (2024) "Value Residual Learning for Alleviating Attention Concentration" - ResFormer
- Heddes et al. (2025) "DeepCrossAttention: Supercharging Transformer Residual Connections" - DCA
- Chen et al. (2026) "Attention Residuals" - AttnRes (this paper)

---

## Critical Perspectives

### Ziming Liu's Analysis: "When Does Attention Residuals Work?"

Liu (MIT/Caltech) provides a nuanced analysis:

1. **When AttnRes excels**: Tasks that require skipping intermediate layers efficiently. The attention mechanism can learn to focus on specific layers without needing to suppress intermediate representations sequentially.

2. **When AttnRes struggles**: During initialization or poor training, alpha weights may converge toward uniform distributions, causing "representation collapse." This uniform bias averages all previous hidden states, potentially limiting expressive capacity.

3. **No Free Lunch**: Success depends on task characteristics. Natural language's structured nature may explain Kimi's strong empirical results, but random memorization tasks can favor standard residuals.

4. **Experimental validation**: Toy datasets confirmed that AttnRes outperforms standard connections for structured/linear tasks but underperforms for pure memorization tasks.

---

## Visual Opportunities (Diagrams Needed)

### 1. fig_prenorm_dilution_problem
- **What**: Show how hidden state magnitude grows with depth in standard residuals. Left side shows a stack of layers with h_l = h_{l-1} + f_l, with the hidden state vector getting progressively larger. Right side shows a bar chart of ||h_l|| vs layer number, showing linear O(L) growth. Each layer's contribution shrinks as a fraction.
- **Type**: comparison / before-after

### 2. fig_standard_vs_attnres_overview
- **What**: Side-by-side comparison of standard residual connections (left) and AttnRes (right). Standard: simple arrows adding into a single stream. AttnRes: each layer has a query vector that attends to all previous layer outputs via softmax, selectively weighting them.
- **Type**: architecture comparison

### 3. fig_depth_time_duality
- **What**: The "90-degree rotation" insight. Left: horizontal attention across tokens in a sequence (standard self-attention). Right: vertical attention across layers in depth (AttnRes). Show the parallel structure.
- **Type**: conceptual / comparison

### 4. fig_attnres_mechanism_detailed
- **What**: Detailed walkthrough of one layer's AttnRes computation. Show the pseudo-query w_l, the keys from previous layers (RMSNorm applied), the dot products producing logits, softmax producing weights α, and the weighted sum producing h_l. Use concrete shapes and a running example.
- **Type**: step-by-step / data flow

### 5. fig_block_attnres_architecture
- **What**: The Block AttnRes architecture. Show L layers partitioned into N=8 blocks. Within blocks: standard residual arrows. At block boundaries: attention mechanism across block representations. Show token embeddings as a separate always-available source.
- **Type**: architecture

### 6. fig_attention_weight_heatmap
- **What**: Visualization of learned attention weights α_{i→l}. Heatmap with source layer on x-axis, target layer on y-axis. Show patterns: some layers attend strongly to embeddings, some to nearby layers, some to distant layers. Illustrate that different layers learn different aggregation patterns.
- **Type**: visualization / heatmap

### 7. fig_hidden_state_norm_comparison
- **What**: Line plot comparing hidden state norms across layers. Standard residuals: linearly increasing line. AttnRes: bounded, roughly flat line. Show that AttnRes eliminates the O(L) growth problem.
- **Type**: comparison / chart

### 8. fig_gradient_distribution
- **What**: Gradient norm distribution across layers. Standard: concentrated in shallow layers, decaying toward deeper layers. AttnRes: more uniform distribution across all layers, enabling effective training of deep layers.
- **Type**: comparison / chart

### 9. fig_scaling_law_curves
- **What**: Scaling law curves showing validation loss vs compute. Standard residuals curve and AttnRes curve. AttnRes consistently below (better). Annotate the 1.25x compute advantage: mark where AttnRes curve matches the standard curve at 1.25x more compute.
- **Type**: chart

### 10. fig_benchmark_results_comparison
- **What**: Grouped bar chart comparing baseline vs AttnRes across all benchmarks (MMLU, GPQA-Diamond, BBH, TriviaQA, Math, HumanEval, MBPP, CMMLU, C-Eval). Highlight the biggest gains (GPQA-Diamond +7.5, Math +3.6, HumanEval +3.1).
- **Type**: chart / comparison

### 11. fig_evolution_residual_connections
- **What**: Timeline/evolution diagram showing the progression from standard residuals (2016) to DenseNet (2016) to DenseFormer (2024) to ResFormer (2024) to DeepCrossAttention (2025) to AttnRes (2026). For each, show a small schematic of the connection pattern.
- **Type**: timeline / evolution

### 12. fig_running_example_standard_residual
- **What**: Concrete numerical example showing 4 tokens flowing through 4 layers with standard residuals. Show how h_l grows as vectors accumulate. Use small dimensions for clarity (e.g., d=4). Show actual vector values to demonstrate magnitude growth.
- **Type**: step-by-step / numerical walkthrough

### 13. fig_running_example_attnres
- **What**: Same 4 tokens, same 4 layers, but with AttnRes. Show the pseudo-query, dot products with previous layer outputs, softmax weights, and the resulting bounded hidden state. Contrast with the standard residual version.
- **Type**: step-by-step / numerical walkthrough

### 14. fig_pipeline_parallelism
- **What**: How Block AttnRes works in pipeline-parallel training. Show multiple GPU stages, each holding a block. Cache-based communication of block representations between stages. Two-phase computation within each stage.
- **Type**: system architecture / data flow

### 15. fig_depth_vs_width_tradeoff
- **What**: Show that standard residuals favor wider/shallower architectures while AttnRes enables narrower/deeper ones. Two iso-performance curves on a depth vs width grid, showing the optimal points shifting.
- **Type**: comparison / chart

---

## Running Example

For the article, use this consistent running example:

- **Tokens**: "The", "cat", "sat", "down" (4 tokens)
- **Embedding dimension**: d = 8
- **Input matrix**: X with shape (4, 8)
- **Number of layers**: L = 6
- **Number of blocks**: N = 2 (3 layers per block) for simplicity
- **Pseudo-query dimension**: d = 8 (same as embedding)

Trace through:
1. Standard residual: show h_0 through h_6 with growing magnitudes
2. AttnRes: show the same layers but with attention weights and bounded magnitudes
3. Block AttnRes: show intra-block addition and inter-block attention

Use concrete (small) numerical values to make the computation tangible.

---

## Key Sources

- [Attention Residuals (arXiv:2603.15031)](https://arxiv.org/abs/2603.15031) - Original paper by Kimi team at Moonshot AI (March 2026). Full mathematical formulation, scaling law experiments, Kimi Linear integration (48B/3B, 1.4T tokens), benchmark results.

- [MoonshotAI/Attention-Residuals GitHub](https://github.com/MoonshotAI/Attention-Residuals) - Official implementation with PyTorch pseudocode, Block AttnRes details, benchmark tables. 2.56k stars.

- [Kimi Attention Residuals Challenges Decade-Old Foundation (BigGo)](https://biggo.com/news/202603181054_Kimi_Attention_Residuals_AI_Efficiency_Breakthrough) - Accessible explanation of the depth-time duality, PreNorm dilution problem, 1.25x compute advantage, architectural implications for deeper/narrower models.

- [When Does Attention Residuals Work? (Ziming Liu)](https://kindxiaoming.github.io/blog/2026/attention-residual/) - Critical analysis by MIT/Caltech researcher. Shows AttnRes excels on structured tasks but not pure memorization. No Free Lunch perspective.

- [A New Way to Handle Residual Connections (Daily Dose of DS)](https://blog.dailydoseofds.com/p/a-new-way-to-handle-residual-connections) - Clear explanation of PreNorm dilution, Block AttnRes mechanism, and performance results.

- [Moonshot AI's Attention Residuals (NerdSchalk)](https://nerdschalk.com/moonshot-ais-attention-residuals-for-kimi-could-change-how-ai-models-use-layers/) - Overview of the problem, solution, and expected benefits.

- [HuggingFace Paper Page](https://huggingface.co/papers/2603.15031) - 145 upvotes, community discussion including author response about optimal block count (N=8).

- [DenseFormer (arXiv:2402.02622)](https://arxiv.org/abs/2402.02622) - Predecessor using Depth-Weighted Averaging with input-independent scalar weights. Important comparison point.

- [DeepCrossAttention (arXiv:2502.06785)](https://arxiv.org/abs/2502.06785) - Google's learnable input-dependent cross-layer weights. Claims 3x training speedup. Key competitor/related work.

- [Value Residual Learning / ResFormer (arXiv:2410.17897)](https://arxiv.org/abs/2410.17897) - Residual connections on value vectors to address attention concentration. ResFormer saves 10-14% parameters.

- [Emergent Mind Analysis](https://www.emergentmind.com/papers/2603.15031) - Technical summary with mathematical formulations and implementation details.
