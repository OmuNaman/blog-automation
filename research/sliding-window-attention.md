# Research: Sliding Window Attention

## Quick Summary

Sliding window attention (SWA) is a sparse attention mechanism that restricts each token to attending only to a fixed-size local window of neighboring tokens, reducing the computational complexity of self-attention from O(n^2) to O(n * w), where w is the window size. Introduced in the Longformer paper (Beltagy et al., 2020) and popularized in modern LLMs by Mistral 7B (2023), SWA exploits the insight that most useful attention is local, and that stacking multiple layers of local attention creates a large effective receptive field. This technique enables dramatic KV cache memory savings during inference through a rolling buffer design that discards keys/values outside the window.

---

## Core Concepts

### What is Sliding Window Attention?

Standard self-attention allows every token in a sequence to attend to every other token. This creates an attention matrix of size n x n (where n is the sequence length), requiring O(n^2) compute and memory. Sliding window attention restricts each token to only attend to its w nearest neighbors (typically the preceding w tokens in a causal/autoregressive setting). This creates a banded attention matrix where only a diagonal band of width w contains non-zero values.

### Why does it exist? What problem does it solve?

The quadratic cost of full attention is the primary bottleneck for processing long sequences. For a sequence of 32,768 tokens, full attention requires approximately 1.07 billion attention score computations. SWA with w=4,096 reduces this to approximately 134 million, an 8x reduction. Beyond compute, the KV cache during autoregressive generation grows linearly with sequence length in full attention. SWA caps the cache at a fixed size w, providing substantial memory savings that grow with sequence length.

### What did it replace or improve upon?

SWA replaced full O(n^2) attention with O(n*w) attention. It is part of a family of "efficient attention" mechanisms that also includes:
- **Sparse Transformer** (Child et al., 2019): Fixed sparse patterns (strided and local)
- **Longformer** (Beltagy et al., 2020): Sliding window + dilated window + global attention
- **BigBird** (Zaheer et al., 2020): Local window + random attention + global attention
- **Linear attention** variants (Katharopoulos et al., 2020): Kernel-based approximations

SWA is the simplest and most hardware-friendly of these approaches, which is why it was adopted by production LLMs like Mistral.

---

## How It Works

### Step 1: Constructing the Attention Mask

For a causal (autoregressive) transformer with sliding window size w, the attention mask for position i allows attending to positions max(0, i - w + 1) through i. This creates a banded lower-triangular matrix.

For a sequence of length 8 and window size w=3, the mask looks like:

```
Token:  0  1  2  3  4  5  6  7
  0  [  1  0  0  0  0  0  0  0 ]   <- token 0 attends to itself only
  1  [  1  1  0  0  0  0  0  0 ]   <- token 1 attends to 0,1
  2  [  1  1  1  0  0  0  0  0 ]   <- token 2 attends to 0,1,2
  3  [  0  1  1  1  0  0  0  0 ]   <- token 3 attends to 1,2,3 (window kicks in)
  4  [  0  0  1  1  1  0  0  0 ]   <- token 4 attends to 2,3,4
  5  [  0  0  0  1  1  1  0  0 ]   <- token 5 attends to 3,4,5
  6  [  0  0  0  0  1  1  1  0 ]   <- token 6 attends to 4,5,6
  7  [  0  0  0  0  0  1  1  1 ]   <- token 7 attends to 5,6,7
```

Compare to full causal attention where the entire lower triangle is filled with 1s.

### Step 2: Computing Attention Scores

The attention computation is identical to standard attention, but with the mask applied:

```
Attention(Q, K, V) = softmax((Q * K^T / sqrt(d_k)) + M) * V
```

Where M is the sliding window mask (0 for allowed positions, -infinity for blocked positions). After softmax, -infinity becomes exactly zero attention weight.

### Step 3: Information Propagation Across Layers

This is the key insight that makes SWA work despite the limited local window. In a transformer with L layers and window size w:

- At layer 1: token i can access tokens in range [i-w+1, i]
- At layer 2: token i can access tokens in range [i-2w+2, i] (because the tokens it attends to at layer 1 each had their own window at the layer below)
- At layer L: token i can access tokens in range [i-L*w+L, i]

The theoretical receptive field grows as: **R_L = L * w**

For Mistral 7B with 32 layers and w=4,096:
- R_32 = 32 * 4,096 = 131,072 tokens

This means that by the final layer, information from up to 131K tokens away can influence a token's representation, even though each individual layer only looks at 4,096 neighbors.

### Step 4: Rolling Buffer KV Cache (Inference)

During autoregressive generation, standard transformers store all past key-value pairs, growing linearly. With SWA, tokens beyond the window will never be attended to again, so their KV entries can be overwritten.

The rolling buffer uses modular indexing:
```
cache_position = current_position % window_size
```

This creates a fixed-size circular buffer of size w, regardless of how long the generated sequence becomes.

### Step 5: Pre-fill Chunking

For long input prompts, instead of processing the entire prompt in one forward pass (which would require O(n^2) memory for the attention matrix), Mistral processes the prompt in chunks of size w. Each chunk computes attention over its local window, and the KV cache is filled incrementally. This reduces peak memory during pre-fill from O(n^2) to O(w^2).

---

## Mathematical Foundation

### Standard Self-Attention

```
Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
```

Where Q, K, V are of shape (n, d_k), and Q * K^T produces an (n, n) matrix.

**Compute cost**: O(n^2 * d_k) for the matrix multiplication
**Memory cost**: O(n^2) for the attention matrix

### Sliding Window Attention

```
SWA(Q, K, V) = softmax((Q * K^T / sqrt(d_k)) + M_swa) * V
```

Where M_swa[i,j] = 0 if |i - j| < w and j <= i (causal), else -infinity.

**Compute cost**: O(n * w * d_k)
**Memory cost**: O(n * w) for the attention scores

### Effective Receptive Field

For a model with L layers and window size w:

```
R_L = L * w
```

This assumes each layer perfectly transmits information from its entire window, which is an upper bound. In practice, the effective receptive field may be smaller due to attention weight distribution.

### KV Cache Memory

**Full attention KV cache size:**
```
M_full = 2 * L * n * h_kv * d_h * bytes_per_element
```

**SWA rolling buffer KV cache size:**
```
M_swa = 2 * L * w * h_kv * d_h * bytes_per_element
```

For Mistral 7B (L=32, w=4096, h_kv=8 (GQA), d_h=128, fp16):
```
M_swa = 2 * 32 * 4096 * 8 * 128 * 2 bytes = 512 MB (fixed)
```

Compare to full attention at 16K tokens: 4,096 MB (growing with n).

### Memory Savings Ratio

```
Savings = 1 - (w / n)     when n > w
```

| Sequence Length (n) | Window (w=4096) | Savings |
|---------------------|-----------------|---------|
| 8,192               | 4,096           | 50%     |
| 16,384              | 4,096           | 75%     |
| 32,768              | 4,096           | 87.5%   |
| 65,536              | 4,096           | 93.75%  |

---

## Comparisons and Alternatives

### SWA vs Full Attention

| Aspect | Full Attention | Sliding Window Attention |
|--------|---------------|-------------------------|
| Compute per layer | O(n^2 * d) | O(n * w * d) |
| Memory per layer | O(n^2) | O(n * w) |
| KV cache | O(n) per layer, growing | O(w) per layer, fixed |
| Receptive field (1 layer) | Entire sequence | w tokens |
| Receptive field (L layers) | Entire sequence | L * w tokens |
| Hardware efficiency | Highly optimized (FlashAttention) | Less optimized (many small matmuls) |
| Accuracy on short sequences | Baseline | Comparable |
| Accuracy on long sequences | Quadratic cost | Linear cost, slight degradation possible |

### SWA vs Longformer (Sliding + Dilated + Global)

Longformer extends basic SWA with:
1. **Dilated sliding window**: Gaps of size d between attended positions, expanding receptive field to L * d * w without extra compute
2. **Global attention**: Pre-selected tokens (e.g., [CLS]) attend to all positions and all positions attend to them

Longformer is more expressive but more complex to implement. Modern LLMs (Mistral, Gemma) use simpler SWA, sometimes with a hybrid approach (alternating SWA and full attention layers).

### SWA vs BigBird (Local + Random + Global)

BigBird adds random attention connections on top of local windows and global tokens. The random connections are theoretically justified: they ensure O(log n) information propagation paths, making the sparse attention a universal approximator.

BigBird complexity: O(w*n + r*n + g*n) where r = random connections, g = global tokens.

### Hybrid Approaches in Modern LLMs

**Gemma 3 (2025)**: 5:1 ratio of sliding window layers to full attention layers. Window size = 1,024 tokens. Only 1 global (full attention) layer for every 5 local (SWA) layers. KV cache reduced by approximately 5x with less than 0.3% perplexity loss.

**Gemma 2 (2024)**: 1:1 ratio, alternating SWA (w=4,096) and full attention layers.

**Mistral 7B (2023)**: All layers use SWA with w=4,096. Combined with GQA (4:1 ratio, 32 query heads, 8 KV heads). Total KV cache reduction: 8x (4x from GQA, 2x from SWA).

**Mistral Small 3.1 (2025)**: Dropped SWA entirely, returning to full GQA + FlashAttention. Suggests that for some model sizes and use cases, optimized full attention may be preferable.

### Quantitative Benchmarks (Mistral 7B)

| Model | Parameters | MMLU | MMLU/B (efficiency) |
|-------|-----------|------|---------------------|
| LLaMA 2 7B | 7B | 46.8 | 6.69 |
| Mistral 7B | 7.3B | 60.1 | 8.23 |
| LLaMA 2 13B | 13B | 54.8 | 4.22 |

Mistral 7B achieves LLaMA 2 13B-level performance with 44% fewer parameters.

At 8K tokens, Mistral's KV cache is 8x smaller than LLaMA 2's (4x from GQA + 2x from SWA).

### SWAT: SWA During Training (Feb 2025)

Recent research (arXiv:2502.18845) proposes training with SWA from the start rather than only using it at inference. Key innovations:
- Replace softmax with sigmoid to avoid "attention sink" problem
- Use balanced ALiBi (half forward-looking, half backward-looking slopes)
- Add RoPE for positional stability
- 340M model achieves 46.88% avg accuracy vs 42.92% for vanilla transformers
- Maintains consistent performance across varying sequence lengths

---

## Visual Opportunities

### 1. fig_full_vs_swa_attention_matrix
- **What it shows**: Side-by-side comparison of a full causal attention matrix (lower triangle filled) vs a sliding window attention matrix (banded diagonal). Use a concrete 8x8 example with colored cells for attended positions and white for masked positions.
- **Type**: comparison

### 2. fig_swa_mask_construction
- **What it shows**: Step-by-step construction of the sliding window mask for an 8-token sequence with w=3. Show the causal mask, the window mask, and their intersection (element-wise AND). Highlight how the band forms.
- **Type**: step-by-step

### 3. fig_receptive_field_growth
- **What it shows**: How the receptive field grows across layers. Show 3-4 layers stacked, with token 7 at the top layer. Draw expanding cones showing which tokens from layer 0 can influence token 7 at each layer. Label R_1=w, R_2=2w, R_3=3w.
- **Type**: architecture / step-by-step

### 4. fig_rolling_buffer_kv_cache
- **What it shows**: The rolling/circular buffer mechanism for the KV cache during autoregressive generation. Show a fixed-size buffer array of size w, with an arrow showing the write position wrapping around. Show old entries being overwritten as new tokens arrive.
- **Type**: step-by-step / architecture

### 5. fig_kv_cache_memory_comparison
- **What it shows**: Bar chart or visual comparison of KV cache memory usage: full attention (growing with n) vs SWA (fixed at w) vs SWA+GQA (even smaller). Use concrete numbers for n=8K, 16K, 32K.
- **Type**: comparison

### 6. fig_attention_score_computation
- **What it shows**: The Q*K^T computation for a single query token with SWA. Show how only w key vectors participate instead of all n. Include the mask application, softmax, and multiplication with V.
- **Type**: step-by-step / matrix-operation

### 7. fig_prefill_chunking
- **What it shows**: How a long prompt (e.g., 8K tokens) is split into chunks of size w for pre-fill. Show chunks being processed sequentially, each building on the KV cache from the previous chunk.
- **Type**: step-by-step / architecture

### 8. fig_sparse_attention_family
- **What it shows**: A taxonomy/comparison of sparse attention patterns: full attention, sliding window (Longformer basic), dilated sliding window, sliding + global (Longformer full), local + random + global (BigBird). Show the attention matrix pattern for each.
- **Type**: comparison

### 9. fig_hybrid_architecture
- **What it shows**: How modern models like Gemma 3 alternate between SWA layers and full attention layers. Show a stack of transformer layers with different colors for SWA (local) vs full (global) layers, annotated with the 5:1 ratio.
- **Type**: architecture

### 10. fig_information_flow_path
- **What it shows**: How information from a distant token (e.g., position 0) reaches a later token (e.g., position 15) through multiple layers of SWA. Trace the path through intermediate tokens across layers, showing the "relay" effect.
- **Type**: flowchart / step-by-step

---

## Running Example

Throughout the article, we will use the following consistent example:

- **Sequence**: 8 tokens: "The", "quick", "brown", "fox", "jumps", "over", "the", "fence"
- **Sequence length**: n = 8
- **Embedding dimension**: d_model = 16
- **Number of attention heads**: h = 4
- **Head dimension**: d_k = d_v = 4
- **Sliding window size**: w = 3
- **Number of layers**: L = 4

**Input matrix shape**: (8, 16), representing 8 tokens each with a 16-dimensional embedding.

After projection for a single head:
- Q shape: (8, 4)
- K shape: (8, 4)
- V shape: (8, 4)

**Full attention Q*K^T**: (8, 8) = 64 score computations
**SWA Q*K^T effective**: approximately 8*3 = 24 score computations (with mask zeroing the rest)
**Savings**: 62.5% reduction in meaningful computations

**Theoretical receptive field**: R_4 = 4 * 3 = 12 tokens (exceeds our sequence length of 8, so effectively full coverage by layer 4)

For demonstrating KV cache savings, we will also show a longer inference example:
- **Generation length**: 32 tokens
- **Full attention KV cache at step 32**: stores 32 entries per layer
- **SWA rolling buffer at step 32**: stores 3 entries per layer (fixed)
- **Savings**: 90.6%

---

## Key Sources

### Primary Papers

- **[Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150)** (Beltagy, Peters, Cohan, 2020)
  - Introduced sliding window attention for transformers
  - Combined with dilated sliding window and global attention
  - Three attention patterns: sliding, dilated sliding, global+sliding
  - Showed linear scaling enables processing thousands of tokens
  - State-of-the-art on WikiHop and TriviaQA at the time

- **[Mistral 7B](https://arxiv.org/pdf/2310.06825)** (Jiang et al., September 2023)
  - Popularized SWA in modern decoder-only LLMs
  - Window size w=4,096, context length 8,192
  - Rolling buffer KV cache with modular indexing
  - Combined with GQA for 8x total KV cache reduction
  - Pre-fill chunking for memory-efficient prompt processing
  - Achieved LLaMA 2 13B performance with 7B parameters

- **[BigBird: Transformers for Longer Sequences](https://papers.neurips.cc/paper_files/paper/2020/file/c8512d142a2d849725f31a9a7a361ab9-Paper.pdf)** (Zaheer et al., 2020)
  - Local window + random attention + global attention
  - Proved sparse attention can approximate full attention (universal approximator)
  - Block-level implementation for hardware efficiency

- **[SWAT: Sliding Window Attention Training](https://arxiv.org/abs/2502.18845)** (Feb 2025)
  - Proposes training with SWA from the start (not just inference)
  - Replaces softmax with sigmoid to avoid attention sink
  - Uses balanced ALiBi + RoPE for position encoding
  - Shows consistent performance across varying sequence lengths

### Blog Posts and Explanations

- **[Amaarora: Sliding Window Attention with Animations and PyTorch](https://amaarora.github.io/posts/2024-07-04%20SWA.html)**
  - Chunked computation approach for efficient implementation
  - Overlapping chunks of size 2w with w overlap
  - PyTorch einsum implementation: `bcxd,bcyd->bcxy`
  - Shows the _chunk function implementation

- **[Naoki Shibuya: Longformer (2020)](https://naokishibuya.github.io/blog/2022-11-27-longformer-2020/index.html)**
  - Clear explanation of three attention patterns
  - Receptive field formula: l * w (sliding), l * d * w (dilated)
  - Dual projection sets: (Qs, Ks, Vs) and (Qg, Kg, Vg)

- **[Michael Brenndoerfer: Mistral Architecture](https://mbrenndoerfer.com/writing/mistral-architecture-sliding-window-attention)**
  - Detailed layer-by-layer receptive field analysis
  - Rolling buffer cache formula: cache_position = current_position % w
  - KV cache memory calculation: 512 MB fixed for Mistral 7B
  - Pre-fill chunking memory savings: 74.6% at 8K tokens

- **[Sebastian Raschka: The Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison)**
  - Gemma 3: 5:1 SWA-to-full ratio, w=1,024
  - Gemma 2: 1:1 alternating, w=4,096
  - Mistral Small 3.1: dropped SWA entirely
  - Trend: hybrid approaches are evolving rapidly

- **[Omri Mallis: Techniques for KV Cache Optimization](https://www.omrimallis.com/posts/techniques-for-kv-cache-optimization/)**
  - SWA provides 2x cache reduction
  - Combined with GQA (4x) for 8x total in Mistral
  - Comparison table of cache optimization techniques

- **[HuggingFace: Understanding BigBird's Block Sparse Attention](https://huggingface.co/blog/big-bird)**
  - Block-level implementation details
  - Shifting trick for efficient local attention
  - Graph-theoretic justification for sparse attention
  - O(log n) information propagation with random connections

---

## Historical Context and Evolution

### Timeline

1. **2019 - Sparse Transformer** (Child et al.): First demonstration that fixed sparse attention patterns could maintain model quality while reducing computation. Used strided and local patterns.

2. **2020 - Longformer** (Beltagy et al.): Introduced sliding window attention as a formal mechanism. Combined with dilated windows and task-specific global attention. Focused on encoder models for document understanding.

3. **2020 - BigBird** (Zaheer et al.): Extended sparse attention with random connections. Proved theoretically that sparse attention with global + local + random is a universal approximator.

4. **2023 - Mistral 7B** (Jiang et al.): Brought SWA to decoder-only generative LLMs. Introduced rolling buffer KV cache. Combined with GQA. Showed SWA is practical for production-scale models.

5. **2024 - Gemma 2** (Google): Hybrid approach, alternating SWA and full attention layers (1:1 ratio). Window size 4,096.

6. **2025 - Gemma 3** (Google): More aggressive hybrid ratio (5:1 SWA-to-full). Window size reduced to 1,024. Less than 0.3% perplexity loss.

7. **2025 - Mistral Small 3.1**: Dropped SWA entirely, returning to full attention with FlashAttention. Suggests the landscape is still evolving.

8. **2025 - SWAT paper**: Proposes aligning training and inference by using SWA during training. Replaces softmax with sigmoid to avoid attention sink issues.

### Key Insight

The evolution shows a pendulum: from full attention (original Transformer) to sparse (Longformer/BigBird) to hybrid (Gemma) and in some cases back to full (Mistral Small 3.1 with FlashAttention). The optimal choice depends on the specific model size, target context length, and inference hardware.

---

## Implementation Notes

### Mask Construction (PyTorch)

```python
def create_swa_mask(seq_len, window_size):
    """Create a causal sliding window attention mask."""
    # Start with causal mask
    mask = torch.tril(torch.ones(seq_len, seq_len))
    # Apply window constraint
    for i in range(seq_len):
        for j in range(seq_len):
            if i - j >= window_size:
                mask[i, j] = 0
    # Convert to attention mask (0 -> -inf)
    mask = mask.masked_fill(mask == 0, float('-inf'))
    mask = mask.masked_fill(mask == 1, 0.0)
    return mask
```

More efficient vectorized version:
```python
def create_swa_mask_efficient(seq_len, window_size):
    """Efficient sliding window causal mask."""
    row_idx = torch.arange(seq_len).unsqueeze(1)
    col_idx = torch.arange(seq_len).unsqueeze(0)
    # Causal: col <= row, Window: row - col < window_size
    mask = (col_idx <= row_idx) & (row_idx - col_idx < window_size)
    return mask.float().masked_fill(~mask, float('-inf')).masked_fill(mask, 0.0)
```

### Chunked Computation (Longformer style)

```python
def _chunk(hidden_states, window_overlap):
    """Split sequence into overlapping chunks of size 2*w."""
    chunk_size = [
        hidden_states.size(0),  # batch
        hidden_states.size(1) // window_overlap - 1,  # num_chunks
        window_overlap * 2,  # chunk_width
        hidden_states.size(2),  # embedding_dim
    ]
    overlapping_chunks = torch.empty(chunk_size)
    for chunk in range(chunk_size[1]):
        overlapping_chunks[:, chunk, :, :] = hidden_states[
            :, chunk * window_overlap : chunk * window_overlap + 2 * window_overlap, :
        ]
    return overlapping_chunks

# Attention within chunks
scores = torch.einsum("bcxd,bcyd->bcxy", query_chunks, key_chunks)
```

### Rolling Buffer Cache (Mistral style)

```python
class RollingKVCache:
    def __init__(self, window_size, num_layers, num_kv_heads, head_dim):
        self.window_size = window_size
        self.k_cache = torch.zeros(num_layers, window_size, num_kv_heads, head_dim)
        self.v_cache = torch.zeros(num_layers, window_size, num_kv_heads, head_dim)
        self.position = 0

    def update(self, layer_idx, new_k, new_v):
        """Write new KV pair at circular buffer position."""
        idx = self.position % self.window_size
        self.k_cache[layer_idx, idx] = new_k
        self.v_cache[layer_idx, idx] = new_v

    def advance(self):
        self.position += 1

    def get_keys(self, layer_idx):
        """Return all valid keys in cache for this layer."""
        if self.position < self.window_size:
            return self.k_cache[layer_idx, :self.position]
        return self.k_cache[layer_idx]  # full buffer
```

### FlexAttention (PyTorch 2.5+)

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def sliding_window(b, h, q_idx, kv_idx):
    """Mask function for sliding window attention."""
    causal = q_idx >= kv_idx
    window = q_idx - kv_idx < WINDOW_SIZE
    return causal & window

block_mask = create_block_mask(sliding_window, B=1, H=1, Q_LEN=seq_len, KV_LEN=seq_len)
output = flex_attention(query, key, value, block_mask=block_mask)
```
