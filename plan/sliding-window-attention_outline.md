# Article Plan: Sliding Window Attention

## Title
**Sliding Window Attention: How Modern LLMs See the World Through a Narrow Lens**

## This Article Covers
- **The quadratic bottleneck**: Why standard self-attention becomes prohibitively expensive for long sequences, and how sliding window attention solves it by restricting each token to a fixed local neighborhood.
- **The mechanics of SWA**: A hands-on, step-by-step walkthrough of how the sliding window mask is constructed, how attention scores are computed within the window, and how information still flows globally through stacked layers.
- **The rolling buffer KV cache**: How Mistral 7B exploits sliding window attention to build a fixed-size circular buffer that caps memory usage during autoregressive generation, regardless of sequence length.
- **The family of sparse attention**: How SWA relates to Longformer's dilated and global attention, BigBird's random connections, and the hybrid architectures used by Gemma 2 and Gemma 3.
- **Quantifying the gains**: Concrete numbers showing compute savings, memory reduction, and benchmark performance for real-world models.

## Running Example

Throughout the article, we use a consistent, simple example:

- **Sequence**: 8 tokens: "The", "cat", "sat", "on", "the", "warm", "soft", "mat"
- **Sequence length**: n = 8
- **Embedding dimension**: d_model = 8
- **Number of attention heads**: h = 2
- **Head dimension**: d_k = d_v = 4
- **Sliding window size**: w = 3
- **Number of layers**: L = 4

**Input matrix shape**: (8, 8), representing 8 tokens each with an 8-dimensional embedding.

After projection for a single head:
- Q shape: (8, 4)
- K shape: (8, 4)
- V shape: (8, 4)

**Full attention**: Q*K^T = (8, 8) = 64 score computations
**SWA attention**: Effective scores = 8 * 3 = 24 computations (62.5% reduction)

**Receptive field**: R_4 = 4 * 3 = 12 tokens (exceeds n=8, so full coverage by layer 4)

For the KV cache section, we extend to a longer inference scenario:
- **Generation**: 20 tokens total
- **Full attention cache at step 20**: 20 entries per layer (growing)
- **SWA rolling buffer at step 20**: 3 entries per layer (fixed)

---

## Diagram Master List

| #  | ID | Section | Type | Caption |
|----|-----|---------|------|---------|
| 1  | fig_roadmap | Opening | architecture | Figure 1 The landscape of attention mechanisms in transformers. This article focuses on sliding window attention, highlighted in the center, and its relationship to full attention, sparse attention variants, and modern hybrid approaches. |
| 2  | fig_full_attention_cost | 1 - The Problem | comparison | Figure 2 The quadratic cost of full self-attention. For a sequence of n tokens, every token attends to every other token, producing an n x n attention matrix. As n grows, the number of computations explodes. |
| 3  | fig_full_causal_attention_matrix | 1 - The Problem | matrix-op | Figure 3 The full causal attention matrix for our 8-token sequence. Every token attends to itself and all preceding tokens, forming a lower-triangular matrix with 36 active attention scores. |
| 4  | fig_swa_core_idea | 2 - The Core Idea | architecture | Figure 4 The core idea of sliding window attention. Instead of attending to all previous tokens, each token only looks at its w nearest neighbors. The attention "window" slides along the sequence, giving each token a limited but consistent local view. |
| 5  | fig_full_vs_swa_side_by_side | 2 - The Core Idea | comparison | Figure 5 Full causal attention (left) versus sliding window attention with w=3 (right) for our 8-token sequence. The full attention matrix has 36 active cells; the sliding window matrix has only 21, a 42% reduction. The banded diagonal pattern is the signature of SWA. |
| 6  | fig_causal_mask | 3 - Building the Mask | step-by-step | Figure 6 Step 1 of mask construction: the causal mask. This lower-triangular matrix ensures that each token can only attend to itself and previous tokens, never to future tokens. |
| 7  | fig_window_mask | 3 - Building the Mask | step-by-step | Figure 7 Step 2 of mask construction: the window distance mask. For window size w=3, a token at position i can attend to positions within distance w-1=2 to its left. This creates a banded matrix centered on the diagonal. |
| 8  | fig_combined_swa_mask | 3 - Building the Mask | step-by-step | Figure 8 Step 3: the final sliding window attention mask, formed by the intersection of the causal mask and the window mask. Only positions that satisfy both constraints (causal AND within window) receive attention. |
| 9  | fig_mask_to_scores | 3 - Building the Mask | step-by-step | Figure 9 Converting the binary mask to attention scores. Allowed positions receive a mask value of 0 (no change to the score), while blocked positions receive negative infinity. After softmax, negative infinity becomes exactly zero attention weight. |
| 10 | fig_qk_full_computation | 4 - Computing Attention | matrix-op | Figure 10 The full Q*K^T matrix multiplication for standard attention. All 8 query vectors multiply against all 8 key vectors, producing 64 raw attention scores before masking. |
| 11 | fig_qk_swa_computation | 4 - Computing Attention | matrix-op | Figure 11 The Q*K^T computation with sliding window attention. Only the scores within the window band are meaningful; all others will be masked to zero. The effective computation involves just 24 scores instead of 64. |
| 12 | fig_single_token_attention | 4 - Computing Attention | step-by-step | Figure 12 A detailed walkthrough for token "soft" (position 6). Its query vector multiplies with only three key vectors (positions 4, 5, 6), producing three attention scores. After softmax normalization over just these three scores, the weighted sum of three value vectors produces the output. |
| 13 | fig_softmax_comparison | 4 - Computing Attention | comparison | Figure 13 Softmax normalization comparison. In full attention, token 6 distributes its attention weight across 7 positions. In SWA, the same token concentrates its attention across only 3 positions, creating sharper, more focused attention weights within the window. |
| 14 | fig_output_computation | 4 - Computing Attention | matrix-op | Figure 14 The final output computation. The attention weights (after softmax and masking) multiply the value matrix V, producing the output for each token. Each output is a weighted combination of only w value vectors. |
| 15 | fig_single_layer_view | 5 - The Layer Stacking Trick | architecture | Figure 15 The view from a single layer of sliding window attention with w=3. Token "mat" (position 7) can only see tokens at positions 5, 6, and 7. It has no direct access to any earlier tokens. |
| 16 | fig_two_layer_view | 5 - The Layer Stacking Trick | architecture | Figure 16 After two layers, the effective receptive field expands. Token "mat" at layer 2 attends to positions 5, 6, 7 at layer 1, but those tokens themselves attended to positions 3, 4, 5, 6, 7 at layer 0. The receptive field has grown from 3 to 5 tokens. |
| 17 | fig_four_layer_receptive_field | 5 - The Layer Stacking Trick | architecture | Figure 17 The expanding cone of influence across all four layers. By layer 4, token "mat" has indirect access to information from all 8 tokens in our sequence, even though each individual layer only looks at 3 neighbors. The receptive field formula is R_L = L * w. |
| 18 | fig_information_relay | 5 - The Layer Stacking Trick | flowchart | Figure 18 The information relay effect. Information from token "The" (position 0) reaches token "mat" (position 7) through a chain of intermediate tokens across layers. The path passes through at least 3 relay points, highlighted in the diagram. |
| 19 | fig_receptive_field_formula | 5 - The Layer Stacking Trick | comparison | Figure 19 Receptive field growth for Mistral 7B. With 32 layers and w=4096, the theoretical receptive field reaches 131,072 tokens by the final layer, far exceeding the 8,192-token context window. Each layer adds w tokens to the receptive field. |
| 20 | fig_kv_cache_standard_growing | 6 - The Rolling Buffer | step-by-step | Figure 20 The standard KV cache during autoregressive generation. At each step, a new key-value pair is appended. After generating 20 tokens, the cache stores 20 entries per layer, and continues growing indefinitely. |
| 21 | fig_rolling_buffer_concept | 6 - The Rolling Buffer | architecture | Figure 21 The rolling buffer KV cache. A fixed-size circular buffer of size w=3 stores only the most recent key-value pairs. When the buffer is full, new entries overwrite the oldest ones using modular indexing: position = step % w. |
| 22 | fig_rolling_buffer_walkthrough | 6 - The Rolling Buffer | step-by-step | Figure 22 A step-by-step walkthrough of the rolling buffer for 8 generation steps with w=3. Steps 0-2 fill the buffer normally. At step 3, position 0 is overwritten. At step 4, position 1 is overwritten. The buffer always contains the 3 most recent entries. |
| 23 | fig_kv_cache_memory_comparison | 6 - The Rolling Buffer | comparison | Figure 23 KV cache memory usage comparison. Full attention cache grows linearly with sequence length (blue line). The SWA rolling buffer remains fixed at w entries (orange line). At 32K tokens with w=4096, the rolling buffer uses 87.5% less memory. |
| 24 | fig_mistral_combined_savings | 6 - The Rolling Buffer | comparison | Figure 24 Combined memory savings in Mistral 7B. GQA reduces the cache by 4x (32 query heads sharing 8 KV heads). SWA reduces it by another 2x (rolling buffer of size 4096 vs full 8192 context). The total reduction is 8x compared to standard multi-head attention with full context. |
| 25 | fig_prefill_problem | 7 - Pre-fill Chunking | comparison | Figure 25 The pre-fill memory problem. Processing an 8K-token prompt in a single forward pass requires materializing the full attention matrix, consuming massive memory. Chunked pre-fill processes the prompt in window-sized pieces, dramatically reducing peak memory. |
| 26 | fig_prefill_chunks_sequential | 7 - Pre-fill Chunking | step-by-step | Figure 26 Chunked pre-fill in action. An 8-token prompt is processed in chunks of size w=3. Each chunk computes attention only within its local window and updates the rolling buffer KV cache. The next chunk continues from where the previous one left off. |
| 27 | fig_sparse_attention_taxonomy | 8 - The Family | comparison | Figure 27 A taxonomy of sparse attention patterns, shown as attention matrices. From left to right: full attention (lower triangle), sliding window (banded diagonal), dilated sliding window (banded with gaps), sliding + global (band plus full rows/columns for special tokens), and BigBird (band + global + random). |
| 28 | fig_dilated_sliding_window | 8 - The Family | matrix-op | Figure 28 Longformer's dilated sliding window. Instead of attending to consecutive neighbors, every other position within the window is skipped (dilation d=2). This doubles the effective receptive field per layer without increasing the number of attention scores computed. |
| 29 | fig_global_attention | 8 - The Family | architecture | Figure 29 Longformer's global attention mechanism. Pre-selected tokens (such as [CLS]) attend to all positions in the sequence, and all positions attend to them. These global tokens act as information hubs, enabling long-range communication alongside local sliding window attention. |
| 30 | fig_bigbird_three_components | 8 - The Family | architecture | Figure 30 BigBird's three attention components. Local windowed attention (blue) captures nearby context. Global attention tokens (green) connect to all positions. Random attention connections (orange) create shortcut paths, ensuring information can propagate across the full sequence in O(log n) layers. |
| 31 | fig_hybrid_layer_stack | 9 - Hybrid Architectures | architecture | Figure 31 Hybrid architecture design as used in Gemma 3. A stack of transformer layers alternates between sliding window attention layers (blue, 5 layers) and full attention layers (green, 1 layer) in a 5:1 ratio. The full attention layers act as periodic "global checkpoints" that allow long-range information to flow freely. |
| 32 | fig_hybrid_evolution | 9 - Hybrid Architectures | comparison | Figure 32 The evolution of hybrid attention strategies across model generations. Gemma 2 (2024) uses a 1:1 alternating ratio. Gemma 3 (2025) uses a more aggressive 5:1 ratio with a smaller window. Mistral Small 3.1 (2025) dropped SWA entirely, returning to full attention with FlashAttention. |
| 33 | fig_compute_scaling_curves | 10 - Quantifying the Gains | chart | Figure 33 Compute scaling comparison. Full attention grows quadratically (O(n^2)), while sliding window attention grows linearly (O(n*w)). At sequence length 32K with w=4096, SWA requires 8x fewer attention score computations. |
| 34 | fig_memory_savings_table | 10 - Quantifying the Gains | comparison | Figure 34 KV cache memory savings at different sequence lengths for w=4096. At 8K tokens, SWA saves 50%. At 16K tokens, 75%. At 32K tokens, 87.5%. At 64K tokens, 93.75%. The longer the sequence, the greater the benefit. |
| 35 | fig_code_mask_visualization | 11 - Implementation | matrix-op | Figure 35 Visualization of the attention mask produced by our PyTorch implementation. The colored band along the diagonal represents the sliding window (w=3) within the causal constraint. White cells represent masked positions that receive negative infinity before softmax. |

**Total: 35 diagrams**

---

## Section Outline

### Opening

**Title**: Sliding Window Attention: How Modern LLMs See the World Through a Narrow Lens

**"This article covers" bullets**: (listed above)

**Bridge paragraph**: If you have read about the standard transformer architecture and understand how self-attention works (queries, keys, values, and the softmax-weighted sum), you have everything you need. We will build sliding window attention from scratch, starting with the problem it solves.

**Figures**: fig_roadmap

**Transition**: "Let's begin with the problem that motivated sliding window attention in the first place."

---

### Section 1: The quadratic bottleneck (Head 1)

The problem with full self-attention and why it breaks down for long sequences.

#### 1.1: Every token talks to every other token (Head 2)
**Key Points**:
- In standard self-attention, every token computes an attention score with every other token
- This produces an n x n attention matrix
- For our 8-token example, that is 8 x 8 = 64 scores (with masking, 36 active)
- For real sequences (4K, 8K, 32K tokens), this scales quadratically
**Figures**: fig_full_attention_cost, fig_full_causal_attention_matrix
**Transition**: "The quadratic cost creates two separate problems: computation time and memory."

#### 1.2: The memory wall (Head 2)
**Key Points**:
- The attention matrix itself consumes O(n^2) memory
- The KV cache during generation grows linearly with every new token
- At 32K tokens, a standard model needs to store 32K key-value pairs per layer
- Concrete numbers: LLaMA 2 7B at 8K tokens requires significant KV cache memory
**Figures**: (uses fig_full_attention_cost from above, text-driven section)
**Transition**: "What if most of this computation is wasted? What if tokens don't actually need to see the entire sequence?"

---

### Section 2: The core idea of sliding window attention (Head 1)

The insight that most useful attention is local, and the solution.

#### 2.1: Most attention is local (Head 2)
**Key Points**:
- Empirical observation: in trained transformers, attention weights are heavily concentrated on nearby tokens
- A token's meaning is most influenced by its immediate neighbors
- Distant tokens contribute diminishing attention weight
- This suggests we can restrict attention to a local window without losing much
**Figures**: fig_swa_core_idea
**Transition**: "This is the core insight behind sliding window attention. Let's see exactly what it looks like."

#### 2.2: From full to windowed: the visual transformation (Head 2)
**Key Points**:
- Side-by-side comparison: full causal matrix vs SWA matrix
- The banded diagonal pattern: the signature of SWA
- Count the active cells: 36 (full) vs 21 (SWA with w=3) for our 8-token example
- The window "slides" along the diagonal as we move through the sequence
**Figures**: fig_full_vs_swa_side_by_side
**Transition**: "Now that we have the core intuition, it's time to open the black box. Let's build the sliding window mask step by step."

---

### Section 3: Building the sliding window mask (Head 1)

A hands-on, step-by-step construction of the SWA mask.

#### 3.1: Step 1, the causal constraint (Head 2)
**Key Points**:
- The causal mask ensures autoregressive property: no peeking at future tokens
- Lower-triangular matrix where M[i,j] = 1 if j <= i
- For our 8-token example, this is the familiar lower triangle
- This alone gives us full causal attention (what we want to improve)
**Figures**: fig_causal_mask
**Transition**: "The causal mask is necessary but not sufficient. We need a second constraint."

#### 3.2: Step 2, the window distance constraint (Head 2)
**Key Points**:
- The window mask restricts how far back each token can look
- For window size w=3, token i can attend to positions where i - j < w (i.e., distance < 3)
- This creates a band of width w centered on the diagonal
- Unlike the causal mask, this band extends in both directions (but we will intersect with causal)
**Figures**: fig_window_mask
**Transition**: "The final SWA mask is simply the intersection of these two constraints."

#### 3.3: Step 3, the intersection (Head 2)
**Key Points**:
- SWA mask = causal mask AND window mask
- A position is attended to only if it satisfies BOTH constraints
- The result is a banded lower-triangular matrix
- Walk through each row of our 8-token example, showing which positions are active
**Figures**: fig_combined_swa_mask
**Transition**: "We have our mask. But how does it actually affect the attention computation?"

#### 3.4: From mask to scores: the negative infinity trick (Head 2)
**Key Points**:
- Convert binary mask to additive mask: 1 becomes 0, 0 becomes -infinity
- Add this mask to the raw Q*K^T scores before softmax
- After softmax, e^(-infinity) = 0, so masked positions contribute exactly zero attention
- This is the standard "masked attention" pattern, with SWA simply using a different mask shape
**Figures**: fig_mask_to_scores
**Transition**: "Now let's trace the full attention computation through our running example."

---

### Section 4: Computing attention scores with a sliding window (Head 1)

The complete mathematical walkthrough with concrete numbers.

#### 4.1: The Q*K^T multiplication (Head 2)
**Key Points**:
- Q shape (8, 4) times K^T shape (4, 8) produces (8, 8) scores matrix
- In full attention, all 64 cells are computed
- In SWA, we still compute the full matrix but most entries will be masked
- (In optimized implementations, the masked entries are never computed at all)
**Figures**: fig_qk_full_computation, fig_qk_swa_computation
**Transition**: "Let's zoom in on a single token to see exactly what happens."

#### 4.2: A single token's journey through SWA (Head 2)
**Key Points**:
- Focus on token "soft" (position 6, w=3)
- It attends to positions 4 ("the"), 5 ("warm"), 6 ("soft")
- Compute 3 attention scores: q_6 dot k_4, q_6 dot k_5, q_6 dot k_6
- Scale by 1/sqrt(d_k) = 1/sqrt(4) = 1/2
- Apply softmax over just these 3 scores (not 7)
- Multiply by value vectors v_4, v_5, v_6 to get the output
**Figures**: fig_single_token_attention
**Transition**: "Notice something important about the softmax normalization."

#### 4.3: Sharper attention through a smaller window (Head 2)
**Key Points**:
- In full attention, softmax distributes weight across up to 7 positions
- In SWA, softmax distributes weight across only 3 positions
- The result: more concentrated, sharper attention weights within the window
- This can actually be beneficial, forcing the model to focus on the most relevant nearby context
**Figures**: fig_softmax_comparison
**Transition**: "Let's see the complete output computation for all tokens."

#### 4.4: The output matrix (Head 2)
**Key Points**:
- Each token's output is a weighted sum of at most w value vectors
- The output matrix has the same shape as the input: (8, 4)
- Shape summary: Q(8,4) * K^T(4,8) -> Scores(8,8) -> Masked+Softmax -> Weights(8,8) * V(8,4) -> Output(8,4)
**Figures**: fig_output_computation
**Transition**: "We have successfully built a working sliding window attention layer. But a critical question remains: if each layer can only see w=3 tokens, how does the model understand the full sequence?"

---

### Section 5: The layer stacking trick: how local becomes global (Head 1)

The key insight that makes SWA viable despite the limited window.

#### 5.1: The view from a single layer (Head 2)
**Key Points**:
- At layer 1, token "mat" (position 7) sees only positions 5, 6, 7
- It has zero direct access to "The" (position 0) or "cat" (position 1)
- This seems like a severe limitation
**Figures**: fig_single_layer_view
**Transition**: "But remember, transformers have many layers. Let's see what happens when we stack them."

#### 5.2: Two layers: the receptive field expands (Head 2)
**Key Points**:
- At layer 2, token "mat" attends to positions 5, 6, 7 from layer 1
- But position 5 at layer 1 attended to positions 3, 4, 5 from layer 0
- So token "mat" at layer 2 has indirect access to positions 3 through 7
- The receptive field grew from 3 tokens to 5 tokens
**Figures**: fig_two_layer_view
**Transition**: "This pattern continues with every additional layer."

#### 5.3: Four layers: full coverage for our example (Head 2)
**Key Points**:
- The receptive field formula: R_L = L * w
- At layer 4: R_4 = 4 * 3 = 12, which exceeds our sequence length of 8
- By layer 4, every token has indirect access to the entire sequence
- Visualize as an expanding cone of influence
**Figures**: fig_four_layer_receptive_field
**Transition**: "Let's trace exactly how information from a distant token reaches through this relay."

#### 5.4: The information relay (Head 2)
**Key Points**:
- Trace "The" (position 0) reaching "mat" (position 7) across 4 layers
- Layer 0: "The" contributes to "sat" (position 2) via attention
- Layer 1: "sat" contributes to "the" (position 4)
- Layer 2: "the" contributes to "soft" (position 6)
- Layer 3: "soft" contributes to "mat" (position 7)
- The information passes through relay tokens at each layer
**Figures**: fig_information_relay
**Transition**: "This relay mechanism is powerful, but it is an upper bound."

#### 5.5: Scaling to real models (Head 2)
**Key Points**:
- Mistral 7B: 32 layers, w=4096, R_32 = 131,072 tokens
- This far exceeds the 8,192 context window
- In practice, the effective receptive field is smaller than the theoretical maximum
- The quality of information degrades with each relay step (unlike full attention's direct path)
**Figures**: fig_receptive_field_formula
**Transition**: "Now that we understand how SWA works during the forward pass, let's see how it transforms inference efficiency with the rolling buffer KV cache."

---

### Section 6: The rolling buffer KV cache (Head 1)

How SWA enables dramatic memory savings during autoregressive generation.

#### 6.1: The problem with standard KV caches (Head 2)
**Key Points**:
- During autoregressive generation, each new token needs to attend to all previous tokens
- Standard approach: store every past key and value vector in a growing cache
- After generating 1000 tokens, the cache has 1000 entries per layer
- Memory grows linearly and unboundedly with generation length
**Figures**: fig_kv_cache_standard_growing
**Transition**: "Sliding window attention changes this equation entirely."

#### 6.2: The circular buffer design (Head 2)
**Key Points**:
- With SWA, a token at position t only attends to positions [t-w+1, t]
- Tokens at positions < t-w+1 will NEVER be attended to again
- Their KV entries can be safely overwritten
- Use modular indexing: cache_position = t % w
- The buffer has a fixed size of w, regardless of how long generation continues
**Figures**: fig_rolling_buffer_concept
**Transition**: "Let's walk through this step by step."

#### 6.3: Step-by-step buffer walkthrough (Head 2)
**Key Points**:
- Show 8 generation steps with w=3
- Steps 0, 1, 2: buffer fills positions 0, 1, 2
- Step 3: new token writes to position 3 % 3 = 0, overwriting the oldest entry
- Step 4: writes to position 4 % 3 = 1
- Step 5: writes to position 5 % 3 = 2
- The buffer always contains exactly the 3 most recent tokens
**Figures**: fig_rolling_buffer_walkthrough
**Transition**: "Let's quantify the memory savings."

#### 6.4: Memory savings quantified (Head 2)
**Key Points**:
- Full attention at 8K: stores 8,192 entries per layer
- SWA at 8K (w=4096): stores 4,096 entries per layer (50% savings)
- At 32K: stores 4,096 entries (87.5% savings)
- Formula: savings = 1 - (w/n)
- The longer the sequence, the greater the benefit
- Mistral 7B: fixed 512 MB cache regardless of sequence length
**Figures**: fig_kv_cache_memory_comparison
**Transition**: "Mistral 7B combines SWA with another technique for even greater savings."

#### 6.5: Combining with grouped query attention (Head 2)
**Key Points**:
- GQA: 32 query heads share 8 KV heads (4:1 ratio), reducing cache by 4x
- SWA rolling buffer reduces cache by another 2x
- Combined: 8x total reduction
- At 8K tokens: Mistral's cache is 8x smaller than LLaMA 2's equivalent
- Concrete numbers: 512 MB (Mistral) vs 4,096 MB (full attention, full heads, 16K)
**Figures**: fig_mistral_combined_savings
**Transition**: "There is one more inference optimization that SWA enables: pre-fill chunking."

---

### Section 7: Pre-fill chunking (Head 1)

Processing long prompts efficiently.

#### 7.1: The pre-fill memory spike (Head 2)
**Key Points**:
- When a user sends a long prompt, the model must process it all at once (the "pre-fill" phase)
- Standard approach: one forward pass over the entire prompt
- This requires materializing the full n x n attention matrix
- For an 8K prompt, that is a massive memory spike
**Figures**: fig_prefill_problem
**Transition**: "SWA provides a natural solution."

#### 7.2: Chunking the prompt (Head 2)
**Key Points**:
- Split the prompt into chunks of size w
- Process each chunk sequentially
- Each chunk only needs to compute attention within its local window
- The KV cache is filled incrementally, chunk by chunk
- Peak memory drops from O(n^2) to O(w^2)
- For 8K prompt with w=4096: two chunks instead of one massive pass
**Figures**: fig_prefill_chunks_sequential
**Transition**: "We have now covered the complete sliding window attention mechanism. Let's zoom out and see how it fits into the broader family of efficient attention."

---

### Section 8: The family of sparse attention (Head 1)

How SWA relates to other efficient attention mechanisms.

#### 8.1: The taxonomy of sparse attention (Head 2)
**Key Points**:
- Full attention: baseline, O(n^2)
- Sliding window: local band, O(n*w)
- Dilated sliding window: gaps for wider reach
- Global + sliding: designated hub tokens
- Local + random + global (BigBird): theoretical universality
- Show all five patterns side by side as attention matrices
**Figures**: fig_sparse_attention_taxonomy
**Transition**: "Let's examine each variant in more detail."

#### 8.2: Longformer's dilated sliding window (Head 2)
**Key Points**:
- Instead of attending to w consecutive neighbors, skip every d positions
- Dilation d=2 means attending to positions i, i-2, i-4, ... instead of i, i-1, i-2, ...
- Receptive field per layer becomes d * w instead of w
- Total receptive field: L * d * w
- Same compute cost as basic SWA, but wider reach
- Used in higher layers of Longformer, while lower layers use standard SWA
**Figures**: fig_dilated_sliding_window
**Transition**: "Longformer adds one more component: global attention."

#### 8.3: Global attention tokens (Head 2)
**Key Points**:
- Certain pre-selected tokens (e.g., [CLS], question tokens) get full attention
- These tokens attend to ALL positions and ALL positions attend to them
- They act as information hubs connecting distant parts of the sequence
- Global attention uses separate projection matrices: (Qg, Kg, Vg)
- Cost: O(g * n) where g is the number of global tokens (g << n)
**Figures**: fig_global_attention
**Transition**: "BigBird takes a different approach to long-range connections."

#### 8.4: BigBird: adding randomness (Head 2)
**Key Points**:
- BigBird combines three types: local window + global tokens + random connections
- Random attention: each token randomly attends to r tokens anywhere in the sequence
- Graph theory justification: random edges create O(log n) shortest paths
- This makes BigBird a universal approximator of full attention
- Block-level implementation for GPU efficiency (block_size=64)
- Typical: 192 local + 128 global + 192 random = 512 tokens attended per query (vs 4096 for full)
**Figures**: fig_bigbird_three_components
**Transition**: "Modern LLMs have taken a simpler approach: hybrid architectures that alternate between SWA and full attention."

---

### Section 9: Hybrid architectures in modern LLMs (Head 1)

How the latest models combine SWA with full attention.

#### 9.1: The hybrid design (Head 2)
**Key Points**:
- Instead of complex sparse patterns, alternate between SWA layers and full attention layers
- Full attention layers act as periodic "global checkpoints"
- The ratio of SWA-to-full determines the trade-off between efficiency and expressiveness
- Gemma 2: 1:1 ratio (alternating), w=4096
- Gemma 3: 5:1 ratio, w=1024 (more aggressive)
- Less than 0.3% perplexity loss with 5:1 ratio
**Figures**: fig_hybrid_layer_stack
**Transition**: "The landscape is evolving rapidly."

#### 9.2: The evolving landscape (Head 2)
**Key Points**:
- 2023: Mistral 7B, all SWA layers
- 2024: Gemma 2, 1:1 hybrid
- 2025: Gemma 3, 5:1 hybrid (more aggressive)
- 2025: Mistral Small 3.1, dropped SWA entirely (FlashAttention makes full attention fast enough)
- The optimal choice depends on model size, context length, and hardware
- No single approach dominates; the field is actively experimenting
**Figures**: fig_hybrid_evolution
**Transition**: "Having explored the theory and its variants, let's quantify the concrete gains."

---

### Section 10: Quantifying the gains (Head 1)

Concrete numbers comparing SWA to full attention.

#### 10.1: Compute savings (Head 2)
**Key Points**:
- Full attention at n=32K: approximately 1.07 billion scores
- SWA at n=32K, w=4096: approximately 134 million scores (8x reduction)
- The savings grow with sequence length: O(n^2) vs O(n*w)
- At n=8K: 2x reduction. At n=64K: 16x reduction.
- Show the scaling curves
**Figures**: fig_compute_scaling_curves
**Transition**: "Memory savings are equally dramatic."

#### 10.2: Memory savings (Head 2)
**Key Points**:
- Present the savings table: 50% at 8K, 75% at 16K, 87.5% at 32K, 93.75% at 64K
- Fixed 512 MB for Mistral 7B vs 4GB+ for equivalent full-attention model at 16K
- Combined with GQA: 8x total reduction
- These savings make long-context inference practical on consumer hardware
**Figures**: fig_memory_savings_table
**Transition**: "Having explored the theory, let's put our knowledge to the test with code."

---

### Section 11: Implementation in PyTorch (Head 1)

Working code with step-by-step annotations.

#### 11.1: Building the mask (Head 2)
**Key Points**:
- Naive implementation: double loop over (i, j) positions
- Efficient vectorized version using broadcasting
- Two conditions: (col <= row) for causal, (row - col < w) for window
- Convert boolean mask to additive mask (-inf for blocked positions)
**Figures**: fig_code_mask_visualization
**Code**: Mask construction function (both naive and efficient versions)

#### 11.2: The complete SWA forward pass (Head 2)
**Key Points**:
- Standard Q, K, V projections (unchanged from normal attention)
- Compute Q*K^T / sqrt(d_k)
- Add the SWA mask
- Apply softmax
- Multiply by V
- Show that SWA changes exactly ONE thing: the mask shape
**Code**: Complete forward pass function

#### 11.3: The rolling buffer cache (Head 2)
**Key Points**:
- RollingKVCache class with fixed-size buffers
- Modular indexing for circular writes
- get_keys/get_values methods that return valid entries
- Show how it integrates with the attention computation during generation
**Code**: RollingKVCache class

#### 11.4: Modern approach: FlexAttention (Head 2)
**Key Points**:
- PyTorch 2.5+ FlexAttention API
- Define a mask_mod function for sliding window
- create_block_mask handles efficient computation
- Fused kernels, no manual mask materialization
- 3 lines of code vs the manual implementation above
**Code**: FlexAttention sliding window example

**Transition**: "Let's wrap up everything we have learned."

---

### Section 12: Summary (Head 1)

**Key Takeaways** (bullet points, each 2-3 sentences):

- **The quadratic bottleneck**: Standard self-attention computes an n x n attention matrix, making it prohibitively expensive for long sequences. Sliding window attention solves this by restricting each token to attend only to its w nearest neighbors, reducing complexity from O(n^2) to O(n*w).

- **The sliding window mask**: The SWA mask is the intersection of two simple constraints: the causal mask (no future tokens) and the window distance mask (attend only within distance w). This produces a banded lower-triangular matrix that is trivial to implement.

- **The layer stacking trick**: Despite the limited per-layer window, stacking L layers creates a receptive field of L*w tokens. For Mistral 7B with 32 layers and w=4096, this theoretical receptive field reaches 131K tokens, far exceeding the context window.

- **The rolling buffer KV cache**: During autoregressive generation, SWA enables a fixed-size circular buffer that caps memory at w entries per layer, regardless of sequence length. This provides 50% savings at 8K tokens and 87.5% at 32K tokens.

- **The broader landscape**: SWA is the simplest member of a family of sparse attention mechanisms. Longformer adds dilated windows and global tokens; BigBird adds random connections. Modern LLMs like Gemma use hybrid approaches that alternate SWA and full attention layers.

- **A still-evolving field**: The optimal attention strategy depends on model size, context length, and hardware. Some models are moving to more aggressive SWA ratios (Gemma 3's 5:1), while others have dropped SWA entirely in favor of optimized full attention (Mistral Small 3.1). The field continues to experiment.

---

## Callout Boxes (2 total)

### Callout 1 (in Section 3)
> **What is an Attention Mask?** An attention mask is a matrix of the same shape as the attention scores that controls which positions each token can attend to. Positions with a mask value of 0 are allowed; positions with negative infinity are blocked. After the softmax operation, blocked positions receive exactly zero attention weight, as if they do not exist.

### Callout 2 (in Section 5)
> **Theoretical vs Effective Receptive Field**: The formula R_L = L * w gives the theoretical maximum receptive field, assuming each layer perfectly transmits all information from its entire window. In practice, the effective receptive field is smaller because attention weights are not uniform. Information degrades with each relay step, similar to the game of telephone. This is a key trade-off of SWA compared to full attention, where every token has a direct, one-hop path to every other token.

---

## Estimated Article Length
- **Text**: approximately 6,000-8,000 words
- **Diagrams**: 35 figures
- **Code blocks**: 4 (mask construction naive, mask construction efficient, rolling buffer class, FlexAttention example)
- **Tables**: 3-4 (comparison tables, memory savings)
