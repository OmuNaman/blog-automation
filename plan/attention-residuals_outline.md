# Article Plan: Attention Residuals

## Title
Attention Residuals: Teaching transformers to choose which layers matter

## Subtitle
How Moonshot AI replaced a decade-old fixed-weight residual connection with learned depth-wise attention, unlocking 25% more compute efficiency in Kimi

## This Article Covers
- **The PreNorm dilution problem**: Why standard residual connections cause hidden states to grow uncontrollably with depth, drowning out individual layer contributions
- **The depth-time duality**: The elegant insight that information dilution across depth is structurally identical to memory loss across a sequence, and can be solved the same way
- **Attention Residuals (AttnRes)**: How replacing fixed accumulation with softmax attention over previous layer outputs gives each layer selective, input-dependent access to earlier representations
- **Block AttnRes**: The practical variant that partitions layers into blocks, reducing memory from O(Ld) to O(Nd) while preserving most gains
- **Quantifying the gains**: Scaling law experiments showing a 1.25x compute advantage, with benchmark improvements up to +7.5 points on GPQA-Diamond

## Running Example

Throughout this article, we use a consistent small-scale example:

- **Tokens**: "The", "cat", "sat", "down" (4 tokens)
- **Embedding dimension**: d = 8
- **Input matrix X**: shape (4, 8), four tokens each with an 8-dimensional embedding
- **Number of layers**: L = 6
- **Number of blocks** (for Block AttnRes): N = 2, with 3 layers per block
- **Pseudo-query vector w_l**: dimension d = 8, one per layer

We trace these tokens through standard residuals (showing magnitude growth) and AttnRes (showing selective, bounded aggregation) to make the math tangible.

---

## Diagram Master List

| #  | ID | Section | Type | Caption |
|----|-----|---------|------|---------|
| 1  | fig_roadmap | 1.0 Opening | roadmap | Figure 1. Roadmap of this article. We start with the problem of PreNorm dilution in standard residual connections, build the intuition behind depth-wise attention, walk through the AttnRes mechanism step by step, introduce the practical Block AttnRes variant, formalize the math, and quantify the gains. |
| 2  | fig_residual_stream_overview | 2.1 | architecture | Figure 2. The residual stream in a standard transformer. Each layer adds its output to the running sum with a fixed weight of 1.0, creating a single, ever-growing hidden state. |
| 3  | fig_prenorm_dilution_growth | 2.2 | chart | Figure 3. Hidden state magnitude growth across depth in a standard PreNorm transformer. The norm grows linearly with the number of layers, while each individual layer's fractional contribution shrinks toward zero. |
| 4  | fig_layer_contribution_pie | 2.3 | comparison | Figure 4. The contribution of each layer as a fraction of the total hidden state, shown for a 6-layer model. Layer 1's output, which might encode crucial low-level features, represents only 1/6 of the final hidden state. |
| 5  | fig_redundant_layers | 2.4 | comparison | Figure 5. The redundancy problem. Research has shown that entire layers can be removed from deep LLMs with minimal performance impact, suggesting that uniform accumulation wastes model capacity. |
| 6  | fig_sequence_attention_analogy | 3.1 | conceptual | Figure 6. The memory loss problem in sequences. Without attention, early tokens get forgotten as the sequence grows. The transformer solved this by allowing any position to selectively access any previous position. |
| 7  | fig_depth_time_duality | 3.2 | comparison | Figure 7. The depth-time duality. Left: standard self-attention operates horizontally across sequence positions, letting each token attend to previous tokens. Right: AttnRes operates vertically across network depth, letting each layer attend to previous layers. The mathematical structure is identical. |
| 8  | fig_90_degree_rotation | 3.3 | conceptual | Figure 8. Rotating attention 90 degrees. Standard attention selectively retrieves information across time (sequence length). AttnRes selectively retrieves information across depth (layer count). Both use softmax-weighted combinations to solve information dilution. |
| 9  | fig_standard_vs_attnres_side_by_side | 4.0 | architecture | Figure 9. Standard residual connections vs Attention Residuals. Left: each layer blindly adds its output to the accumulated sum. Right: each layer uses a learned query to selectively weight previous layer outputs via softmax attention. |
| 10 | fig_standard_residual_walkthrough | 4.1 | step-by-step | Figure 10. Standard residual computation for our running example. Four tokens flow through 6 layers, with each layer's output added with weight 1.0. The hidden state vectors grow progressively larger at each layer. |
| 11 | fig_standard_residual_magnitudes | 4.1 | chart | Figure 11. Hidden state magnitudes for our 4 tokens across 6 layers with standard residuals. All four tokens show monotonically increasing norms, confirming the O(L) growth problem in our running example. |
| 12 | fig_pseudo_query_concept | 4.2 | conceptual | Figure 12. The pseudo-query vector. Each layer l has a learned vector w_l that acts as a "question": which previous layers are most relevant for my computation? This query attends over all previous layer outputs to produce input-dependent weights. |
| 13 | fig_attnres_single_layer_computation | 4.3 | step-by-step | Figure 13. AttnRes computation at layer 4 in detail. The pseudo-query w_4 computes dot products with RMSNorm-normalized representations from layers 0 through 3, producing logits that pass through softmax to yield attention weights alpha. The output h_4 is the weighted sum of previous outputs. |
| 14 | fig_attnres_attention_weights_example | 4.3 | matrix-op | Figure 14. Attention weight computation for layer 4. The pseudo-query w_4 is dotted with each normalized previous-layer output. After softmax, the weights might be [0.05, 0.10, 0.60, 0.25], meaning layer 4 draws 60% of its input from layer 2's output and largely ignores the token embedding. |
| 15 | fig_attnres_full_walkthrough | 4.4 | step-by-step | Figure 15. Full AttnRes computation for our running example. The same 4 tokens flow through 6 layers, but now each layer selectively weights its inputs. The hidden state magnitudes remain bounded throughout, unlike the standard residual version. |
| 16 | fig_attnres_magnitudes_comparison | 4.4 | chart | Figure 16. Hidden state magnitudes comparison. Standard residuals (dashed, growing linearly) vs AttnRes (solid, bounded). AttnRes keeps magnitudes stable because each hidden state is a convex combination (weights sum to 1) of previous outputs. |
| 17 | fig_input_dependent_weights | 4.5 | comparison | Figure 17. Input-dependent attention weights. Different tokens produce different attention weight distributions over previous layers. Token "The" might attend strongly to the embedding layer, while token "down" might focus on intermediate layers. This is the key advantage over fixed-weight approaches. |
| 18 | fig_block_attnres_overview | 5.1 | architecture | Figure 18. Block AttnRes architecture. 48 layers partitioned into 8 blocks of 6 layers each. Within each block, standard residual connections accumulate outputs. At block boundaries, attention-based aggregation selectively combines block representations plus the original token embedding. |
| 19 | fig_block_vs_full_memory | 5.2 | comparison | Figure 19. Memory comparison. Full AttnRes stores all L layer outputs, requiring O(Ld) memory. Block AttnRes stores only N block-level representations, reducing memory to O(Nd). For a 48-layer model with 8 blocks, this is a 6x reduction. |
| 20 | fig_block_attnres_walkthrough | 5.3 | step-by-step | Figure 20. Block AttnRes step-by-step for our running example with L=6 layers and N=2 blocks. Block 1 (layers 1-3) uses standard residuals internally. At the block boundary, attention aggregates Block 1's output and the token embedding. Block 2 (layers 4-6) then proceeds with standard residuals, using the attention-aggregated input. |
| 21 | fig_block_boundary_attention | 5.3 | step-by-step | Figure 21. Attention computation at a block boundary. The current partial sum and all previous block representations (plus the token embedding) are stacked, normalized with RMSNorm, and scored using the learned projection. Softmax weights determine the aggregated output that feeds into the next block. |
| 22 | fig_pipeline_parallelism | 5.4 | system-arch | Figure 22. Block AttnRes in pipeline-parallel training. Each GPU stage holds one or more blocks. Block representations are cached and communicated between stages via point-to-point transfers, avoiding redundant recomputation. The two-phase strategy computes standard attention first, then applies cross-block attention using cached representations. |
| 23 | fig_convex_combination_bound | 6.1 | mathematical | Figure 23. The convex combination bound. Since softmax weights sum to 1, the AttnRes output is a weighted average of previous layer outputs. The magnitude of a weighted average can never exceed the largest input magnitude, bounding the hidden state regardless of depth. |
| 24 | fig_gradient_flow_standard | 6.2 | data-flow | Figure 24. Gradient flow in standard residuals. The gradient from the loss must propagate backward through a chain of (L-l) multiplicative terms. While the identity shortcut helps, gradients can still concentrate in shallow layers and attenuate in deeper ones. |
| 25 | fig_gradient_flow_attnres | 6.2 | data-flow | Figure 25. Gradient flow in AttnRes. Each layer receives a direct gradient signal weighted by its attention weight alpha. Layers that contribute more to the output get proportionally stronger gradient signals, creating a self-reinforcing learning dynamic and more uniform gradient distribution. |
| 26 | fig_gradient_norm_comparison | 6.2 | chart | Figure 26. Gradient norm distribution across layers. Standard residuals (left) show gradients concentrated in shallow layers. AttnRes (right) shows a more uniform distribution, enabling effective training of deep layers and better utilization of model capacity. |
| 27 | fig_linear_attention_equivalence | 6.3 | conceptual | Figure 27. Standard residuals as low-rank linear attention. The standard residual sum (all weights = 1) is equivalent to linear attention with constant attention weights over depth. AttnRes generalizes this to full-rank softmax attention, providing strictly more expressive depth-wise routing. |
| 28 | fig_scaling_law_curves | 7.1 | chart | Figure 28. Scaling law curves for standard residuals vs Block AttnRes. Validation loss plotted against training compute. AttnRes consistently achieves lower loss at every compute budget. The annotation shows that AttnRes matches the standard baseline trained with 1.25x more compute. |
| 29 | fig_benchmark_results_bar | 7.2 | chart | Figure 29. Benchmark results comparing baseline vs AttnRes on Kimi Linear (48B total, 3B activated, 1.4T tokens). The largest gains appear in multi-step reasoning (GPQA-Diamond +7.5) and mathematics (Math +3.6). |
| 30 | fig_depth_width_optimal | 7.3 | chart | Figure 30. Optimal depth-to-width ratio. Standard residuals favor shallower, wider architectures (left star). AttnRes shifts the optimum toward deeper, narrower architectures (right star), suggesting that effective depth-wise information routing fundamentally changes optimal model design. |
| 31 | fig_overhead_breakdown | 7.4 | chart | Figure 31. Overhead breakdown for Block AttnRes. Training cost increases by less than 4%, inference latency increases by less than 2%. The gains (1.25x compute advantage, benchmark improvements) far outweigh the overhead. |
| 32 | fig_evolution_timeline | 8.1 | timeline | Figure 32. The evolution of depth-wise aggregation in deep learning. From fixed residual connections (2016) through DenseNet, DenseFormer, ResFormer, and DeepCrossAttention, to Attention Residuals (2026). Each step adds more expressiveness to how layers combine information across depth. |
| 33 | fig_method_comparison_table | 8.2 | comparison | Figure 33. Comparison of depth-wise aggregation methods. Standard residuals use fixed weights, DenseFormer uses learned input-independent scalars, DeepCrossAttention uses input-dependent cross-attention, and AttnRes uses input-dependent softmax attention via pseudo-queries. Each approach trades off expressiveness against computational cost. |
| 34 | fig_attention_weight_heatmap | 9.1 | heatmap | Figure 34. Learned depth-wise attention weight heatmap from a trained model. Source layers on the x-axis, target layers on the y-axis. Notable patterns include: early layers attending to token embeddings, middle layers attending to nearby predecessors, and later layers developing long-range depth-wise connections. |
| 35 | fig_code_pseudocode | 9.2 | code | Figure 35. Annotated PyTorch pseudocode for Block AttnRes. The implementation stacks previous block representations, normalizes with RMSNorm, computes attention logits via einsum, applies softmax over the block dimension, and produces the weighted aggregation. |

**Total: 35 diagrams**

---

## Section Outline

### Section 1: Opening

**Title**: Attention Residuals: Teaching transformers to choose which layers matter

**Key Points:**
- Present the article scope: "This article covers" with 5 bullet points
- Bridge from prerequisite knowledge: assumes familiarity with transformers, self-attention, and residual connections
- Introduce the key question: what if residual connections could be smarter about how they combine layer outputs?
- Roadmap figure

**Figures:** fig_roadmap

**Transition:** "To understand why Attention Residuals matter, we first need to confront a hidden problem lurking inside every modern LLM."

---

### Section 2: The hidden flaw in residual connections (Head 1)

**Key Points:**
- Residual connections are the backbone of deep learning, yet they have a fundamental design limitation
- Every layer output is added with equal weight, creating an ever-growing hidden state
- This "PreNorm dilution" problem wastes model capacity and limits depth

**Figures:** fig_residual_stream_overview, fig_prenorm_dilution_growth, fig_layer_contribution_pie, fig_redundant_layers

**Transition:** "Now that we see the problem clearly, a natural question arises: can we fix this by applying the same trick that transformers already use to solve a structurally identical problem?"

#### 2.1: The residual stream (Head 2)
**Key Points:**
- Recap how residual connections work in transformers: h_l = h_{l-1} + f_l(RMSNorm(h_{l-1}))
- Each sublayer (attention, FFN) adds its contribution to a running sum
- This running sum is the "residual stream" that carries information across layers
- Visual: show the residual stream as a highway with on-ramps from each layer

**Figures:** fig_residual_stream_overview

#### 2.2: The magnitude growth problem (Head 2)
**Key Points:**
- Since every layer adds its output with weight 1.0, the hidden state norm grows as O(L)
- In a 50-layer model, the hidden state is the sum of ~50 vectors
- Use our running example: trace ||h_l|| for 4 tokens across 6 layers, showing the growth
- The RMSNorm before each sublayer normalizes the input, but the accumulated output keeps growing

**Figures:** fig_prenorm_dilution_growth

#### 2.3: The signal dilution effect (Head 2)
**Key Points:**
- Each layer's output is a constant-magnitude vector being added to a growing sum
- Layer l's contribution is roughly 1/L of the total hidden state
- Early layers that encode fundamental features (syntax, morphology) become "drowned out"
- Different sublayers receive the same blended signal, even though they may need different information

**Figures:** fig_layer_contribution_pie

#### 2.4: Evidence of wasted capacity (Head 2)
**Key Points:**
- Research shows entire layers can be removed from deep LLMs with minimal performance impact
- This suggests the uniform accumulation makes many layers effectively redundant
- The model cannot efficiently utilize depth because each additional layer has diminishing marginal impact

**Figures:** fig_redundant_layers

---

### Section 3: The depth-time duality (Head 1)

**Key Points:**
- The conceptual breakthrough that makes AttnRes possible
- Information dilution across depth is structurally identical to memory loss across a sequence
- The transformer already solved the sequence version with attention
- AttnRes applies the same solution to depth: "rotating attention 90 degrees"

**Figures:** fig_sequence_attention_analogy, fig_depth_time_duality, fig_90_degree_rotation

**Transition:** "With this duality in mind, let's see exactly how Attention Residuals work, step by step."

#### 3.1: The sequence analogy (Head 2)
**Key Points:**
- In early sequence models (RNNs), information from early tokens was progressively lost as the sequence grew
- The transformer's self-attention mechanism solved this: any position can directly access any previous position with learned, content-dependent weights
- This prevented the "forgetting" problem by enabling selective retrieval

**Figures:** fig_sequence_attention_analogy

#### 3.2: Rotating attention from time to depth (Head 2)
**Key Points:**
- In depth, the same "forgetting" happens: early layer contributions are progressively diluted
- The fix is structurally identical: let each layer directly access any previous layer with learned, content-dependent weights
- Standard self-attention: attend across sequence positions (horizontal)
- AttnRes: attend across layer depth (vertical)
- The mathematical structure is the same, just applied to a different dimension

**Figures:** fig_depth_time_duality, fig_90_degree_rotation

---

### Section 4: The mechanics of Attention Residuals: a hands-on walkthrough (Head 1)

**Key Points:**
- Detailed step-by-step walkthrough of how AttnRes replaces standard residuals
- Compare standard residual computation with AttnRes computation side by side
- Use the running example with concrete numbers
- Build from simple concepts (the pseudo-query) to the full mechanism

**Figures:** fig_standard_vs_attnres_side_by_side, fig_standard_residual_walkthrough, fig_standard_residual_magnitudes, fig_pseudo_query_concept, fig_attnres_single_layer_computation, fig_attnres_attention_weights_example, fig_attnres_full_walkthrough, fig_attnres_magnitudes_comparison, fig_input_dependent_weights

**Transition:** "We now have the full AttnRes mechanism, but there is a practical challenge: storing all L layer outputs requires O(Ld) memory. For massive models, we need a more efficient approach."

#### 4.1: Standard residuals in action (Head 2)
**Key Points:**
- Walk through our running example with standard residuals
- 4 tokens, 6 layers, d=8
- Show h_0 through h_6: each is the sum of all previous layer outputs
- Visualize the growing magnitudes at each layer
- Show the side-by-side comparison figure to frame what comes next

**Figures:** fig_standard_vs_attnres_side_by_side, fig_standard_residual_walkthrough, fig_standard_residual_magnitudes

#### 4.2: The pseudo-query vector (Head 2)
**Key Points:**
- Each layer l has a learned vector w_l ∈ R^d (the "pseudo-query")
- This vector encodes the question: "which previous layers matter most for my computation?"
- Unlike standard self-attention (which uses the input to compute Q), the pseudo-query is a fixed learned parameter
- However, the attention weights are still input-dependent because the "keys" come from previous layer outputs, which depend on the input

**Figures:** fig_pseudo_query_concept

#### 4.3: Computing attention weights over depth (Head 2)
**Key Points:**
- At layer l, we have l previous outputs: v_0 (embeddings), v_1, ..., v_{l-1}
- Each v_i is normalized with RMSNorm to produce a "key" k_i
- The pseudo-query w_l is dotted with each normalized key to produce logits
- Softmax over these logits produces attention weights α_{i→l}
- Walk through concrete numbers for layer 4: show the dot products, softmax, and resulting weights
- The formula: α_{i→l} = softmax(w_l^T · RMSNorm(k_i))

**Figures:** fig_attnres_single_layer_computation, fig_attnres_attention_weights_example

#### 4.4: The full AttnRes forward pass (Head 2)
**Key Points:**
- The new hidden state: h_l = Σ α_{i→l} · v_i (weighted sum of previous outputs)
- Walk through all 6 layers with AttnRes
- Show that magnitudes remain bounded (convex combination property)
- Compare side by side with the standard residual magnitudes

**Figures:** fig_attnres_full_walkthrough, fig_attnres_magnitudes_comparison

#### 4.5: Input-dependent layer selection (Head 2)
**Key Points:**
- Different tokens produce different keys at each layer (because v_i depends on the input)
- The same pseudo-query w_l produces different attention weights for different tokens
- Token "The" might attend strongly to the embedding; token "down" might focus on intermediate layers
- This is the key advantage over fixed-weight approaches like DenseFormer
- Note: "What is a pseudo-query? A pseudo-query is a learned parameter vector that functions like a query in standard attention, but instead of being derived from the input, it is a fixed vector that each layer learns during training."

**Figures:** fig_input_dependent_weights

---

### Section 5: Block AttnRes: scaling to real-world models (Head 1)

**Key Points:**
- Full AttnRes requires storing all layer outputs: O(Ld) memory
- Block AttnRes partitions layers into N blocks (~8 blocks optimal)
- Standard residuals within blocks, attention across blocks
- Memory reduced to O(Nd), recovering most gains
- Pipeline parallelism implementation with cache-based communication

**Figures:** fig_block_attnres_overview, fig_block_vs_full_memory, fig_block_attnres_walkthrough, fig_block_boundary_attention, fig_pipeline_parallelism

**Transition:** "Having built the complete mechanism, both the idealized full version and the practical block variant, let's formalize the mathematics and prove why AttnRes solves the dilution problem."

#### 5.1: The memory challenge (Head 2)
**Key Points:**
- Full AttnRes at layer l needs to store all l-1 previous outputs
- For a 48-layer model with d=8192, this is significant memory overhead
- In pipeline-parallel training, outputs from different layers live on different GPUs
- The solution: group layers into blocks and apply attention only at block boundaries

**Figures:** fig_block_attnres_overview, fig_block_vs_full_memory

#### 5.2: How Block AttnRes works (Head 2)
**Key Points:**
- Partition L layers into N blocks (8 blocks of ~6 layers in the Kimi 48B model)
- Within each block: standard residual connections (simple h_l = h_{l-1} + f_l)
- At block boundaries: compute attention over all previous block representations plus token embeddings
- The block output becomes one "value" in the attention for future blocks
- Walk through our running example: L=6, N=2, with block boundary at layer 3

**Figures:** fig_block_attnres_walkthrough, fig_block_boundary_attention

#### 5.3: Pipeline parallelism and system design (Head 2)
**Key Points:**
- In pipeline-parallel training, different GPU stages hold different blocks
- Challenge: the attention at block boundaries needs representations from blocks on other GPUs
- Solution: cache-based point-to-point communication of block representations between stages
- Two-phase computation: (1) standard attention within block, (2) cross-block attention using cached representations
- Online softmax during inference amortizes the cost

**Figures:** fig_pipeline_parallelism

---

### Section 6: The mathematics of Attention Residuals (Head 1)

**Key Points:**
- Formal proof that AttnRes bounds hidden state growth
- Gradient flow analysis showing more uniform gradient distribution
- The equivalence between standard residuals and low-rank linear attention over depth

**Figures:** fig_convex_combination_bound, fig_gradient_flow_standard, fig_gradient_flow_attnres, fig_gradient_norm_comparison, fig_linear_attention_equivalence

**Transition:** "The math confirms what our intuition suggested: AttnRes fundamentally solves the dilution problem. Now, let's see how much this matters in practice."

#### 6.1: Bounding the hidden state (Head 2)
**Key Points:**
- Key insight: softmax weights sum to 1, so AttnRes produces a convex combination
- For a convex combination: ||Σ α_i v_i|| ≤ max_i ||v_i|| (bounded by largest input)
- Standard residuals: ||Σ v_i|| can grow as O(L) (unbounded with depth)
- This single property eliminates the entire PreNorm dilution problem
- "This is the mathematical magic of Attention Residuals"

**Figures:** fig_convex_combination_bound

#### 6.2: Gradient flow and training dynamics (Head 2)
**Key Points:**
- Standard residuals: gradients flow through a chain of multiplicative terms, concentrating in shallow layers
- AttnRes: each layer receives a direct gradient signal weighted by α_{l→k}
- Layers that contribute more get proportionally stronger gradient signals
- This creates a self-reinforcing learning dynamic: useful layers get trained more
- Result: more uniform gradient norms across depth, better utilization of all layers

**Figures:** fig_gradient_flow_standard, fig_gradient_flow_attnres, fig_gradient_norm_comparison

#### 6.3: Standard residuals as low-rank linear attention (Head 2)
**Key Points:**
- Elegant theoretical result: standard residuals compute h_l = Σ 1 · v_i
- This is exactly linear attention with constant weights over depth
- AttnRes generalizes this to full-rank softmax attention over depth
- Standard residuals are a special case of AttnRes where all weights are equal
- This framing reveals that the "residual stream" has always been doing a form of attention over depth, just a very limited one

**Figures:** fig_linear_attention_equivalence

---

### Section 7: Quantifying the gains (Head 1)

**Key Points:**
- Scaling law experiments showing 1.25x compute advantage
- Benchmark results on Kimi Linear (48B/3B, 1.4T tokens)
- Depth vs width optimal ratio shift
- Overhead analysis: <4% training, <2% inference

**Figures:** fig_scaling_law_curves, fig_benchmark_results_bar, fig_depth_width_optimal, fig_overhead_breakdown

**Transition:** "These improvements did not emerge from a vacuum. Attention Residuals represent the latest step in a decade-long quest to improve how deep networks combine information across depth."

#### 7.1: Scaling law experiments (Head 2)
**Key Points:**
- Trained multiple model sizes with and without Block AttnRes
- AttnRes consistently achieves lower validation loss at every compute budget
- The key result: Block AttnRes matches baseline performance trained with 1.25x more compute
- This means a model with AttnRes is equivalent to a 25% larger standard model
- Plot showing the compute advantage clearly

**Figures:** fig_scaling_law_curves

#### 7.2: Benchmark results (Head 2)
**Key Points:**
- Kimi Linear: 48B total parameters, 3B activated (mixture-of-experts), 1.4T training tokens
- Largest improvements in multi-step reasoning: GPQA-Diamond +7.5, Math +3.6, HumanEval +3.1
- Consistent improvements across all 9 benchmarks tested
- Show the full results table with all numbers

**Figures:** fig_benchmark_results_bar

#### 7.3: Deeper models, narrower architectures (Head 2)
**Key Points:**
- Standard residuals favor wider, shallower architectures (because depth is poorly utilized)
- AttnRes enables narrower, deeper architectures (because depth is now effectively utilized)
- This suggests AttnRes does not just improve performance at fixed architecture, it changes the optimal architecture itself
- Fundamental shift in model design philosophy

**Figures:** fig_depth_width_optimal

#### 7.4: Overhead analysis (Head 2)
**Key Points:**
- Training cost increase: <4% (from additional attention computation at block boundaries)
- Inference latency increase: <2% (block attention is fast relative to standard attention)
- The 1.25x compute advantage far outweighs the <4% overhead
- 8 blocks is the optimal trade-off between performance and overhead (author-confirmed)

**Figures:** fig_overhead_breakdown

---

### Section 8: The evolution of depth-wise aggregation (Head 1)

**Key Points:**
- Place AttnRes in historical context
- Show the progression from fixed residuals to learned, input-dependent depth attention
- Compare all major approaches: standard, DenseFormer, ResFormer, DCA, AttnRes
- Critical perspective: when does AttnRes not work?

**Figures:** fig_evolution_timeline, fig_method_comparison_table

**Transition:** "Having explored the theory, results, and context, let's look inside a trained model to see what patterns the depth-wise attention actually learns."

#### 8.1: From ResNet to AttnRes: a ten-year journey (Head 2)
**Key Points:**
- 2016: He et al. introduce residual connections (fixed weight 1.0)
- 2016: DenseNet concatenates all previous outputs (vision domain)
- 2024: DenseFormer learns input-independent scalar weights per layer pair
- 2024: ResFormer adds residual connections on value vectors
- 2025: DeepCrossAttention uses full input-dependent cross-layer weights
- 2026: AttnRes applies softmax attention over depth with pseudo-queries
- Each step increases expressiveness of depth-wise information routing

**Figures:** fig_evolution_timeline

#### 8.2: Comparing the approaches (Head 2)
**Key Points:**
- Standard: fixed weights, zero overhead, O(L) growth
- DenseFormer: learned scalars, input-independent, minimal overhead
- DeepCrossAttention: input-dependent weights, full cross-attention, claims 3x speedup
- AttnRes: input-dependent softmax via pseudo-queries, <2% overhead, 1.25x compute advantage
- ResFormer: value-specific residuals, addresses attention concentration (different problem)
- Trade-off axis: expressiveness vs computational cost

**Figures:** fig_method_comparison_table

#### 8.3: When does AttnRes struggle? (Head 2)
**Key Points:**
- Ziming Liu's analysis: No Free Lunch theorem applies
- AttnRes excels on structured tasks (natural language) where skipping intermediate layers is valuable
- AttnRes can struggle on pure memorization tasks where uniform blending works fine
- Risk: if attention weights converge to uniform distribution, AttnRes degenerates to averaging all layers (representation collapse)
- Natural language's structured nature likely explains Kimi's strong empirical results

**Figures:** (none, text-based critical analysis)

---

### Section 9: Inside a trained model (Head 1)

**Key Points:**
- What patterns do the learned attention weights reveal?
- Attention weight heatmap showing layer-to-layer preferences
- Implementation walkthrough with annotated pseudocode

**Figures:** fig_attention_weight_heatmap, fig_code_pseudocode

#### 9.1: What the attention weights learn (Head 2)
**Key Points:**
- Visualize the attention weight heatmap from a trained Kimi model
- Notable patterns: early layers attend to token embeddings, middle layers to nearby predecessors, deep layers develop long-range connections
- Some layers consistently attend to the same source regardless of input (stable features)
- Other layers show highly input-dependent attention patterns (dynamic routing)

**Figures:** fig_attention_weight_heatmap

#### 9.2: Implementation walkthrough (Head 2)
**Key Points:**
- Walk through the official PyTorch pseudocode step by step
- The block_attn_res function: stack values, RMSNorm, einsum for logits, softmax, weighted aggregation
- Checkpointing at block boundaries
- How it integrates into a standard transformer training loop
- Minimal code changes required (drop-in replacement)

**Figures:** fig_code_pseudocode

---

### Section 10: Summary

**Key Points (each 2-3 sentences):**
- **The PreNorm dilution problem**: Standard residual connections add all layer outputs with fixed weight 1.0, causing hidden state magnitudes to grow as O(L) and progressively diluting each layer's contribution. This fundamental flaw wastes model capacity and limits the effective utilization of depth.
- **The depth-time duality**: Information dilution across network depth is structurally identical to memory loss across a sequence. Just as self-attention solved the sequence problem by enabling selective access to any previous position, AttnRes solves the depth problem by enabling selective access to any previous layer.
- **The AttnRes mechanism**: Each layer uses a learned pseudo-query vector to compute softmax attention weights over all previous layer outputs, producing an input-dependent weighted combination. Since softmax weights sum to 1, the resulting hidden state is a bounded convex combination, completely eliminating the O(L) growth problem.
- **Block AttnRes for scalability**: The practical variant partitions layers into approximately 8 blocks, using standard residuals within blocks and attention across block boundaries. This reduces memory from O(Ld) to O(Nd) while recovering most of the full AttnRes gains with less than 2% inference overhead.
- **Quantified improvements**: Block AttnRes achieves a 1.25x compute advantage, matching baseline models trained with 25% more resources. On Kimi Linear (48B/3B, 1.4T tokens), it improved GPQA-Diamond by +7.5, Math by +3.6, and HumanEval by +3.1, with the largest gains in multi-step reasoning tasks.

**Figures:** (none)

---

## Figure Description File Mapping

Each diagram will have a description file saved to `figures/descriptions/`:

| # | Description File | Output File |
|---|-----------------|-------------|
| 1 | fig_roadmap.txt | fig_roadmap.png |
| 2 | fig_residual_stream_overview.txt | fig_residual_stream_overview.png |
| 3 | fig_prenorm_dilution_growth.txt | fig_prenorm_dilution_growth.png |
| 4 | fig_layer_contribution_pie.txt | fig_layer_contribution_pie.png |
| 5 | fig_redundant_layers.txt | fig_redundant_layers.png |
| 6 | fig_sequence_attention_analogy.txt | fig_sequence_attention_analogy.png |
| 7 | fig_depth_time_duality.txt | fig_depth_time_duality.png |
| 8 | fig_90_degree_rotation.txt | fig_90_degree_rotation.png |
| 9 | fig_standard_vs_attnres_side_by_side.txt | fig_standard_vs_attnres_side_by_side.png |
| 10 | fig_standard_residual_walkthrough.txt | fig_standard_residual_walkthrough.png |
| 11 | fig_standard_residual_magnitudes.txt | fig_standard_residual_magnitudes.png |
| 12 | fig_pseudo_query_concept.txt | fig_pseudo_query_concept.png |
| 13 | fig_attnres_single_layer_computation.txt | fig_attnres_single_layer_computation.png |
| 14 | fig_attnres_attention_weights_example.txt | fig_attnres_attention_weights_example.png |
| 15 | fig_attnres_full_walkthrough.txt | fig_attnres_full_walkthrough.png |
| 16 | fig_attnres_magnitudes_comparison.txt | fig_attnres_magnitudes_comparison.png |
| 17 | fig_input_dependent_weights.txt | fig_input_dependent_weights.png |
| 18 | fig_block_attnres_overview.txt | fig_block_attnres_overview.png |
| 19 | fig_block_vs_full_memory.txt | fig_block_vs_full_memory.png |
| 20 | fig_block_attnres_walkthrough.txt | fig_block_attnres_walkthrough.png |
| 21 | fig_block_boundary_attention.txt | fig_block_boundary_attention.png |
| 22 | fig_pipeline_parallelism.txt | fig_pipeline_parallelism.png |
| 23 | fig_convex_combination_bound.txt | fig_convex_combination_bound.png |
| 24 | fig_gradient_flow_standard.txt | fig_gradient_flow_standard.png |
| 25 | fig_gradient_flow_attnres.txt | fig_gradient_flow_attnres.png |
| 26 | fig_gradient_norm_comparison.txt | fig_gradient_norm_comparison.png |
| 27 | fig_linear_attention_equivalence.txt | fig_linear_attention_equivalence.png |
| 28 | fig_scaling_law_curves.txt | fig_scaling_law_curves.png |
| 29 | fig_benchmark_results_bar.txt | fig_benchmark_results_bar.png |
| 30 | fig_depth_width_optimal.txt | fig_depth_width_optimal.png |
| 31 | fig_overhead_breakdown.txt | fig_overhead_breakdown.png |
| 32 | fig_evolution_timeline.txt | fig_evolution_timeline.png |
| 33 | fig_method_comparison_table.txt | fig_method_comparison_table.png |
| 34 | fig_attention_weight_heatmap.txt | fig_attention_weight_heatmap.png |
| 35 | fig_code_pseudocode.txt | fig_code_pseudocode.png |
