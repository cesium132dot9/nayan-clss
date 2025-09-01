# Accessible Graphs: Llama-3 Forbidden Word Control Analysis

This folder contains 10 simple graphs explaining how the AI controls forbidden words.

## Graph Descriptions:

1. **Overall Effect Distribution**: Distribution of attention head effects on target logit suppression
2. **Layer Effects**: Box plot of attention head effects for each transformer layer
3. **Attention Head Heatmap**: Head effect magnitudes across all layers and positions, with a legend indicating the mapping of colors to values.
4. **Head Effect Balance**: Balance of positive and negative effect heads per layer
5. **Head Effect Flow**: Cumulative effects and layer transition analysis
6. **Head Effect Scatter**: Simple scatter plot of head effects by layer
7. **Effect Magnitudes**: Distribution of absolute head effect values
8. **Layer Progression**: Layer-wise trends in head effect patterns
9. **Circuit Map**: Sparse visualization of significant attention head effects
10. **Summary Statistics**: Aggregate metrics and distributions of head effects

## Technical Terms:
- **Head Effect**: Change in target token logit when ablating specific attention head
- **Δ Logit**: Difference in log probability between baseline and intervention
- **Layer Index**: Transformer layer position (0-31 for 32-layer model)
- **Head Index**: Attention head position within layer (0-31 for 32-head layers)
