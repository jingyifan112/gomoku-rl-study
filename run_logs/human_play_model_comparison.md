
## Single-label human fine-tuning result

I tested the single-label human threat fine-tuning model. It still did not reliably block clear human threats. In one test, the human player formed a vertical five-in-a-row, and the model still failed to block in time.

This suggests that the current one-hidden-layer MLP may not be sufficient for learning stable Gomoku defensive patterns, even with augmented human threat examples.

Next step: try a deeper MLP structure while keeping the same neural-network-only move selection.

## Clean tactical fine-tuning result

I tested clean tactical fine-tuning from the current best model.

Benchmark results improved for some tactical cases:

- block_open_three improved
- followup_block improved
- block_broken_four remained high

However, manual play did not show a reliable improvement. In manual tests, the model still failed to block obvious human threats, including vertical and horizontal line-building patterns.

The e30 model had better benchmark scores than e10, but it still failed to consistently block in interactive play. Therefore, I did not select it as the default model.

Current default model remains:

- `gomoku_stronger_humanplay.bin`
