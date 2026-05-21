# Final Training Observation

I tested several network-based improvements for the Gomoku model.

Experiments included:

- perspective-based / improved board encoding attempts
- mixed training and self-play
- synthetic tactical pretraining
- human game logging
- human-augmented fine-tuning
- single-label threat examples
- deeper MLP experiment
- clean supervised tactical data pipeline
- automatic tactical benchmark
- failure-driven fine-tuning from manual play failures

The automatic benchmark improved after the supervised tactical pipeline. For example, the pipeline model improved open-three blocking and follow-up blocking accuracy compared with the previous baseline.

However, the improvement did not reliably transfer to manual human play. In interactive games, the model still often failed to block obvious human threats, including direct horizontal or vertical five-in-a-row patterns.

Current conclusion:

The MLP-based model can learn some tactical correlations, but it still does not reliably understand the 15x15 board's spatial structure in real play. Further improvement likely requires a stronger model design, such as a CNN-based policy, or a more advanced training setup.
