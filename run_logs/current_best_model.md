# Current Best Model

Current selected model:

- `gomoku_stronger_humanplay.bin`

Training setup:

- one-hidden-layer MLP
- perspective-based input encoding
- mixed training
- synthetic tactical examples
- no rule-based tactical check during play

Manual observation:

This model is more interactive than the original random-training version. It sometimes responds near human threats, but it still does not reliably block continuous human line-building threats.

I selected this model as the current best baseline because later experiments such as heavier tactical fine-tuning, h512, and deeper MLP did not clearly improve manual play.
