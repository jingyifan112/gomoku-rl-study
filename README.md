# gomoku-rl-study

This repo keeps the original common.h / train.c / play.c / Makefile structure from Arthur Chiao's tic-tac-toe reinforcement learning example, and modifies it into a 15x15 Gomoku experiment.

## Main changes

Compared with the original tic-tac-toe version:

- Board size: 3x3 -> 15x15
- Board positions: 9 -> 225
- Input size: 18 -> 450
- Output size: 9 -> 225
- Win condition: 3 in a row -> 5 in a row
- Human input: 0-8 -> 0-224
- Game-over checking: fixed tic-tac-toe lines -> directional five-in-a-row checking

## Training improvement

After removing the rule-based tactical check, I kept move selection based only on the neural network output.

To improve the pure neural network version, I adjusted the training setup:

- Increased `NN_HIDDEN_SIZE` from 100 to 256.
- Increased `MAX_REPLAY_MOVES` from 40 to 80.
- Increased training from 1000 games to 5000 games.

With `./train 5000`, the model reached a 99.8% win rate against the random player in the training log.

This result is still against a random opponent, so I treat it as a training sanity check rather than a full evaluation of Gomoku strength.
After getting the basic 15x15 Gomoku version running, I also tried a small improvement to increase the win rate against the random opponent.

## Files

- `src/common.h`: board representation, neural network structure, forward pass, move selection, win checking, save/load functions
- `src/train.c`: training loop against a random opponent
- `src/play.c`: interactive human-vs-computer play mode
- `src/Makefile`: build commands

## How to run

Compile:

```bash
cd src
make
