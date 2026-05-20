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

After getting the basic 15x15 Gomoku version running, I also tried a small improvement to increase the win rate against the random opponent.

The main updates were:

- Increased `NN_HIDDEN_SIZE` from 100 to 256.
- Increased `MAX_REPLAY_MOVES` from 40 to 80.
- Added a simple tactical move check before using the neural network output:
  - if `O` can win immediately, choose that move;
  - otherwise, if `X` can win immediately, block that move;
  - otherwise, use the neural network's best legal move.

With this version, I ran:

```bash
./train 1000

The training log showed:￼Total games counted: 1000
Wins: 1000 (100.0%)
Losses: 0 (0.0%)
Ties: 0 (0.0%)This result is against a random player, so I treat it as a sanity check for the modified training/play pipeline rather than a full evaluation of Gomoku strength.￼

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
