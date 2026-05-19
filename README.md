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
