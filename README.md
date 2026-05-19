# gomoku-rl-study

This repo is an experimental extension of a tic-tac-toe reinforcement learning project into a 15x15 Gomoku-style game.

The goal is to modify the original small reinforcement learning pipeline so it can run on a larger board with a five-in-a-row win condition.

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
