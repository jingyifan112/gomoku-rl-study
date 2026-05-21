import argparse
import csv
import random
from pathlib import Path

BOARD_SIZE = 15
BOARD_CELLS = BOARD_SIZE * BOARD_SIZE

DIRECTIONS = [
    (0, 1),
    (1, 0),
    (1, 1),
    (1, -1),
]

SCENARIOS = [
    "block_four",
    "win_four",
    "block_broken_four",
    "block_open_three",
    "followup_block",
    "extend_open_three",
]


def in_bounds(r, c):
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE


def rc_to_pos(r, c):
    return r * BOARD_SIZE + c


def pos_to_rc(pos):
    return divmod(pos, BOARD_SIZE)


def transform_pos(pos, t):
    r, c = pos_to_rc(pos)
    n = BOARD_SIZE

    if t == 0:
        rr, cc = r, c
    elif t == 1:
        rr, cc = c, n - 1 - r
    elif t == 2:
        rr, cc = n - 1 - r, n - 1 - c
    elif t == 3:
        rr, cc = n - 1 - c, r
    elif t == 4:
        rr, cc = r, n - 1 - c
    elif t == 5:
        rr, cc = n - 1 - r, c
    elif t == 6:
        rr, cc = c, r
    elif t == 7:
        rr, cc = n - 1 - c, n - 1 - r
    else:
        raise ValueError(t)

    return rc_to_pos(rr, cc)


def transform_board(board, target, t):
    new_board = ["."] * BOARD_CELLS

    for pos, value in enumerate(board):
        new_board[transform_pos(pos, t)] = value

    return "".join(new_board), transform_pos(target, t)


def random_window():
    while True:
        dr, dc = random.choice(DIRECTIONS)
        r = random.randrange(BOARD_SIZE)
        c = random.randrange(BOARD_SIZE)

        end_r = r + dr * 4
        end_c = c + dc * 4

        if in_bounds(end_r, end_c):
            return [rc_to_pos(r + dr * k, c + dc * k) for k in range(5)]


def add_fillers(board, forbidden, max_fillers=12):
    fillers = random.randint(0, max_fillers)

    for _ in range(fillers):
        candidates = [
            i for i, v in enumerate(board)
            if v == "." and i not in forbidden
        ]

        if not candidates:
            return

        pos = random.choice(candidates)
        board[pos] = random.choice(["X", "O"])


def make_example():
    scenario = random.choice(SCENARIOS)
    board = ["."] * BOARD_CELLS
    window = random_window()
    forbidden = set(window)

    if scenario == "block_four":
        # X X X X .  -> O should block the empty spot
        target_idx = random.randrange(5)
        target = window[target_idx]

        for i, pos in enumerate(window):
            if i != target_idx:
                board[pos] = "X"

    elif scenario == "win_four":
        # O O O O . -> O should win
        target_idx = random.randrange(5)
        target = window[target_idx]

        for i, pos in enumerate(window):
            if i != target_idx:
                board[pos] = "O"

    elif scenario == "block_broken_four":
        # X X . X X -> O should block the middle gap
        pattern = ["X", "X", ".", "X", "X"]
        target = window[2]

        for value, pos in zip(pattern, window):
            board[pos] = value

    elif scenario == "block_open_three":
        # . X X X . -> choose one clear block point
        pattern = [".", "X", "X", "X", "."]
        target_idx = random.choice([0, 4])
        target = window[target_idx]

        for value, pos in zip(pattern, window):
            board[pos] = value

    elif scenario == "followup_block":
        # O X X X . or . X X X O -> O should block remaining end
        if random.choice([True, False]):
            pattern = ["O", "X", "X", "X", "."]
            target_idx = 4
        else:
            pattern = [".", "X", "X", "X", "O"]
            target_idx = 0

        target = window[target_idx]

        for value, pos in zip(pattern, window):
            board[pos] = value

    elif scenario == "extend_open_three":
        # . O O O . -> O should extend
        pattern = [".", "O", "O", "O", "."]
        target_idx = random.choice([0, 4])
        target = window[target_idx]

        for value, pos in zip(pattern, window):
            board[pos] = value

    else:
        raise ValueError(scenario)

    add_fillers(board, forbidden | {target})

    return board, target, scenario


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="run_logs/tactical_supervised_clean.csv")
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    random.seed(args.seed)

    rows = []
    example_id = 0

    while len(rows) < args.samples:
        board, target, scenario = make_example()

        # Use 8 symmetry transforms for better coverage.
        transforms = list(range(8))
        random.shuffle(transforms)

        for t in transforms:
            if len(rows) >= args.samples:
                break

            board_str, transformed_target = transform_board(board, target, t)

            rows.append({
                "game_id": f"tactical_{example_id}_{scenario}",
                "move_index": "0",
                "player": "O",
                "source": "human",
                "move": str(transformed_target),
                "winner": "O",
                "board_before": board_str,
            })

        example_id += 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as f:
        fieldnames = [
            "game_id",
            "move_index",
            "player",
            "source",
            "move",
            "winner",
            "board_before",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} examples to {output_path}")


if __name__ == "__main__":
    main()
