import argparse
import csv
import random
from pathlib import Path

BOARD_SIZE = 15
BOARD_CELLS = BOARD_SIZE * BOARD_SIZE
CENTER = BOARD_SIZE // 2

DIRECTIONS = [
    (0, 1),
    (1, 0),
    (1, 1),
    (1, -1),
]

SCENARIOS = [
    "win_four",
    "block_four",
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


def center_distance(pos):
    r, c = pos_to_rc(pos)
    return abs(r - CENTER) + abs(c - CENTER)


def all_windows():
    windows = []

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            for dr, dc in DIRECTIONS:
                end_r = r + dr * 4
                end_c = c + dc * 4

                if not in_bounds(end_r, end_c):
                    continue

                windows.append([
                    rc_to_pos(r + dr * k, c + dc * k)
                    for k in range(5)
                ])

    return windows


WINDOWS = all_windows()


def has_winner(board, symbol):
    for window in WINDOWS:
        if all(board[p] == symbol for p in window):
            return True
    return False


def best_tactical_move_for_o(board):
    candidates = []

    for window in WINDOWS:
        values = [board[p] for p in window]

        x_count = values.count("X")
        o_count = values.count("O")
        empty_positions = [p for p in window if board[p] == "."]

        # Highest priority: O can win immediately.
        if o_count == 4 and x_count == 0 and len(empty_positions) == 1:
            candidates.append((100, empty_positions[0]))

        # Then block X immediate win.
        elif x_count == 4 and o_count == 0 and len(empty_positions) == 1:
            candidates.append((90, empty_positions[0]))

        # Block broken four: X X . X X
        elif values == ["X", "X", ".", "X", "X"]:
            candidates.append((85, window[2]))

        # Follow-up block: O X X X . or . X X X O
        elif values == ["O", "X", "X", "X", "."]:
            candidates.append((80, window[4]))
        elif values == [".", "X", "X", "X", "O"]:
            candidates.append((80, window[0]))

        # Extend O open three.
        elif values == [".", "O", "O", "O", "."]:
            candidates.append((70, window[0]))
            candidates.append((70, window[4]))

        # Block X open three.
        elif values == [".", "X", "X", "X", "."]:
            candidates.append((60, window[0]))
            candidates.append((60, window[4]))

    if not candidates:
        return None

    max_priority = max(priority for priority, _ in candidates)
    top = [pos for priority, pos in candidates if priority == max_priority]

    # Single-label tie break: choose closer to center.
    top = sorted(set(top), key=lambda p: (center_distance(p), p))
    return top[0]


def random_window():
    return random.choice(WINDOWS)


def add_fillers(board, forbidden, max_fillers):
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


def make_pattern_board():
    scenario = random.choice(SCENARIOS)
    board = ["."] * BOARD_CELLS
    window = random_window()
    forbidden = set(window)

    if scenario == "win_four":
        empty_idx = random.randrange(5)
        for i, pos in enumerate(window):
            if i != empty_idx:
                board[pos] = "O"

    elif scenario == "block_four":
        empty_idx = random.randrange(5)
        for i, pos in enumerate(window):
            if i != empty_idx:
                board[pos] = "X"

    elif scenario == "block_broken_four":
        pattern = ["X", "X", ".", "X", "X"]
        for value, pos in zip(pattern, window):
            board[pos] = value

    elif scenario == "block_open_three":
        pattern = [".", "X", "X", "X", "."]
        for value, pos in zip(pattern, window):
            board[pos] = value

    elif scenario == "followup_block":
        if random.choice([True, False]):
            pattern = ["O", "X", "X", "X", "."]
        else:
            pattern = [".", "X", "X", "X", "O"]

        for value, pos in zip(pattern, window):
            board[pos] = value

    elif scenario == "extend_open_three":
        pattern = [".", "O", "O", "O", "."]
        for value, pos in zip(pattern, window):
            board[pos] = value

    else:
        raise ValueError(scenario)

    # Add some background stones, but not too many.
    add_fillers(board, forbidden, max_fillers=10)

    return board, scenario


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    random.seed(args.seed)

    rows = []
    seen = set()
    attempts = 0

    while len(rows) < args.samples and attempts < args.samples * 100:
        attempts += 1

        board, scenario = make_pattern_board()

        # Keep boards clean: no winner already before O moves.
        if has_winner(board, "X") or has_winner(board, "O"):
            continue

        target = best_tactical_move_for_o(board)

        if target is None:
            continue

        if board[target] != ".":
            continue

        board_str = "".join(board)
        key = (board_str, target)

        if key in seen:
            continue

        seen.add(key)

        rows.append({
            "game_id": f"tactical_{scenario}_{len(rows)}",
            "move_index": "0",
            "player": "O",
            "source": "human",
            "move": str(target),
            "winner": "O",
            "board_before": board_str,
        })

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
    print(f"Attempts: {attempts}")


if __name__ == "__main__":
    main()
