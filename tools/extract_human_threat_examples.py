import csv
from pathlib import Path
import argparse

BOARD_SIZE = 15
BOARD_CELLS = BOARD_SIZE * BOARD_SIZE
CENTER = BOARD_SIZE // 2

DIRECTIONS = [
    (0, 1),
    (1, 0),
    (1, 1),
    (1, -1),
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

                window = [
                    rc_to_pos(r + dr * k, c + dc * k)
                    for k in range(5)
                ]
                windows.append(window)

    return windows


WINDOWS = all_windows()


def find_best_block_target(board):
    candidates = []

    for window in WINDOWS:
        values = [board[p] for p in window]

        x_count = values.count("X")
        o_count = values.count("O")
        empty_positions = [p for p in window if board[p] == "."]

        # Priority 4: X has four stones in a five-cell window.
        if x_count == 4 and o_count == 0 and len(empty_positions) == 1:
            candidates.append((4, empty_positions[0]))

        # Priority 3: follow-up block after one side is already blocked.
        elif x_count == 3 and o_count == 1 and len(empty_positions) == 1:
            candidates.append((3, empty_positions[0]))

        # Priority 2: open three.
        elif x_count == 3 and o_count == 0 and len(empty_positions) == 2:
            for pos in empty_positions:
                candidates.append((2, pos))

    if not candidates:
        return None

    max_priority = max(priority for priority, _ in candidates)
    top = [pos for priority, pos in candidates if priority == max_priority]

    # Avoid contradictory labels by choosing only one target.
    # Tie-break: choose the candidate closer to the center.
    top = sorted(set(top), key=lambda p: (center_distance(p), p))

    return top[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="run_logs/human_games.csv")
    parser.add_argument("--output", default="run_logs/human_games_augmented_single.csv")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    with input_path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    output_rows = list(rows)
    added = 0

    for row in rows:
        # Only inspect computer turns. This is the board before O moves.
        if row.get("player") != "O":
            continue

        board = row.get("board_before", "")

        if len(board) != BOARD_CELLS:
            continue

        target = find_best_block_target(board)

        if target is None:
            continue

        added += 1

        output_rows.append({
            "game_id": f"threat_{added}",
            "move_index": "0",
            "player": "O",
            "source": "human",
            "move": str(target),
            "winner": "O",
            "board_before": board,
        })

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
        writer.writerows(output_rows)

    print(f"Original rows: {len(rows)}")
    print(f"Added single-label blocking examples: {added}")
    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    main()
