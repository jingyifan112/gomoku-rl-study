#include "common.h"

static int bench_in_bounds(int row, int col) {
    return row >= 0 && row < BOARD_SIZE && col >= 0 && col < BOARD_SIZE;
}

static int bench_pos(int row, int col) {
    return row * BOARD_SIZE + col;
}

static void clear_state(GameState *state) {
    init_game(state);
}

static int make_window(int *window) {
    int dirs[4][2] = {
        {0, 1},
        {1, 0},
        {1, 1},
        {1, -1}
    };

    for (int attempt = 0; attempt < 1000; attempt++) {
        int d = rand() % 4;
        int dr = dirs[d][0];
        int dc = dirs[d][1];

        int row = rand() % BOARD_SIZE;
        int col = rand() % BOARD_SIZE;

        int end_row = row + dr * 4;
        int end_col = col + dc * 4;

        if (!bench_in_bounds(end_row, end_col)) {
            continue;
        }

        for (int k = 0; k < 5; k++) {
            window[k] = bench_pos(row + dr * k, col + dc * k);
        }

        return 1;
    }

    return 0;
}

static int target_hit(int move, int *targets, int target_count) {
    for (int i = 0; i < target_count; i++) {
        if (move == targets[i]) {
            return 1;
        }
    }

    return 0;
}

static int make_case(GameState *state, int case_type, int *targets, int *target_count) {
    int window[5];

    clear_state(state);

    if (!make_window(window)) {
        return 0;
    }

    *target_count = 0;

    if (case_type == 0) {
        /* block_four: X X X X . */
        int empty_idx = rand() % 5;

        for (int i = 0; i < 5; i++) {
            if (i == empty_idx) {
                targets[(*target_count)++] = window[i];
            } else {
                state->board[window[i]] = 'X';
            }
        }
    } else if (case_type == 1) {
        /* win_four: O O O O . */
        int empty_idx = rand() % 5;

        for (int i = 0; i < 5; i++) {
            if (i == empty_idx) {
                targets[(*target_count)++] = window[i];
            } else {
                state->board[window[i]] = 'O';
            }
        }
    } else if (case_type == 2) {
        /* block_broken_four: X X . X X */
        state->board[window[0]] = 'X';
        state->board[window[1]] = 'X';
        targets[(*target_count)++] = window[2];
        state->board[window[3]] = 'X';
        state->board[window[4]] = 'X';
    } else if (case_type == 3) {
        /* block_open_three: . X X X . ; either end is acceptable */
        targets[(*target_count)++] = window[0];
        state->board[window[1]] = 'X';
        state->board[window[2]] = 'X';
        state->board[window[3]] = 'X';
        targets[(*target_count)++] = window[4];
    } else if (case_type == 4) {
        /* followup_block: O X X X . or . X X X O */
        if (rand() % 2 == 0) {
            state->board[window[0]] = 'O';
            state->board[window[1]] = 'X';
            state->board[window[2]] = 'X';
            state->board[window[3]] = 'X';
            targets[(*target_count)++] = window[4];
        } else {
            targets[(*target_count)++] = window[0];
            state->board[window[1]] = 'X';
            state->board[window[2]] = 'X';
            state->board[window[3]] = 'X';
            state->board[window[4]] = 'O';
        }
    } else {
        return 0;
    }

    return 1;
}

static void run_benchmark_case(NeuralNetwork *nn, const char *name, int case_type, int samples) {
    int correct = 0;
    int valid = 0;

    for (int i = 0; i < samples; i++) {
        GameState state;
        int targets[2];
        int target_count = 0;

        if (!make_case(&state, case_type, targets, &target_count)) {
            continue;
        }

        int move = get_computer_move(&state, nn, 0);

        if (target_hit(move, targets, target_count)) {
            correct++;
        }

        valid++;
    }

    if (valid > 0) {
        printf("%s accuracy: %d/%d = %.2f%%\n",
               name, correct, valid, 100.0f * correct / valid);
    }
}

int main(int argc, char **argv) {
    const char *model_file = MODEL_FILE;
    int samples = 1000;

    if (argc > 1) {
        model_file = argv[1];
    }

    if (argc > 2) {
        samples = atoi(argv[2]);
    }

    srand(1);

    NeuralNetwork nn;

    if (!load_neural_network(&nn, model_file)) {
        printf("Could not load model: %s\n", model_file);
        return 1;
    }

    printf("Running tactical benchmark on model: %s\n", model_file);
    printf("Samples per case: %d\n", samples);

    run_benchmark_case(&nn, "block_four", 0, samples);
    run_benchmark_case(&nn, "win_four", 1, samples);
    run_benchmark_case(&nn, "block_broken_four", 2, samples);
    run_benchmark_case(&nn, "block_open_three", 3, samples);
    run_benchmark_case(&nn, "followup_block", 4, samples);

    return 0;
}
