#include "common.h"

static int load_board(GameState *state, const char *board_str) {
    init_game(state);

    for (int i = 0; i < BOARD_CELLS; i++) {
        char ch = board_str[i];

        if (ch == '\0') {
            return 0;
        }

        if (ch == '.' || ch == 'X' || ch == 'O') {
            state->board[i] = ch;
        } else {
            return 0;
        }
    }

    return 1;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Usage:\n");
        printf("  %s <model.bin> <test.csv>\n", argv[0]);
        return 1;
    }

    const char *model_file = argv[1];
    const char *csv_file = argv[2];

    NeuralNetwork nn;

    if (!load_neural_network(&nn, model_file)) {
        printf("Could not load model: %s\n", model_file);
        return 1;
    }

    FILE *file = fopen(csv_file, "r");

    if (file == NULL) {
        printf("Could not open CSV: %s\n", csv_file);
        return 1;
    }

    char line[2048];

    /* Skip header. */
    fgets(line, sizeof(line), file);

    int total = 0;
    int correct = 0;

    while (fgets(line, sizeof(line), file)) {
        char *game_id = strtok(line, ",");
        char *move_index_str = strtok(NULL, ",");
        char *player_str = strtok(NULL, ",");
        char *source = strtok(NULL, ",");
        char *move_str = strtok(NULL, ",");
        char *winner_str = strtok(NULL, ",");
        char *board_str = strtok(NULL, ",\n");

        if (game_id == NULL || move_index_str == NULL || player_str == NULL ||
            source == NULL || move_str == NULL || winner_str == NULL || board_str == NULL) {
            continue;
        }

        char player = player_str[0];

        /* Current play mode only uses the computer O. */
        if (player != 'O') {
            continue;
        }

        int label_move = atoi(move_str);

        GameState state;

        if (!load_board(&state, board_str)) {
            continue;
        }

        if (label_move < 0 || label_move >= BOARD_CELLS || state.board[label_move] != '.') {
            continue;
        }

        int model_move = get_computer_move(&state, &nn, 0);

        if (model_move == label_move) {
            correct++;
        }

        total++;
    }

    fclose(file);

    if (total == 0) {
        printf("No valid examples found.\n");
        return 1;
    }

    printf("CSV tactical benchmark: %d/%d = %.2f%%\n",
           correct, total, 100.0f * correct / total);

    return 0;
}
