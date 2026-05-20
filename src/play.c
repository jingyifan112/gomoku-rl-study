#include "common.h"

typedef struct {
    char board_before[BOARD_CELLS + 1];
    char player;
    char source[16];
    int move;
} MoveRecord;

static void copy_board_string(GameState *state, char *dest) {
    for (int i = 0; i < BOARD_CELLS; i++) {
        dest[i] = state->board[i];
    }
    dest[BOARD_CELLS] = '\0';
}

static void append_game_log(MoveRecord *records, int num_records, char winner) {
    const char *log_path = "../run_logs/human_games.csv";

    int need_header = 0;
    FILE *check = fopen(log_path, "r");

    if (check == NULL) {
        need_header = 1;
    } else {
        fclose(check);
    }

    FILE *file = fopen(log_path, "a");

    if (file == NULL) {
        printf("Warning: could not open %s for writing.\n", log_path);
        return;
    }

    if (need_header) {
        fprintf(file, "game_id,move_index,player,source,move,winner,board_before\n");
    }

    long game_id = (long)time(NULL);

    for (int i = 0; i < num_records; i++) {
        fprintf(file, "%ld,%d,%c,%s,%d,%c,%s\n",
                game_id,
                i,
                records[i].player,
                records[i].source,
                records[i].move,
                winner,
                records[i].board_before);
    }

    fclose(file);

    printf("Saved game data to %s\n", log_path);
}

static int read_player_move(GameState *state) {
    int move;

    while (1) {
        printf("Your move (0-%d): ", BOARD_CELLS - 1);

        if (scanf("%d", &move) != 1) {
            int ch;
            while ((ch = getchar()) != '\n' && ch != EOF) {
            }

            printf("Invalid input. Please enter a number.\n");
            continue;
        }

        if (move < 0 || move >= BOARD_CELLS) {
            printf("Invalid move. Please enter a number from 0 to %d.\n", BOARD_CELLS - 1);
            continue;
        }

        if (state->board[move] != '.') {
            printf("That position is already occupied. Try again.\n");
            continue;
        }

        return move;
    }
}

static void play_game(NeuralNetwork *nn) {
    GameState state;
    char winner = 0;
    MoveRecord records[BOARD_CELLS];
    int num_records = 0;

    init_game(&state);

    printf("Welcome to 15x15 Gomoku! You are X, the computer is O.\n");
    printf("Enter positions as numbers from 0 to %d.\n", BOARD_CELLS - 1);
    printf("This game will be logged to run_logs/human_games.csv.\n");

    display_board(&state);

    while (!check_game_over(&state, &winner)) {
        int move;
        char symbol = (state.current_player == 0) ? 'X' : 'O';

        if (num_records < BOARD_CELLS) {
            copy_board_string(&state, records[num_records].board_before);
            records[num_records].player = symbol;
        }

        if (state.current_player == 0) {
            move = read_player_move(&state);

            if (num_records < BOARD_CELLS) {
                snprintf(records[num_records].source, sizeof(records[num_records].source), "human");
                records[num_records].move = move;
            }
        } else {
            printf("Computer's move:\n");
            move = get_computer_move(&state, nn, 1);

            if (move < 0) {
                winner = 'T';
                break;
            }

            printf("Computer placed O at position %d\n", move);

            if (num_records < BOARD_CELLS) {
                snprintf(records[num_records].source, sizeof(records[num_records].source), "computer");
                records[num_records].move = move;
            }
        }

        state.board[move] = symbol;

        if (num_records < BOARD_CELLS) {
            num_records++;
        }

        display_board(&state);

        state.current_player = !state.current_player;
    }

    if (winner == 'X') {
        printf("You win!\n");
    } else if (winner == 'O') {
        printf("Computer wins!\n");
    } else {
        printf("Tie game!\n");
    }

    append_game_log(records, num_records, winner);
}

int main(int argc, char **argv) {
    const char *model_file = MODEL_FILE;

    if (argc > 1) {
        model_file = argv[1];
    }

    NeuralNetwork nn;

    if (!load_neural_network(&nn, model_file)) {
        printf("Could not load model. Please train first:\n");
        printf("  ./train 5000\n");
        return 1;
    }

    printf("Ready to play! You are X, the computer is O.\n");

    char play_again = 'y';

    while (play_again == 'y' || play_again == 'Y') {
        play_game(&nn);

        printf("Play again? (y/n): ");
        scanf(" %c", &play_again);
    }

    return 0;
}
