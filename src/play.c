#include "common.h"

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

    init_game(&state);

    printf("Welcome to 15x15 Gomoku! You are X, the computer is O.\n");
    printf("Enter positions as numbers from 0 to %d.\n", BOARD_CELLS - 1);

    display_board(&state);

    while (!check_game_over(&state, &winner)) {
        int move;

        if (state.current_player == 0) {
            move = read_player_move(&state);
        } else {
            printf("Computer's move:\n");
            move = get_computer_move(&state, nn, 1);

            if (move < 0) {
                winner = 'T';
                break;
            }

            printf("Computer placed O at position %d\n", move);
        }

        char symbol = (state.current_player == 0) ? 'X' : 'O';

        state.board[move] = symbol;

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
}

int main(int argc, char **argv) {
    const char *model_file = MODEL_FILE;

    if (argc > 1) {
        model_file = argv[1];
    }

    NeuralNetwork nn;

    if (!load_neural_network(&nn, model_file)) {
        printf("Could not load model. Please train first:\n");
        printf("  ./train 200\n");
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
