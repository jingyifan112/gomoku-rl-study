#include "common.h"

#define LEARNING_RATE 0.01f
#define MAX_REPLAY_MOVES 40
#define RANDOM_WEIGHT() ((((float)rand() / RAND_MAX) - 0.5f) * 0.1f)

static void init_neural_network(NeuralNetwork *nn) {
    for (int i = 0; i < NN_INPUT_SIZE * NN_HIDDEN_SIZE; i++) {
        nn->weights_ih[i] = RANDOM_WEIGHT();
    }

    for (int i = 0; i < NN_HIDDEN_SIZE * NN_OUTPUT_SIZE; i++) {
        nn->weights_ho[i] = RANDOM_WEIGHT();
    }

    for (int i = 0; i < NN_HIDDEN_SIZE; i++) {
        nn->biases_h[i] = RANDOM_WEIGHT();
    }

    for (int i = 0; i < NN_OUTPUT_SIZE; i++) {
        nn->biases_o[i] = RANDOM_WEIGHT();
    }
}

static void backprop(NeuralNetwork *nn, float *target_probs, float learning_rate, float reward_scaling) {
    float output_deltas[NN_OUTPUT_SIZE];
    float hidden_deltas[NN_HIDDEN_SIZE];

    for (int i = 0; i < NN_OUTPUT_SIZE; i++) {
        output_deltas[i] = (nn->outputs[i] - target_probs[i]) * fabsf(reward_scaling);
    }

    for (int i = 0; i < NN_HIDDEN_SIZE; i++) {
        float error = 0.0f;

        for (int j = 0; j < NN_OUTPUT_SIZE; j++) {
            error += output_deltas[j] * nn->weights_ho[i * NN_OUTPUT_SIZE + j];
        }

        hidden_deltas[i] = error * relu_derivative(nn->hidden[i]);
    }

    for (int i = 0; i < NN_HIDDEN_SIZE; i++) {
        for (int j = 0; j < NN_OUTPUT_SIZE; j++) {
            nn->weights_ho[i * NN_OUTPUT_SIZE + j] -= learning_rate * output_deltas[j] * nn->hidden[i];
        }
    }

    for (int j = 0; j < NN_OUTPUT_SIZE; j++) {
        nn->biases_o[j] -= learning_rate * output_deltas[j];
    }

    for (int i = 0; i < NN_INPUT_SIZE; i++) {
        for (int j = 0; j < NN_HIDDEN_SIZE; j++) {
            nn->weights_ih[i * NN_HIDDEN_SIZE + j] -= learning_rate * hidden_deltas[j] * nn->inputs[i];
        }
    }

    for (int j = 0; j < NN_HIDDEN_SIZE; j++) {
        nn->biases_h[j] -= learning_rate * hidden_deltas[j];
    }
}

static void learn_from_game(NeuralNetwork *nn, int *move_history, int num_moves, int nn_moves_even, char winner) {
    float reward;
    char nn_symbol = nn_moves_even ? 'O' : 'X';

    if (winner == 'T') {
        reward = 0.2f;
    } else if (winner == nn_symbol) {
        reward = 1.0f;
    } else {
        reward = -1.5f;
    }

    GameState state;
    float target_probs[NN_OUTPUT_SIZE];

    for (int move_idx = 0; move_idx < num_moves; move_idx++) {
        if ((nn_moves_even && move_idx % 2 != 1) || (!nn_moves_even && move_idx % 2 != 0)) {
            continue;
        }

        if (num_moves - move_idx > MAX_REPLAY_MOVES) {
            continue;
        }

        init_game(&state);

        for (int i = 0; i < move_idx; i++) {
            char symbol = (i % 2 == 0) ? 'X' : 'O';
            state.board[move_history[i]] = symbol;
        }

        float inputs[NN_INPUT_SIZE];
        board_to_inputs(&state, inputs);
        forward_pass(nn, inputs);

        for (int i = 0; i < NN_OUTPUT_SIZE; i++) {
            target_probs[i] = 0.0f;
        }

        int move = move_history[move_idx];

        float move_importance = 0.5f + 0.5f * (float)move_idx / (float)num_moves;
        float scaled_reward = reward * move_importance;

        if (scaled_reward >= 0) {
            target_probs[move] = 1.0f;
        } else {
            int other_valid_moves = 0;

            for (int i = 0; i < BOARD_CELLS; i++) {
                if (state.board[i] == '.' && i != move) {
                    other_valid_moves++;
                }
            }

            if (other_valid_moves == 0) {
                continue;
            }

            float other_prob = 1.0f / other_valid_moves;

            for (int i = 0; i < BOARD_CELLS; i++) {
                if (state.board[i] == '.' && i != move) {
                    target_probs[i] = other_prob;
                }
            }
        }

        backprop(nn, target_probs, LEARNING_RATE, scaled_reward);
    }
}

static char play_random_game(NeuralNetwork *nn, int *move_history) {
    GameState state;
    char winner = 0;
    int num_moves = 0;

    init_game(&state);

    while (!check_game_over(&state, &winner) && num_moves < BOARD_CELLS) {
        int move;

        if (state.current_player == 0) {
            move = get_random_move(&state);
        } else {
            move = get_computer_move(&state, nn, 0);
        }

        if (move < 0) {
            winner = 'T';
            break;
        }

        char symbol = (state.current_player == 0) ? 'X' : 'O';
        state.board[move] = symbol;
        move_history[num_moves++] = move;

        state.current_player = !state.current_player;
    }

    if (winner == 0) {
        winner = 'T';
    }

    learn_from_game(nn, move_history, num_moves, 1, winner);

    return winner;
}

static void train_against_random(NeuralNetwork *nn, int num_games) {
    int move_history[BOARD_CELLS];
    int wins = 0;
    int losses = 0;
    int ties = 0;

    int interval = 100;
    if (num_games >= 5000) {
        interval = 500;
    }
    if (num_games >= 20000) {
        interval = 1000;
    }

    printf("Training Gomoku neural network with %d games\n", num_games);
    printf("Board: %dx%d, win length: %d\n", BOARD_SIZE, BOARD_SIZE, WIN_LENGTH);

    for (int i = 0; i < num_games; i++) {
        char winner = play_random_game(nn, move_history);

        if (winner == 'O') {
            wins++;
        } else if (winner == 'X') {
            losses++;
        } else {
            ties++;
        }

        if ((i + 1) % interval == 0) {
            printf("Games: %d, Wins: %d, Losses: %d, Ties: %d\n",
                   i + 1, wins, losses, ties);
            wins = 0;
            losses = 0;
            ties = 0;
        }
    }

    printf("Training complete!\n");
}

int main(int argc, char **argv) {
    int random_games = 1000;
    const char *output_file = MODEL_FILE;

    if (argc > 1) {
        random_games = atoi(argv[1]);
    }

    if (argc > 2) {
        output_file = argv[2];
    }

    srand((unsigned)time(NULL));

    NeuralNetwork nn;
    init_neural_network(&nn);

    if (random_games > 0) {
        train_against_random(&nn, random_games);
    }

    save_neural_network(&nn, output_file);

    return 0;
}
