#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define BOARD_SIZE 15
#define BOARD_CELLS (BOARD_SIZE * BOARD_SIZE)
#define WIN_LENGTH 5

#define NN_INPUT_SIZE (BOARD_CELLS * 2)
#define NN_HIDDEN_SIZE 64
#define NN_OUTPUT_SIZE BOARD_CELLS

#define MODEL_FILE "gomoku_nn.bin"

typedef struct {
    char board[BOARD_CELLS];  // '.', 'X', or 'O'
    int current_player;       // 0 for X, 1 for O
} GameState;

typedef struct {
    float weights_ih[NN_INPUT_SIZE * NN_HIDDEN_SIZE];
    float weights_ho[NN_HIDDEN_SIZE * NN_OUTPUT_SIZE];
    float biases_h[NN_HIDDEN_SIZE];
    float biases_o[NN_OUTPUT_SIZE];

    float inputs[NN_INPUT_SIZE];
    float hidden[NN_HIDDEN_SIZE];
    float raw_logits[NN_OUTPUT_SIZE];
    float outputs[NN_OUTPUT_SIZE];
} NeuralNetwork;

static void init_game(GameState *state) {
    for (int i = 0; i < BOARD_CELLS; i++) {
        state->board[i] = '.';
    }
    state->current_player = 0;
}

static float relu(float x) {
    return x > 0 ? x : 0;
}

static float relu_derivative(float x) {
    return x > 0 ? 1.0f : 0.0f;
}

static void softmax(float *input, float *output, int size) {
    float max_val = input[0];

    for (int i = 1; i < size; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    float sum = 0.0f;

    for (int i = 0; i < size; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }

    if (sum > 0) {
        for (int i = 0; i < size; i++) {
            output[i] /= sum;
        }
    } else {
        for (int i = 0; i < size; i++) {
            output[i] = 1.0f / size;
        }
    }
}

static void board_to_inputs(GameState *state, float *inputs) {
    for (int i = 0; i < BOARD_CELLS; i++) {
        if (state->board[i] == '.') {
            inputs[i * 2] = 0;
            inputs[i * 2 + 1] = 0;
        } else if (state->board[i] == 'X') {
            inputs[i * 2] = 1;
            inputs[i * 2 + 1] = 0;
        } else {
            inputs[i * 2] = 0;
            inputs[i * 2 + 1] = 1;
        }
    }
}

static void forward_pass(NeuralNetwork *nn, float *inputs) {
    memcpy(nn->inputs, inputs, NN_INPUT_SIZE * sizeof(float));

    for (int i = 0; i < NN_HIDDEN_SIZE; i++) {
        float sum = nn->biases_h[i];

        for (int j = 0; j < NN_INPUT_SIZE; j++) {
            sum += inputs[j] * nn->weights_ih[j * NN_HIDDEN_SIZE + i];
        }

        nn->hidden[i] = relu(sum);
    }

    for (int i = 0; i < NN_OUTPUT_SIZE; i++) {
        nn->raw_logits[i] = nn->biases_o[i];

        for (int j = 0; j < NN_HIDDEN_SIZE; j++) {
            nn->raw_logits[i] += nn->hidden[j] * nn->weights_ho[j * NN_OUTPUT_SIZE + i];
        }
    }

    softmax(nn->raw_logits, nn->outputs, NN_OUTPUT_SIZE);
}

static int check_direction(GameState *state, int row, int col, int dr, int dc, char symbol) {
    for (int k = 0; k < WIN_LENGTH; k++) {
        int r = row + dr * k;
        int c = col + dc * k;

        if (r < 0 || r >= BOARD_SIZE || c < 0 || c >= BOARD_SIZE) {
            return 0;
        }

        if (state->board[r * BOARD_SIZE + c] != symbol) {
            return 0;
        }
    }

    return 1;
}

static int check_game_over(GameState *state, char *winner) {
    for (int row = 0; row < BOARD_SIZE; row++) {
        for (int col = 0; col < BOARD_SIZE; col++) {
            char symbol = state->board[row * BOARD_SIZE + col];

            if (symbol == '.') {
                continue;
            }

            if (check_direction(state, row, col, 0, 1, symbol) ||
                check_direction(state, row, col, 1, 0, symbol) ||
                check_direction(state, row, col, 1, 1, symbol) ||
                check_direction(state, row, col, 1, -1, symbol)) {
                *winner = symbol;
                return 1;
            }
        }
    }

    for (int i = 0; i < BOARD_CELLS; i++) {
        if (state->board[i] == '.') {
            return 0;
        }
    }

    *winner = 'T';
    return 1;
}

static int get_random_move(GameState *state) {
    int legal_moves[BOARD_CELLS];
    int count = 0;

    for (int i = 0; i < BOARD_CELLS; i++) {
        if (state->board[i] == '.') {
            legal_moves[count++] = i;
        }
    }

    if (count == 0) {
        return -1;
    }

    return legal_moves[rand() % count];
}

static int get_computer_move(GameState *state, NeuralNetwork *nn, int display_top_moves) {
    float inputs[NN_INPUT_SIZE];

    board_to_inputs(state, inputs);
    forward_pass(nn, inputs);

    int best_move = -1;
    float best_prob = -1.0f;

    for (int i = 0; i < BOARD_CELLS; i++) {
        if (state->board[i] == '.' && nn->outputs[i] > best_prob) {
            best_prob = nn->outputs[i];
            best_move = i;
        }
    }

    if (display_top_moves) {
        printf("Top move candidates:\n");

        int selected[5];
        for (int i = 0; i < 5; i++) {
            selected[i] = -1;
        }

        for (int rank = 0; rank < 5; rank++) {
            int top_move = -1;
            float top_prob = -1.0f;

            for (int i = 0; i < BOARD_CELLS; i++) {
                if (state->board[i] != '.') {
                    continue;
                }

                int already_selected = 0;
                for (int j = 0; j < rank; j++) {
                    if (selected[j] == i) {
                        already_selected = 1;
                        break;
                    }
                }

                if (!already_selected && nn->outputs[i] > top_prob) {
                    top_prob = nn->outputs[i];
                    top_move = i;
                }
            }

            if (top_move >= 0) {
                selected[rank] = top_move;
                printf("  #%d position %d: %.2f%%\n", rank + 1, top_move, top_prob * 100.0f);
            }
        }
    }

    return best_move;
}

static void display_board(GameState *state) {
    printf("\n");
    for (int row = 0; row < BOARD_SIZE; row++) {
        for (int col = 0; col < BOARD_SIZE; col++) {
            int pos = row * BOARD_SIZE + col;
            if (state->board[pos] == '.') {
                printf("%3d", pos);
            } else {
                printf("  %c", state->board[pos]);
            }
        }
        printf("\n");
    }
    printf("\n");
}

static int save_neural_network(NeuralNetwork *nn, const char *filename) {
    FILE *file = fopen(filename, "wb");

    if (file == NULL) {
        printf("Error opening file for writing: %s\n", filename);
        return 0;
    }

    int ok = 1;

    ok &= fwrite(nn->weights_ih, sizeof(float), NN_INPUT_SIZE * NN_HIDDEN_SIZE, file) == NN_INPUT_SIZE * NN_HIDDEN_SIZE;
    ok &= fwrite(nn->weights_ho, sizeof(float), NN_HIDDEN_SIZE * NN_OUTPUT_SIZE, file) == NN_HIDDEN_SIZE * NN_OUTPUT_SIZE;
    ok &= fwrite(nn->biases_h, sizeof(float), NN_HIDDEN_SIZE, file) == NN_HIDDEN_SIZE;
    ok &= fwrite(nn->biases_o, sizeof(float), NN_OUTPUT_SIZE, file) == NN_OUTPUT_SIZE;

    fclose(file);

    if (ok) {
        printf("Neural network saved to %s\n", filename);
    } else {
        printf("Error saving neural network to %s\n", filename);
    }

    return ok;
}

static int load_neural_network(NeuralNetwork *nn, const char *filename) {
    FILE *file = fopen(filename, "rb");

    if (file == NULL) {
        printf("Error opening file for reading: %s\n", filename);
        return 0;
    }

    int ok = 1;

    ok &= fread(nn->weights_ih, sizeof(float), NN_INPUT_SIZE * NN_HIDDEN_SIZE, file) == NN_INPUT_SIZE * NN_HIDDEN_SIZE;
    ok &= fread(nn->weights_ho, sizeof(float), NN_HIDDEN_SIZE * NN_OUTPUT_SIZE, file) == NN_HIDDEN_SIZE * NN_OUTPUT_SIZE;
    ok &= fread(nn->biases_h, sizeof(float), NN_HIDDEN_SIZE, file) == NN_HIDDEN_SIZE;
    ok &= fread(nn->biases_o, sizeof(float), NN_OUTPUT_SIZE, file) == NN_OUTPUT_SIZE;

    fclose(file);

    if (ok) {
        printf("Neural network loaded from %s\n", filename);
    } else {
        printf("Error loading neural network from %s\n", filename);
    }

    return ok;
}

#endif
