#include "common.h"

#define FINETUNE_LEARNING_RATE 0.01f

static void supervised_backprop(NeuralNetwork *nn, float *target_probs, float learning_rate) {
    float output_deltas[NN_OUTPUT_SIZE];
    float hidden_deltas[NN_HIDDEN_SIZE];

    for (int i = 0; i < NN_OUTPUT_SIZE; i++) {
        output_deltas[i] = nn->outputs[i] - target_probs[i];
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

static int train_one_csv_row(NeuralNetwork *nn, char player, int move, const char *board_str) {
    if (move < 0 || move >= BOARD_CELLS) {
        return 0;
    }

    GameState state;
    init_game(&state);

    for (int i = 0; i < BOARD_CELLS; i++) {
        char ch = board_str[i];

        if (ch == '\0') {
            return 0;
        }

        if (ch == '.' || ch == 'X' || ch == 'O') {
            state.board[i] = ch;
        } else {
            return 0;
        }
    }

    if (state.board[move] != '.') {
        return 0;
    }

    float inputs[NN_INPUT_SIZE];
    float target_probs[NN_OUTPUT_SIZE];

    board_to_inputs(&state, inputs);
    forward_pass(nn, inputs);

    for (int i = 0; i < NN_OUTPUT_SIZE; i++) {
        target_probs[i] = 0.0f;
    }

    target_probs[move] = 1.0f;

    supervised_backprop(nn, target_probs, FINETUNE_LEARNING_RATE);

    return 1;
}

static int finetune_from_csv(NeuralNetwork *nn, const char *csv_path, int epochs) {
    int total_updates = 0;

    printf("Fine-tuning from CSV: %s for %d epochs\n", csv_path, epochs);

    for (int epoch = 0; epoch < epochs; epoch++) {
        FILE *file = fopen(csv_path, "r");

        if (file == NULL) {
            printf("Could not open CSV file: %s\n", csv_path);
            return 0;
        }

        char line[2048];

        /* Skip header. */
        fgets(line, sizeof(line), file);

        int epoch_updates = 0;

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
            char winner = winner_str[0];
            int move = atoi(move_str);

            /* For tactical CSV, rows are player=O, source=human, winner=O.
             * For human logs, this keeps moves from the eventual winner.
             */
            if (strcmp(source, "human") != 0) {
                continue;
            }

            if (winner != player) {
                continue;
            }

            if (train_one_csv_row(nn, player, move, board_str)) {
                epoch_updates++;
                total_updates++;
            }
        }

        fclose(file);

        if (epoch == 0 || (epoch + 1) % 10 == 0 || epoch + 1 == epochs) {
            printf("Epoch %d/%d, updates: %d\n", epoch + 1, epochs, epoch_updates);
        }
    }

    printf("Fine-tuning complete. Total updates: %d\n", total_updates);

    return 1;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        printf("Usage:\n");
        printf("  %s <input_model.bin> <csv_file> <output_model.bin> <epochs>\n", argv[0]);
        return 1;
    }

    const char *input_model = argv[1];
    const char *csv_file = argv[2];
    const char *output_model = argv[3];
    int epochs = atoi(argv[4]);

    NeuralNetwork nn;

    if (!load_neural_network(&nn, input_model)) {
        printf("Could not load input model: %s\n", input_model);
        return 1;
    }

    if (!finetune_from_csv(&nn, csv_file, epochs)) {
        return 1;
    }

    if (!save_neural_network(&nn, output_model)) {
        printf("Could not save output model: %s\n", output_model);
        return 1;
    }

    return 0;
}
