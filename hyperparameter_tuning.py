import numpy as np
from random import seed, shuffle
from sklearn.model_selection import KFold
from simpsons.functions import get_batches
import simpsons.helper as helper
from hyperparameter_tuning_functions import generate_configs
from simpsons.model import RNN

K_FOLDS = 4

# Get pre-procesed data
int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()

# Grid search of these parameters if performed
params_grid = {'num_epochs': [2],
               'batch_size': [256],
               'rnn_size': [2, 3],
               'embed_dim': [2],
               'seq_length': [2],
               'learning_rate': [0.01],
               'show_every_n_batches': [100],
               'dropout_keep_prob': [0.9, 1.0],
               'lstm_layers': [1],
               'save_dir': ['./save']}

configs = generate_configs(params_grid)

# Keep track of best configuration
best_loss = np.inf
best_config = None

# Go through configs in random order
seed(0)
shuffle(configs)

for config in configs:
    print('')
    print('Running config:')
    print(config)

    # Get the batches of data.
    batches = get_batches(int_text, config)

    # Keep track of the validation losses.
    fold_val_losses = []

    # K-fold cross validation.
    # Batches has dims: num_batches x (input, target) x batch_size x seq_len
    kf = KFold(n_splits=K_FOLDS)

    for train_index, val_index in kf.split(batches):
        input_train, input_val = batches[train_index, 0, None, :, :], \
                                 batches[val_index, 0, None, :, :]
        target_train, target_val = batches[train_index, 1, None, :, :], \
                                   batches[val_index, 1, None, :, :]

        # RNN.train method expects the input and target data to be joined
        train_batches = np.concatenate((input_train, target_train), axis=1)
        val_batches = np.concatenate((input_val, target_val), axis=1)

        rnn = RNN(int_to_vocab, config)
        val_losses = rnn.train(config, train_batches, val_batches=val_batches)

        fold_val_losses.append(val_losses)

    # Find out the best num epochs for this config.
    # Array has same length as number of epochs.
    fold_avg_losses = np.array(fold_val_losses).mean(axis=0)

    config_best_loss = np.min(fold_avg_losses)

    # Loss from 1st epoch is stored at 0th index in fold_avg_losses
    config_best_num_epochs = np.argmin(fold_avg_losses) + 1

    if config_best_loss < best_loss:
        # New best loss
        best_loss = config_best_loss
        best_config = config
        best_config['num_epochs'] = config_best_num_epochs

    print('Best config so far:')
    print(best_config)
