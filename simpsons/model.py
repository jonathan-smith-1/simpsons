import tensorflow as tf
import numpy as np
from tensorflow.contrib import seq2seq
from simpsons.functions import get_inputs, get_init_cell, build_nn


class RNN:
    """Recurrent Neural Network for text sequence generation"""

    def __init__(self, int_to_vocab, config):
        """
        Construct Recurrent Neural Network.

        Args:
            config: Dictionary of configuration parameters.
        """

        lstm_layers = config['lstm_layers']
        rnn_size = config['rnn_size']
        embed_dim = config['embed_dim']
        dropout_keep_prob = config['dropout_keep_prob']

        self.train_graph = tf.Graph()

        with self.train_graph.as_default():

            vocab_size = len(int_to_vocab)

            self.input_text, self.targets, self.lr = get_inputs()

            input_data_shape = tf.shape(self.input_text)

            # Unpack the shape of what the RNN outputs (array) to the shape that the RNN expects in its next training
            #  step (tuples)
            self.init_state = tf.placeholder(tf.float32,
                                             [lstm_layers, 2, None, rnn_size], name='initial_state')

            state_per_layer_list = tf.unstack(self.init_state, axis=0)

            rnn_tuple_state = tuple(
                [tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
                 for idx in range(lstm_layers)]
            )

            cell = get_init_cell(rnn_size, dropout_keep_prob, lstm_layers)

            logits, self.final_state = build_nn(cell,
                                                rnn_tuple_state,
                                                self.input_text,
                                                vocab_size,
                                                embed_dim)

            # Probabilities for generating words
            # Not used locally but referred to by tensor name during text
            # generation.
            probs = tf.nn.softmax(logits, name='probs')

            # Loss function
            self.cost = seq2seq.sequence_loss(
                logits,
                self.targets,
                tf.ones([input_data_shape[0], input_data_shape[1]]))

            # Optimizer
            optimizer = tf.train.AdamOptimizer(self.lr)

            # Gradient Clipping
            gradients = optimizer.compute_gradients(self.cost)
            capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var)
                                for grad, var in gradients if grad is not None]

            self.train_op = optimizer.apply_gradients(capped_gradients)

    def train(self, batches, config, verbose=False):

        """
        Train the Recurrent Neural Network.

        Args:
            batches: (numpy array): Input data already processed into batches
                                   by function.get_batches().
            config: Dictionary of parameters.
            verbose (bool): Print outputs

        Returns:
            Nothing

        """

        lstm_layers = config['lstm_layers']
        batch_size = config['batch_size']
        rnn_size = config['rnn_size']
        learning_rate = config['learning_rate']
        num_epochs = config['num_epochs']
        save_dir = config['save_dir']
        show_every_n_batches = config['show_every_n_batches']

        with tf.Session(graph=self.train_graph) as sess:
            sess.run(tf.global_variables_initializer())

            for epoch_i in range(num_epochs):
                state = np.zeros([lstm_layers, 2, batch_size, rnn_size])  # init

                for batch_i, (x, y) in enumerate(batches):
                    feed = {
                        self.input_text: x,
                        self.targets: y,
                        self.init_state: state,
                        self.lr: learning_rate}
                    train_loss, state, _ = sess.run([self.cost, self.final_state, self.train_op], feed)

                    # Show every <show_every_n_batches> batches
                    if verbose and (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                        print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                            epoch_i,
                            batch_i,
                            len(batches),
                            train_loss))

            # Save Model
            saver = tf.train.Saver()
            saver.save(sess, save_dir)
            if verbose:
                print('Model Trained and Saved')