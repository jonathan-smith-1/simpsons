import tensorflow as tf
import numpy as np
from tensorflow.contrib import seq2seq
from simpsons.functions import get_inputs, get_init_cell, build_nn


class RNN:
    """Recurrent Neural Network for text sequence generation"""

    def __init__(self, int_to_vocab, rnn_size, dropout_keep_prob,
                 lstm_layers, embed_dim, batch_size):
        """
        Construct Recurrent Neural Network.

        Args:
            int_to_vocab: Dictionary mapping integers to words in vocabulary.
            rnn_size (int): Integer size (i.e. dimensions) of all layers in
                            RNN.
            dropout_keep_prob (float): Dropout keep probability.
            lstm_layers (int): Number of layers.
            embed_dim (int): Embedding dimension.
        """

        self.train_graph = tf.Graph()

        with self.train_graph.as_default():

            vocab_size = len(int_to_vocab)

            self.input_text, self.targets, self.lr = get_inputs()

            input_data_shape = tf.shape(self.input_text)

            # Unpack the shape of what the RNN outputs (array) to the shape that the RNN expects in its next training
            #  step (tuples)
            self.init_state = tf.placeholder(tf.float32, [lstm_layers, 2, batch_size, rnn_size])

            state_per_layer_list = tf.unstack(self.init_state, axis=0)

            rnn_tuple_state = tuple(
                [tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
                 for idx in range(lstm_layers)]
            )

            cell = get_init_cell(input_data_shape[0], rnn_size, dropout_keep_prob, lstm_layers)

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

    def train(self, num_epochs, batches, show_every_n_batches,
              learning_rate, save_dir, verbose=False):

        """
        Train the Recurrent Neural Network.

        Args:
            num_epochs (int): Number of epochs to train.
            batches (numpy array): Input data already processed into batches
                                   by function.get_batches().
            show_every_n_batches (int): Show progress every n batches.
            learning_rate (float): Learning rate for training.
            save_dir (string): Location where checkpoint is saved.
            verbose (bool): Print outputs

        Returns:
            Nothing

        """
        with tf.Session(graph=self.train_graph) as sess:
            sess.run(tf.global_variables_initializer())

            for epoch_i in range(num_epochs):
                initial_state = np.zeros(sess.run(tf.shape(self.init_state)))

                for batch_i, (x, y) in enumerate(batches):
                    feed = {
                        self.input_text: x,
                        self.targets: y,
                        self.init_state: initial_state,
                        self.lr: learning_rate}
                    train_loss, final_state, _ = sess.run([self.cost, self.final_state, self.train_op], feed)

                    initial_state = final_state

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