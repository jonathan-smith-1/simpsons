import tensorflow as tf
from tensorflow.contrib import seq2seq
from functions import get_inputs, get_init_cell, build_nn


class RNN:
    """Recurrent Neural Network for text sequence generation"""

    def __init__(self, int_to_vocab, rnn_size, dropout_keep_prob,
                 lstm_layers, embed_dim):
        """
        Construct Recurrent Neural Network.

        Args:
            int_to_vocab: Dictionary mapping integers to words in vocabulary.
            rnn_size: Integer size (i.e. dimensions) of all layers in RNN.
            dropout_keep_prob: Dropout
            lstm_layers: Number of layers.
            embed_dim: Embedding dimension.
        """

        self.train_graph = tf.Graph()

        with self.train_graph.as_default():

            vocab_size = len(int_to_vocab)

            self.input_text, self.targets, self.lr = get_inputs()

            input_data_shape = tf.shape(self.input_text)

            cell, self.initial_state = get_init_cell(input_data_shape[0],
                                                     rnn_size,
                                                     dropout_keep_prob,
                                                     lstm_layers)

            logits, self.final_state = build_nn(cell,
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
              learning_rate, save_dir):

        """
        Train the Recurrent Neural Network.

        Args:
            num_epochs: Number of epochs to train.
            batches: Input data already processed into batches.
            show_every_n_batches: Show progress every n batches.
            learning_rate: Learning rate for training.
            save_dir: Location where checkpoint is saved.

        Returns:
            Nothing

        """
        with tf.Session(graph=self.train_graph) as sess:
            sess.run(tf.global_variables_initializer())

            for epoch_i in range(num_epochs):
                state = sess.run(self.initial_state, {self.input_text: batches[0][0]})

                for batch_i, (x, y) in enumerate(batches):
                    feed = {
                        self.input_text: x,
                        self.targets: y,
                        self.initial_state: state,
                        self.lr: learning_rate}
                    train_loss, state, _ = sess.run([self.cost, self.final_state, self.train_op], feed)

                    # Show every <show_every_n_batches> batches
                    if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                        print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                            epoch_i,
                            batch_i,
                            len(batches),
                            train_loss))

            # Save Model
            saver = tf.train.Saver()
            saver.save(sess, save_dir)
            print('Model Trained and Saved')