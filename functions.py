import numpy as np
import tensorflow as tf


def get_inputs():
    """
    Create TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate)
    """
    input_ = tf.placeholder(tf.int32, shape=(None, None), name='input')
    targets = tf.placeholder(tf.int32, shape=(None, None), name='target')
    learning_rate = tf.placeholder(tf.float32)

    return input_, targets, learning_rate


def build_cell(num_units, keep_prob):
    lstm = tf.contrib.rnn.BasicLSTMCell(num_units)
    drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    return drop


def get_init_cell(batch_size, rnn_size, keep_prob, lstm_layers):
    """
    Create an RNN Cell and initialize it.
    :param batch_size: Size of batches
    :param rnn_size: Size of RNNs
    :return: Tuple (cell, initialize state)

    :keep_prob: Dropout keep probability
    :lstm_layers: Number of lstm layers
    """

    cell = tf.contrib.rnn.MultiRNNCell(
        [build_cell(rnn_size, keep_prob) for _ in range(lstm_layers)])
    initial_state = cell.zero_state(batch_size, tf.float32)
    initial_state = tf.identity(initial_state, name='initial_state')

    return cell, initial_state


def get_embed(input_data, vocab_size, embed_dim):
    """
    Create embedding for <input_data>.
    :param input_data: TF placeholder for text input.
    :param vocab_size: Number of words in vocabulary.
    :param embed_dim: Number of embedding dimensions
    :return: Embedded input.
    """

    # use get_variable to get default Xavier initializer
    embedding = tf.get_variable('embedding', (vocab_size, embed_dim))
    embed = tf.nn.embedding_lookup(embedding, input_data)

    return embed


def build_rnn(cell, inputs):
    """
    Create a RNN using a RNN Cell
    :param cell: RNN Cell
    :param inputs: Input text data
    :return: Tuple (Outputs, Final State)
    """

    outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)

    final_state = tf.identity(final_state, 'final_state')

    return outputs, final_state


def build_nn(cell, rnn_size, input_data, vocab_size, embed_dim):
    """
    Build part of the neural network
    :param cell: RNN cell
    :param rnn_size: Size of rnns (i.e number of hidden units)
    :param input_data: Input data
    :param vocab_size: Vocabulary size
    :param embed_dim: Number of embedding dimensions
    :return: Tuple (Logits, FinalState)
    """

    # apply embedding to input
    embed_input = get_embed(input_data, vocab_size, embed_dim)

    # build rnn
    outputs, final_state = build_rnn(cell, embed_input)

    # Connect the RNN outputs to a linear output layer
    logits = tf.contrib.layers.fully_connected(outputs, vocab_size,
                                               activation_fn=None)

    return logits, final_state


def get_batches(int_text, batch_size, seq_length):
    """
    Return batches of input and target
    :param int_text: Text with the words replaced by their ids
    :param batch_size: The size of batch
    :param seq_length: The length of sequence
    :return: Batches as a Numpy array
    """

    elements_per_batch = batch_size * seq_length
    n_batches = len(int_text) // elements_per_batch
    elements_used = elements_per_batch * n_batches

    input_data = np.array(int_text[:elements_used]).reshape(
        (1, batch_size, n_batches, seq_length))
    input_data = input_data.transpose((2, 0, 1, 3))

    output_elements = int_text[1:elements_used] + int_text[0:1]
    output_data = np.array(output_elements).reshape(
        (1, batch_size, n_batches, seq_length))
    output_data = output_data.transpose((2, 0, 1, 3))

    # join together over axis 1 to form output
    return np.concatenate((input_data, output_data), axis=1)


