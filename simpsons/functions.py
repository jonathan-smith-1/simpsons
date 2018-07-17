import numpy as np
import tensorflow as tf


def get_inputs():
    """
    Create Tensorflow Placeholders for input, targets, and learning rate.

    Returns: Tuple (input, targets, learning rate)

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
    Create a multi-layered RNN Cell and initialize it.

    Args:
        batch_size (int): Size of batches
        rnn_size (int): Size of RNN, i.e how many units.
        keep_prob (float): Dropout keep probability, in [0.0, 1.0].
        lstm_layers: Number of LSTM layers.

    Returns:
        Tuple (cell, initialize state).  The initialize state tensor is the
        state input to the RNN

    """
    # TODO - Investigate the init state - should be integrated in rnn, not cell

    cell = tf.nn.rnn_cell.MultiRNNCell(
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
    Create a RNN using a RNN Cell.

    Args:
        cell: A RNN cell that forms the basis of this RNN.
        inputs: Tensor of input data to the RNN

    Returns: Tuple (Outputs, Final State)

    """

    outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)

    final_state = tf.identity(final_state, 'final_state')

    return outputs, final_state


def build_nn(cell, input_data, vocab_size, embed_dim):
    """
    Build part of the neural network.

    Args:
        cell: RNN cell
        input_data: Input data
        vocab_size: Vocabulary size
        embed_dim: Number of embedding dimensions

    Returns: Tuple (Logits, FinalState)

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

    Args:
        int_text: Text with the words replaced by their ids
        batch_size: The size of batch
        seq_length: The length of sequence

    Returns:
        Batches as a Numpy array.

    The returned batches are a Numpy array with the shape
    (number of batches, 2, batch size, sequence length).

    Each batch contains two elements (corresponding to positions 0 and 1 on
    axis 1 of the returned batches):
    - The first element is a single batch of inputs with the shape
    [batch size, sequence length]
    - The second element is a single batch of targets with the shape
    [batch size, sequence length]

    The last batch is dropped if there is insufficient data.

    For example,

    `get_batches([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
    14, 15, 16, 17, 18, 19, 20], 3, 2)` would return a Numpy array of the
    following:

    [
      # First Batch
      [
        # Batch of Input
        [[ 1  2], [ 7  8], [13 14]]
        # Batch of targets
        [[ 2  3], [ 8  9], [14 15]]
      ]

      # Second Batch
      [
        # Batch of Input
        [[ 3  4], [ 9 10], [15 16]]
        # Batch of targets
        [[ 4  5], [10 11], [16 17]]
      ]

      # Third Batch
      [
        # Batch of Input
        [[ 5  6], [11 12], [17 18]]
        # Batch of targets
        [[ 6  7], [12 13], [18  1]]
      ]
    ]

    Notice that the last target value in the last batch is the first input
    value of the first batch. In this case, `1`.
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


