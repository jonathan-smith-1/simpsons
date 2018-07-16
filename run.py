import tensorflow as tf
from tensorflow.contrib import seq2seq
from functions import get_inputs, get_init_cell, build_nn, get_batches
import helper

int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()

# Build the neural network

# Number of Epochs
num_epochs = 1 #150

# Batch Size
batch_size = 2 #256

# RNN Size
rnn_size = 5 #512

# Embedding Dimension Size
embed_dim = 10 #256

# Sequence Length
seq_length = 10

# Learning Rate
learning_rate = 0.01

# Show stats for every n number of batches
show_every_n_batches = 100

save_dir = './save'

train_graph = tf.Graph()

with train_graph.as_default():
    vocab_size = len(int_to_vocab)
    input_text, targets, lr = get_inputs()
    input_data_shape = tf.shape(input_text)
    cell, initial_state = get_init_cell(input_data_shape[0], rnn_size)   # input_data_shape[0] is the batch size
    logits, final_state = build_nn(cell, rnn_size, input_text, vocab_size, embed_dim)

    # Probabilities for generating words
    probs = tf.nn.softmax(logits, name='probs')

    # Loss function
    cost = seq2seq.sequence_loss(
        logits,
        targets,
        tf.ones([input_data_shape[0], input_data_shape[1]]))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(lr)

    # Gradient Clipping
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)


batches = get_batches(int_text, batch_size, seq_length)
batches = batches[:200]   # TODO - remove this


with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(num_epochs):
        state = sess.run(initial_state, {input_text: batches[0][0]})

        for batch_i, (x, y) in enumerate(batches):
            feed = {
                input_text: x,
                targets: y,
                initial_state: state,
                lr: learning_rate}
            train_loss, state, _ = sess.run([cost, final_state, train_op], feed)

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