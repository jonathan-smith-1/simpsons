import numpy as np
import tensorflow as tf


class ScriptGenerator:

    def __init__(self, vocab_to_int, int_to_vocab, token_dict, config):

        np.random.seed(42)

        self.vocab_to_int = vocab_to_int
        self.seq_length = config['seq_length']
        self.token_dict = token_dict
        self.int_to_vocab = int_to_vocab

    @staticmethod
    def get_tensors(loaded_graph):

        """
        Get input, initial state, final state, and probabilities tensor from
        <loaded_graph>
        :param loaded_graph: TensorFlow graph loaded from file
        :return: Tuple (InputTensor, InitialStateTensor, FinalStateTensor,
        ProbsTensor)
        """
        return (loaded_graph.get_tensor_by_name("input:0"),
                loaded_graph.get_tensor_by_name("initial_state:0"),
                loaded_graph.get_tensor_by_name("final_state:0"),
                loaded_graph.get_tensor_by_name("probs:0"))

    @staticmethod
    def pick_word(probabilities, int_to_vocab):
        """
        Pick the next word in the generated text
        :param probabilities: Probabilites of the next word
        :param int_to_vocab: Dictionary of word ids as the keys and words as
        the values
        :return: String of the predicted word
        """
        i = np.random.choice(range(len(probabilities)), p=probabilities)
        return int_to_vocab[i]

    def generate(self, gen_length, prime_word, config):

        load_dir = config['save_dir']

        loaded_graph = tf.Graph()
        with tf.Session(graph=loaded_graph) as sess:
            # Load saved model
            loader = tf.train.import_meta_graph(load_dir + '.meta')
            loader.restore(sess, load_dir)

            # Get Tensors from loaded model
            input_text, initial_state, final_state, probs = self.get_tensors(
                loaded_graph)

            # Sentences generation setup
            gen_sentences = [prime_word + ':']
            batch_size = 1
            lstm_layers = config['lstm_layers']
            rnn_size = config['rnn_size']
            prev_state = np.zeros([lstm_layers, 2, batch_size, rnn_size])  # init

            # Generate sentences
            for n in range(gen_length):
                # Dynamic Input
                dyn_input = [
                    [self.vocab_to_int[word] for word in gen_sentences[-self.seq_length:]]]
                dyn_seq_length = len(dyn_input[0])

                # Get Prediction
                probabilities, prev_state = sess.run(
                    [probs, final_state],
                    {input_text: dyn_input, initial_state: prev_state})

                pred_word = self.pick_word(probabilities[0][dyn_seq_length-1], self.int_to_vocab)

                gen_sentences.append(pred_word)

            # Remove tokens
            tv_script = ' '.join(gen_sentences)
            for key, token in self.token_dict.items():
                tv_script = tv_script.replace(' ' + token.lower(), key)
            tv_script = tv_script.replace('\n ', '\n')
            tv_script = tv_script.replace('( ', '(')

            return tv_script
