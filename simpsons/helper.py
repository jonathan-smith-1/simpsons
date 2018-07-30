# This code was provided by Udacity as part of the Deep Learning Nanodegree

import os
import pickle


def load_data(path):
    """
    Load Dataset from File
    """
    input_file = os.path.join(path)
    with open(input_file, "r") as f:
        data = f.read()

    return data


def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary.

    Args:
        text: The text of tv scripts split into words

    Returns:
        A tuple of dicts (vocab_to_int, int_to_vocab)

    """
    vocab_to_int = {word: i for i, word in enumerate(set(text))}
    int_to_vocab = {i: word for word, i in vocab_to_int.items()}

    return vocab_to_int, int_to_vocab


def token_lookup():
    """
    Generate a dict to turn punctuation into a token.

    Returns:
        Tokenized dictionary where the key is the punctuation and the
        value is the token

    """
    punc_dict = {'.': '||period||',
                 ',': '||comma||',
                 '"': '||quotationmark||',
                 ';': '||semicolon||',
                 '!': '||exclamationmark||',
                 '?': '||questionmark||',
                 '(': '||leftparen||',
                 ')': '||rightparen||',
                 '--': '||dash||',
                 '\n': '||return||'}

    return punc_dict


def preprocess_and_save_data(dataset_path, token_lookup, create_lookup_tables):
    """
    Preprocess Text Data.
    """
    text = load_data(dataset_path)
    
    # Ignore notice, since we don't use it for analysing the data
    text = text[81:]

    token_dict = token_lookup()
    for key, token in token_dict.items():
        text = text.replace(key, ' {} '.format(token))

    text = text.lower()
    text = text.split()

    vocab_to_int, int_to_vocab = create_lookup_tables(text)
    int_text = [vocab_to_int[word] for word in text]
    pickle.dump((int_text, vocab_to_int, int_to_vocab, token_dict), open('preprocess.p', 'wb'))


def load_preprocess():
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    return pickle.load(open('preprocess.p', mode='rb'))


def save_params(params):
    """
    Save parameters to file
    """
    pickle.dump(params, open('params_grid.p', 'wb'))


def load_params():
    """
    Load parameters from file
    """
    return pickle.load(open('params_grid.p', mode='rb'))
