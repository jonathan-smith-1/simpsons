import yaml
from simpsons.functions import get_batches
from simpsons.helper import load_data, token_lookup, create_lookup_tables, \
    preprocess_and_save_data, load_preprocess
from simpsons.model import RNN

# Config
with open("config.yml", 'r') as stream:
    try:
        config = yaml.load(stream)
    except yaml.YAMLError as exc:
        config = None
        print(exc)

# Load data
data_dir = './data/simpsons/moes_tavern_lines.txt'
text = load_data(data_dir)

# Preprocess data
preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)

# Get pre-procesed data
int_text, vocab_to_int, int_to_vocab, token_dict = load_preprocess()

batches = get_batches(int_text, config)

rnn = RNN(int_to_vocab, config)

rnn.train(config, batches, verbose=True)
