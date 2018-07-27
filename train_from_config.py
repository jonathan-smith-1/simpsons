import yaml
from simpsons.functions import get_batches
import simpsons.helper as helper
from simpsons.model import RNN

# Config
with open("config_tiny.yml", 'r') as stream:
    try:
        config = yaml.load(stream)
    except yaml.YAMLError as exc:
        config = None
        print(exc)

# TODO - May need to actually pre-process the data here

# Get pre-procesed data
int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()

batches = get_batches(int_text, config)

rnn = RNN(int_to_vocab, config)

rnn.train(config, batches, verbose=True)
