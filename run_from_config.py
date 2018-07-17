import yaml
from simpsons.functions import get_batches
import simpsons.helper as helper
from simpsons.model import RNN

# Config
with open("config.yml", 'r') as stream:
    try:
        config = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# TODO - Refactor these with block capitals
num_epochs = config['num_epochs']
batch_size = config['batch_size']
rnn_size = config['rnn_size']
embed_dim = config['embed_dim']
seq_length = config['seq_length']
learning_rate = config['learning_rate']
show_every_n_batches = config['show_every_n_batches']
dropout_keep_prob = config['dropout_keep_prob']
lstm_layers = config['lstm_layers']

save_dir = './save'


# Get pre-procesed data
int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()

batches = get_batches(int_text, batch_size, seq_length)
batches = batches[:200]   # TODO - remove this

rnn = RNN(int_to_vocab, rnn_size, dropout_keep_prob, lstm_layers, embed_dim)

rnn.train(num_epochs, batches, show_every_n_batches, learning_rate, save_dir)