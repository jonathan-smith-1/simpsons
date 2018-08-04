import yaml
import simpsons.helper as helper
from simpsons.generator import ScriptGenerator

# Config
with open("config.yml", 'r') as stream:
    try:
        config = yaml.load(stream)
    except yaml.YAMLError as exc:
        config = None
        print(exc)

_, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()

gen_length = 200

# homer_simpson, moe_szyslak, or barney_gumble
prime_word = 'moe_szyslak'

gen = ScriptGenerator(vocab_to_int, int_to_vocab, token_dict, config)
tv_script = gen.generate(gen_length, prime_word, config)

print(tv_script)