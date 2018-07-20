import simpsons.helper as helper
from simpsons.generator import ScriptGenerator

_, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
seq_length, load_dir = helper.load_params()

gen_length = 200

# homer_simpson, moe_szyslak, or barney_gumble
prime_word = 'moe_szyslak'

gen = ScriptGenerator(vocab_to_int, int_to_vocab, seq_length, token_dict)
tv_script = gen.generate(gen_length, prime_word, load_dir)

print(tv_script)