# simpsons
Using Natural Language Processing to generate scripts for The Simpsons.

My aim with this project was to get some practice and experience using NLP 
and Recurrent Neural Networks on a light-hearted application.

For this I chose to generate new scripts for The Simpsons, in particular 
scenes from [Moe's Tavern](https://simpsonswiki.com/wiki/Moe's_Tavern). An 
example extract is:
> Homer_Simpson: Right after Happy Hour!
>
> Moe_Szyslak: Drinking will help us plan.
>
> Homer_Simpson: This Valentine's crap has gone too far.
>
> Men: Yeah. / Yeah.
>
> Moe_Szyslak: (AMID MEN'S REACTIONS) You got that right!
>
> Seymour_Skinner: Edna won't even let me clap her erasers.
>
> ...

I started this project for an assignment for Udacity's excellent [Deep 
Learning Nanodegree](https://www.udacity.com/course/deep-learning-nanodegree--nd101),
 which I would highly recommend.  Udacity supplied the dataset and it has 
 around 4000 lines of text.  I have since extended the project as a learning
  experience to to explore RNNs and NLP further.
 
# How to run
Once you have installed the packages in _requirements.txt_ then the three 
jupyter notebooks should be sufficient and self-explanatory:
- preprocess_data.ipynb
- train_model.ipynb
- generate_scripts.ipynb

Training with the hyperparameters in `config.yml` the model is able to 
produce scripts such as:

> moe_szyslak: homer, i work? while greystash, watchin' i whole roof(clears 
button glass) now you got me special ingredient to healthier" baby in no.
we later after it? while i gonna love!. now we suburban grabs kneeling we 
good who, gr-aargh, it's missed.
>
>football_announcer: sadder somebody orders all something and thank my 
fortune. the see jack honor, i on up?
>
>lenny_leonard: uh, ask moe?
>
>moe_szyslak:(white sign) i, you, he drink again. lenny. you right! aw is our
 trip? instead to wedding cleaned, milhouse, not your spews way guy-- 
 there's would good something on the.
>
>homer_simpson: whatever and sure-- what means i like the name and get a 
crummy people recorder virtual boat. i drunk?
eggshell. you won a for just you could lookin'.(quarry to phone) tabs.
>
>homer_simpson:(turned uncle what tick sound) listen, five crowd is hate you 
lost a ambrosia, pretty whiny!
>
>moe_szyslak: up i'm yieldin'

This is still gibberish, of course, but you can see that it has learnt about
 the structure of language and scripts.  All the following had to be 
 learnt:
 - The first word of each line is the speaker, followed by a colon.
 - Opened brackets should be closed.
 - Question marks come at the end of sentences.
 
 Furthermore, sentences such as this show that some short-term linguistic 
 structure has been learnt - this is not total gibberish:
 >football_announcer: sadder somebody orders all something and thank my 
fortune. the see jack honor, i on up?

And it does bear a resemblance to the linguistic rhythm of the original 
data.

# Ideas for improvements
- Obviously more data would help.  4000 lines of text and a large and 
colloquial vocabulary make this a challenging dataset.
- Transfer learning for word embeddings would help.  Either a shallow 
embedding such as the well-known [word2vec](https://en.wikipedia
.org/wiki/Word2vec) or [GloVe](https://nlp.stanford.edu/projects/glove/), or
 something a little deeper such as one of Tensorflow Hub's [Text modules](https://www.tensorflow.org/hub/modules/text).



# Acknowledgements
- The data and original code came from Udacity.  The code for pre-processing
 the text data is almost unchanged, but the rest has been updated by me.  
 The overall structure of the project is largely unchanged from the original
  Udacity assignment.
