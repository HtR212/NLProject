from fitbert import FitBert


# in theory you can pass a model_name and tokenizer, but currently only
# bert-large-uncased and BertTokenizer are available
# this takes a while and loads a whole big BERT into memory
fb = FitBert(model_name="bert-base-uncased")

# masked_string = "Why Bert, you're looking ***mask*** today!"
masked_string = "From Monday to Friday most people are busy working or studying, but in the evenings and weekends they are free and  ***mask***  themselves. Some watch television or go to the movies, others  take part in  sports. This is decided by their own  ***mask***  . There are many different ways to spend our  ***mask***  time. Almost everyone has some kind of  ***mask***  : it may be something from collecting stamps to  ***mask***  model planes. Some hobbies are very  ***mask***  , but others don't cost anything at all. Some collections are worth  a lot  of money, others are valuable only to their owners. I know a man who has a coin collection worth several  thousand  dollars. A short time ago he bought a rare fifty-cent piece which  ***mask***  him $250!He was very happy about this collection and thought the price was all right . On the other hand, my youngest brother collects match boxes. He has almost 600 of them, but I wonder  ***mask***  they are worth any money. However,  ***mask***  my brother they are quite valuable .  ***mask***  makes him happier than to find a new match box for his collection. That's what a hobby means, I think. It is something we  ***mask***  to do in our free time just for the  ***mask***  of it . The value in dollars is not important. but the pleasure it gives us is."

options = ['buff', 'handsome', 'strong']

ranked_options = fb.rank(masked_string, options=options)
print(ranked_options)
