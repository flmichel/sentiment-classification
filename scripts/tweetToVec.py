import numpy as np
glove = np.load('../embeddings.npy')
vocab = open("../data/vocab_cut.txt", "r")

words = vocab.readlines()
dimention = len(glove[0])
glove_dict = {}
for i in np.arange(glove.shape[0]):
	glove_dict.update({words[i].strip():glove[i]})

def tweet_to_vec(tweet):
	# check if the tweet has at least one word in the dict
	words_in_dict = False
	for word in tweet:
		if word in glove_dict.keys():
			words_in_dict = True
			break

	#converts a tweet to a GloVe embedding
	if words_in_dict:
		return np.mean([glove_dict[word] for word in tweet if word in glove_dict.keys()], axis = 0)
	else:
		return np.zeros(dimention)
def tweets_to_vec(tweets):
	#converts an array of tweets into an array of gloVe embedding
	return [tweet_to_vec(tweet) for tweet in tweets]
