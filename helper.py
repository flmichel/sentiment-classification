import numpy as np

def load_train(pos_path, neg_path):
    """
    Load all the training tweets (positive and negative)
    Return a list of tuples, the first element is -1 for positive tweets and 1
    for negative tweets and the second element is the list of words in the tweet
    """
    words_tr = []
    y_tr = []

    # load neg_path tweets
    with open(neg_path) as f:
        for tweet in f:
            words_tr.append(tweet.split(' '))
            y_tr.append(-1)

    # load pos_path tweets
    with open(pos_path) as f:
        for tweet in f:
            words_tr.append(tweet.split(' '))
            y_tr.append(1)
            
    return words_tr, y_tr

def load_test(test_path):
    """
    Load all the test tweets
    Return a list of tuples, the first element is the index of the tweet and
    the second element is the list of words in the tweet
    """
    words_te = []
    inds = []
    # load pos_path tweets
    with open(test_path) as f:
        for tweet in f:
            ind, tweet = tweet.split(',', 1)
            words_te.append(tweet.split(' '))
            inds.append(inds)

        return words_te, inds
