import numpy as np
import pandas as pd

def load_train(pos_path, neg_path, nrows):
    """
    Load all the training tweets (positive and negative)
    Return a dataframe, the first column is 'label', -1 for positive tweets and 1
    for negative tweets and the second column is 'tweet, the list of words in the tweet
    """
    df_pos_tweets = pd.read_table(pos_path, names=['tweet'], sep ="\n", header=None, nrows=nrows)
    df_pos_tweets['label'] = 1

    df_neg_tweets = pd.read_table(neg_path, names=['tweet'], sep ="\n", header=None, nrows=nrows)
    df_neg_tweets['label'] = -1

    df_tweets = pd.concat((df_pos_tweets, df_neg_tweets))
    return df_tweets


def load_test(test_path):
    """
    Load all the test tweets
    Return a list of tuples, the first element is the index of the tweet and
    the second element is the list of words in the tweet
    """
    return pd.read_csv(test_path, sep='^([^,]+),', engine='python', names=['Id', 'tweet'])
