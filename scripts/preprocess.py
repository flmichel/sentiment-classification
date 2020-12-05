#!/usr/bin/env python
# coding: utf-8

# Pre-processing of the Data

# Imports

import wordninja
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Transformation of contractions

contractions = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "cuz" : "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "hes": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "finna": "going to",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "I'd": "i would",
    "I'd've": "i would have",
    "I'll": "i will",
    "I'll've": "i will have",
    "I'm": "i am",
    "I've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "jk": "just kidding",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "pic": "picture",
    "plz": "please",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "shes": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "tho": "though",
    "to've": "to have",
    "wasn't": "was not",
    "wanna": "want to",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "workin": "working",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "u": "you",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
}

def transform_contractions(path):
    transformed = []
    keys = list(contractions.keys())
    with open(path) as f:
        all_lines = f.readlines()
        for tweet in all_lines:
            aux = tweet
            for word in tweet.split(' '):
                if word in keys:
                    aux = aux.replace(word, contractions[word])
            transformed.append(aux)
    return transformed

def transform_contractions_tweet(tweet):
    aux = tweet
    keys = list(contractions.keys())
    for word in tweet.split(' '):
        if word in keys:
            aux = aux.replace(word, contractions[word])
    return aux


# Tag Deletion

def tag_del(path):
    transformed = []
    max_occur = 100
    with open(path) as f:
        all_lines = f.readlines()
        for tweet in all_lines:
            aux = tweet
            if "<user>" in tweet:
                aux = aux.replace("<user>", '', max_occur)
            if "<url>" in tweet:
                aux = aux.replace("<url>", '', max_occur)
            transformed.append(aux)
    return transformed

def tag_del_tweet(tweet):
    aux = tweet
    max_occur = 100
    if "<user>" in tweet:
        aux = aux.replace("<user>", '', max_occur)
    if "<url>" in tweet:
        aux = aux.replace("<url>", '', max_occur)
    return aux


# Hashtag transformation

def transform_hashtag(path):
    transformed = []
    with open(path) as f:
        all_lines = f.readlines()
        for tweet in all_lines:
            aux = tweet
            for word in tweet.split(' '):
                if len(word)>=1:
                    if word[0] == '#':
                        w_aux = "<hashtag> " + ' '.join(wordninja.split(word))
                        aux = aux.replace(word, w_aux)
            transformed.append(aux)
    return transformed

def transform_hashtag_tweet(tweet):
    aux = tweet
    for word in tweet.split(' '):
        if len(word)>=1:
            if word[0] == '#':
                w_aux = "<hashtag> " + ' '.join(wordninja.split(word))
                aux = aux.replace(word, w_aux)
    return aux

# Sentiment Emphasis

# list of common positive/negative words:
positive_words = list(set(open('../data/positive-words.txt', encoding="ISO-8859-1").read().split()))
negative_words = list(set(open('../data/negative-words.txt', encoding="ISO-8859-1").read().split()))

def sentiment_emph(path):
    transformed = []
    with open(path) as f:
        all_lines = f.readlines()
        for tweet in all_lines:
            aux = tweet
            for word in tweet.split(' '):
                if word in positive_words:
                    aux = aux.replace(word, word + ' positive')
                elif word in negative_words:
                    aux = aux.replace(word, word + ' negative')
            transformed.append(aux)
    return transformed

def sentiment_emph_tweet(tweet):
    aux = tweet
    for word in tweet.split(' '):
        if word in positive_words:
            aux = aux.replace(word, word + ' positive')
        elif word in negative_words:
            aux = aux.replace(word, word + ' negative')
    return aux


# Emoji Transformation

happy_emoji = [":)", ":-)", ":d", ";)", ":p", "=)", "=p", "=d", ";p", ";d", ":-d", ";-)", ":-p", "=-)", "=-p", "=-d", ";-p", ";-d", "=]", "^-^", "(:", "^.^", ":D", "=D" ":3"]
love_emoji = ["<3", "x", "xx", "xo", "xoxo", ":*", "xxx"]
sad_emoji = [":(", ":,(", ":'(", ":-(", ":,-(", ":'-(", "D:", ":c", ":C", "=(", "='(", "T.T", "=[", "D=", "Y_Y"]

def transform_emoji(path):
    transformed = []
    with open(path) as f:
        all_lines = f.readlines()
        for tweet in all_lines:
            aux = tweet
            for word in tweet.split(' '):
                if word in happy_emoji:
                    aux = aux.replace(word, 'happy')
                elif word in love_emoji:
                    aux = aux.replace(word, 'love')
                elif word in sad_emoji:
                    aux = aux.replace(word, 'sad')
            transformed.append(aux)
    return transformed

def transform_emoji_tweet(tweet):
    aux = tweet
    for word in tweet.split(' '):
        if word in happy_emoji:
            aux = aux.replace(word, 'happy')
        elif word in love_emoji:
            aux = aux.replace(word, 'love')
        elif word in sad_emoji:
            aux = aux.replace(word, 'sad')
    return aux


# Stopword removal

stop_words = list(set(stopwords.words('english')))

def remove_stopwords(path):
    transformed = []
    with open(path) as f:
        all_lines = f.readlines()
        for tweet in all_lines:
            aux = tweet
            filtered_sentence = [w for w in aux.split() if not w in stop_words]
            transformed.append(" ".join(filtered_sentence))
    return transformed

def remove_stopwords_tweet(tweet):
    filtered_sentence = [word for word in tweet.split() if not word in stop_words]
    return " ".join(filtered_sentence)


# Combine Pre-processing

def preprocess(path):
    transformed = []
    with open(path) as f:
        all_lines = f.readlines()
        for tweet in all_lines:
            aux = transform_contractions_tweet(tweet)
            aux = tag_del_tweet(aux)
            aux = transform_hashtag_tweet(aux)
            aux = remove_stopwords_tweet(aux)
            aux = sentiment_emph_tweet(aux)
            aux = transform_emoji_tweet(aux)
            transformed.append(aux)
    return transformed
