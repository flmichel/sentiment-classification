import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def compute_bert_mask(input_id):
    """
    Compute the attention mask for a given token
    """
    copy = input_id.clone().detach()
    copy[input_id != 0] = 1
    return copy

def bert_tokenize(tweets_df, out_csv_path, max_len):
    """
    Return a datafame with the token of each input tweet and his attention mask.
    """
    tweets_df['input_ids'] = tweets_df['tweet'].apply(lambda tweet: torch.LongTensor(bert_tokenizer.encode(tweet))[:max_len])
    tweets_df['attention_mask'] = tweets_df.apply(lambda row: compute_bert_mask(row.input_ids), axis=1)
    return tweets_df

def bert_tokenize_train(tweets_df, out_csv_path, max_len=512):
    """
    Return a datafame of the train set with the token of each train tweet
    """
    tweets_df = bert_tokenize(tweets_df, out_csv_path, max_len)
    tweets_df = tweets_df[['label', 'input_ids', 'attention_mask']]
    tweets_df.to_csv(out_csv_path, index=False)
    return tweets_df

def bert_tokenize_test(tweets_df, out_csv_path, max_len=512):
    """
    Return a datafame of the test set with the token of each test tweet
    """
    tweets_df = bert_tokenize(tweets_df, out_csv_path, max_len)
    tweets_df = tweets_df[['Id', 'input_ids', 'attention_mask']]
    tweets_df.to_csv(out_csv_path, index=False)
    return tweets_df

def add_padding(tweets_df):
    """
    Add the padding to the input_ids and to the attention_mask
    Return the dataframe with the padding
    """
    tweets_df['input_ids'] = pad_sequence(tweets_df.input_ids.tolist(), batch_first=True)
    tweets_df['attention_mask'] = pad_sequence(tweets_df.attention_mask.tolist(), batch_first=True)

    return tweets_df
