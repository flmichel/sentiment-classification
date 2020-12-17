import numpy as np
import pandas as pd
import torch
from tqdm.notebook import tqdm
from data_processing import add_padding
import gc

# Select the best device available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_batch(tweets_df, batchsize, index):
    '''
    return a minibatch of the tweets dataframe of size 'batchsize' and starting at index 'batchsize' * 'index'
    '''
    start_index = batchsize * index
    batch = tweets_df.iloc[start_index:start_index + batchsize].copy()
    return add_padding(batch)

def remove_batch(batch):
    '''
    Remove the minibatch from the memory
    '''
    del batch
    gc.collect()

def get_number_of_batch(tweets_df, batchsize):
    '''
    Compute the number of minibatch of size 'batchsize' that there is in the dataframe
    '''
    return int(len(tweets_df) / batchsize)

def accuracy(model, tweets_df, batchsize):
    '''
    Compute the accuracy of th model of the given tweets and label in 'tweets_df'.
    '''
    correct_count = 0
    model.eval()
    batch_num = get_number_of_batch(tweets_df, batchsize)
    for i in tqdm(range(batch_num)):
        batch = get_batch(tweets_df, batchsize, i)
        prediction = model(to_device_batch(batch.input_ids), attention_mask=to_device_batch(batch.attention_mask))[0]
        prediction = prediction.argmax(axis=1)
        label = to_device_batch(batch.label)

        remove_batch(batch)
        correct_count += (prediction == label).float().mean()
    return correct_count / batch_num

def to_device_batch(df):
    '''
    Convert the dataframe to a pythorch tensor and transfer a dataframe to the device
    '''
    return torch.tensor(df.to_list()).to(device)


def fit_model(model, train, validation, batchsize, epochs, optimizer, scheduler, save=False, out_path='../models/model1'):
    '''
    Fit the 'model' with training values 'train' and compute the accuracy and loss a each 'epoch'.
    If we choose to save the model it is done at path 'out_path'.
    The 'optimizer' and 'scheduler' are used to get a better result.
    '''
    for epoch in tqdm(range(epochs)):
        total_loss = 0
        print('epoch', epoch)
        model.train()
        for i in tqdm(range(get_number_of_batch(train, batchsize)), desc="Transfer progress"):
            optimizer.zero_grad()
            batch = get_batch(train, batchsize, i)
            loss, pred = model(to_device_batch(batch.input_ids), attention_mask=to_device_batch(batch.attention_mask), labels=to_device_batch(batch.label))[:2]
            remove_batch(batch)
            total_loss += loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            scheduler.step()
        print('loss', total_loss)
        print('accuracy', accuracy(model, validation, batchsize))

    if save:
        model.save_pretrained(out_path)

def change_zero(value):
    '''
    If the value is 0 change it to -1 in order to be in the right format for the submission
    '''
    if value == 0:
        value = -1
    return value

def get_prediction(model, tweets_df, batchsize, out_path):
    '''
    Predict the value of all the tweets in 'tweets_df' using the model 'model'.
    The prediction is save at 'out_path'.
    '''
    predictions = []
    model.eval()
    batch_num = get_number_of_batch(tweets_df, batchsize)
    for i in tqdm(range(batch_num)):
        batch = get_batch(tweets_df, batchsize, i)
        prediction = model(to_device_batch(batch.input_ids), attention_mask=to_device_batch(batch.attention_mask))[0]
        del batch
        prediction = prediction.argmax(axis=1).tolist()
        predictions += prediction

    tweets_df['Prediction'] = predictions
    tweets_df['Prediction'] = tweets_df['Prediction'].apply(lambda prediction: change_zero(prediction))
    tweets_df = tweets_df[['Id', 'Prediction']]
    tweets_df.to_csv(out_path, index=False)
    return tweets_df[['Id', 'Prediction']]
