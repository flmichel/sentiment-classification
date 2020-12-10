import numpy as np
import pandas as pd
import torch
from tqdm.notebook import tqdm
from data_processing import add_padding
import gc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_batch(tweets_df, batchsize, index):
    start_index = batchsize * index
    batch = tweets_df.iloc[start_index:start_index + batchsize].copy()
    return add_padding(batch)

def remove_batch(batch):
    del batch
    gc.collect()

def get_number_of_batch(tweets_df, batchsize):
    return int(len(tweets_df) / batchsize)

def accuracy(model, tweets_df, batchsize):
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
    return torch.tensor(df.to_list()).to(device)


def fit_model(model, train, validation, batchsize, epochs, optimizer, scheduler, save=False, out_path='../models/model1'):
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
    if value == 0:
        value = -1
    return value

def get_prediction(model, tweets_df, batchsize, out_path):
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
