import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from data_processing import add_padding
import gc

def get_batch(tweets_df, batchsize, index):
    start_index = batchsize * index
    batch = tweets_df.iloc[start_index:start_index + batchsize].copy()
    return add_padding(batch)

def remove_batch(batch):
    del batch
    gc.collect()

def get_number_of_batch(tweets_df, batchsize):
    return int(len(tweets_df) / batchsize)

def accuracy(model, tweets_df, batchsize, device):
    correct_count = 0
    model.eval()
    batch_num = get_number_of_batch(tweets_df, batchsize)
    for i in tqdm(range(batch_num)):
        batch = get_batch(tweets_df, batchsize, i)
        prediction = model(to_device_batch(batch.input_ids, device), attention_mask=to_device_batch(batch.attention_mask, device))[0]
        prediction = prediction.argmax(axis=1)
        label = to_device_batch(batch.label, device)

        remove_batch(batch)
        correct_count += (prediction == label).float().mean()
    return correct_count / batch_num

def to_device_batch(df, device):
    return torch.tensor(df.to_list()).to(device)


def fit_model(model, train, validation, batchsize, epochs, optimizer, scheduler, device):
    for epoch in tqdm(range(epochs)):
        total_loss = 0
        print('epoch', epoch)
        model.train()
        for i in tqdm(range(get_number_of_batch(train, batchsize)), desc="Transfer progress"):
            optimizer.zero_grad()
            batch = get_batch(train, batchsize, i)
            loss, pred = model(to_device_batch(batch.input_ids, device), attention_mask=to_device_batch(batch.attention_mask, device), labels=to_device_batch(batch.label, device))[:2]
            remove_batch(batch)
            total_loss += loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            scheduler.step()
        print('loss', total_loss)
        print('accuracy', accuracy(model, validation, batchsize, device))

def change_zero(value):
    if value == 0:
        value = -1
    return value

def get_prediction(model, tweets_df, batchsize, out_path, device):
    predictions = []
    model.eval()
    batch_num = get_number_of_batch(tweets_df, batchsize)
    for i in tqdm(range(batch_num)):
        batch = get_batch(tweets_df, batchsize, i)
        prediction = model(to_device_batch(batch.input_ids, device), attention_mask=to_device_batch(batch.attention_mask, device))[0]
        del batch
        prediction = prediction.argmax(axis=1).tolist()
        predictions += prediction

    tweets_df['Prediction'] = predictions
    tweets_df['Prediction'] = tweets_df['Prediction'].apply(lambda prediction: change_zero(prediction))
    tweets_df = tweets_df[['Id', 'Prediction']]
    tweets_df.to_csv(out_path, index=False)
    return tweets_df[['Id', 'Prediction']]
