#from preprocess import preprocess
from bert import *
from data_processing import *
from helper import *

import numpy as np
import torch
from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split


# Set all parameters :
SEED = 42
NUM_TRAIN_TWEETS = 200000
BATCH_LEN = 40
TRAIN_EPOCHS = 1
TOKEN_LEN = 60  # the maximum is 512
RATIO_TRAIN = 0.98
LEARNING_RATE = 1e-5
EPSILON = 1e-8

# Set all the seeds
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

# Select the best device available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# path for google colab:
# path_to_tweets = '/content/drive/MyDrive/Colab Notebooks/ml-project2/data/twitter-datasets/'

path_to_tweets = '../data/twitter-datasets/'
path_to_models = '../data/models'
model_name = 'model1'


# if you want to train the model uncomment the code below. It might take a lot of time.
pos_path = path_to_tweets + 'train_pos_full.txt'
neg_path = path_to_tweets + 'train_neg_full.txt'
test_path = path_to_tweets + 'test_data.txt'


# ## Prepare the training data

# Load the training data
df_train = load_train(pos_path, neg_path, NUM_TRAIN_TWEETS)

# Preprocess the training data
out_pre_train = path_to_tweets + 'pre_train.csv'
#df_train = preprocess(df_train, out_pre_train)

# Tokenize the training data
out_token_train = path_to_tweets + 'token_train.csv'
df_train = bert_tokenize_train(df_train, out_token_train, max_len=TOKEN_LEN)


# ## Create and train the model

# create the model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased').to(device)

# split the data into a training set and a validation set
train, validation = train_test_split(df_train, train_size=RATIO_TRAIN, random_state=SEED)

# create the optimizer and the scheduler
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=EPSILON)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train)*TRAIN_EPOCHS)

# train the model
fit_model(model, train, validation, BATCH_LEN, TRAIN_EPOCHS, optimizer, scheduler, save=True, out_path=path_to_models+model_name)

# ## Evalutae the test data and create submission

# Load the test data
df_test = load_test(test_path)

# Load the model
model = BertForSequenceClassification.from_pretrained(path_to_models+model_name).to(device)
# Preprocess the test data
# out_pre_test = path_to_tweets + 'pre_test.csv'
# df_test = preprocess(df_test, out_pre_test)

# Tokenize the test data
out_csv_path = path_to_tweets + 'token_test.csv'
df_test = bert_tokenize_test(df_test, out_csv_path, max_len=TOKEN_LEN)

# Compute and save the prediction
out_path = path_to_tweets + 'sub.txt'
prediction = get_prediction(model, df_test, BATCH_LEN, out_path)
