# Project Text 2: Sentiment Classification

This repository contains the code for the second project of EPFL's Machine Learning course.

## Team Members

- Adrien LAYDU
- Manon MICHEL
- François MICHEL

## Goal

The goal of the project is to predict as correctly as possible on a large dataset if tweet messages used to contain a positive `:)` or negative `:(` smiley, by considering only their remaining text.

## Package requirements

To run our code you need to install the following packages:

- Numpy

  ```
  pip install numpy
  ```

- Scikit-Learn
  ```
  pip install sklearn
  ```
- NLTK (The Natural Language Processing Toolkit)
  ```
  pip install nltk
  ```
- Wordninja
  ```
  pip install wordninja
  ```
- Pandas
  ```
  pip install pandas
  ```
- PyTorch
  ```
  pip install torch
  ```
- TQDM
  ```
  pip install tqdm
  ```
- The Transformers repository by Huggingface
  ```
  pip install git+https://github.com/huggingface/transformers.git
  ```
- Regularized logistic regression using GD or SGD.
  ```
  pip install numpy
  ```

## Reproducing this Project

To run our Machine Learning algorithm with the best parameters we found:

- Clone this project
- Download the file `twitter-datasets.zip` on AIcrowd containing the datasets
- Unzip it
- Put it in a folder called `/data` inside the repository
- Download all the packages required
- Run the run.py file with the following command:
  ```
  python3 run.py
  ```

## Files description

Each file can be found in the folder `scripts/`:

- `run.py` contains the code to get a prediction of the sentiments on the test dataset.
- `helper.py` contains functions to import the datasets.
- `preprocess.py` contains functions to preprocess tweets.
- `bert.py` contains functions to train the BERT model and evaluate the accuracy of the model.
- `tweetToVec` contains the function to convert the tweets into vectors using GloVe.
- `data_handler.py` contains the function to split the datas into a training and a validation set.
- `data_processing.py` contains functions to tokenize the tweets using the BERT technique.
- `bert_collab.ipynb` is a notebook used to train the BERT model using Google Colab.
- `Simple_models.ipynb` is a notebook containing all others the models (excluding BERT) we used to perform the sentiments analysis.
