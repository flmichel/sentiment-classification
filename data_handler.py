import numpy as np

# split the data
def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to testing
    """
    # number of value
    num_points = len(y)
    # compute the index that split the datas
    split = int(np.floor(num_points * ratio))

    # set the seed to the given value
    np.random.seed(seed)
    # compute random indexes for training and testing
    rand_indexes = np.random.permutation(num_points)
    index_training = rand_indexes[:split]
    index_testing = rand_indexes[split:]

    return x[index_training], y[index_training], x[index_testing], y[index_testing]
