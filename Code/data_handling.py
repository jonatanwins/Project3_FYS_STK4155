import numpy as np
import tensorflow as tf
from sklearn import datasets
import json

##################################################
##################### Data split, used for MNIST 8
##################################################
def train_test_split(X, Y, percentage, test_index=None):
    """
    X: Feature matrix
    Y: Label vector(size=(n, 1))
    Percentage: How much of the dataset should be used as a test set.
    """

    n = X.shape[0]
    if test_index is None:
        test_index = np.random.choice(n, round(n * percentage), replace=False)
    test_X = X[test_index]
    test_Y = Y[test_index]

    train_X = np.delete(X, test_index, axis=0)
    train_Y = np.delete(Y, test_index, axis=0)

    return train_X, train_Y, test_X, test_Y, test_index


##################################################
##################### Data loading
##################################################
def load_MNIST_8(seed=42, flatten_images=False):

    # download MNIST dataset
    digits = datasets.load_digits()

    # define inputs and labels
    inputs = digits.images
    labels = digits.target

    # RGB images have a depth of 3
    # our images are grayscale so they should have a depth of 1
    X = inputs[:,:,:,np.newaxis].reshape(inputs.shape[0], 8, 8)
    

    # Make feature matrix. We flatten each image and use these 64 values as our input layer
    # Flatten for the non-convolutional models
    if flatten_images:
        X = inputs.reshape((inputs.shape[0], 64))

    # Convert the numbers in y to arrays with 0's and 1's corresponding to classes 
    y = labels.reshape((labels.shape[0]))
    y = np.eye(10)[y]

    # Split in test and train. Fix a seed for this project
    np.random.seed(seed)
    X_train, y_train, X_test, y_test, test_index = train_test_split(X, y, 0.2, test_index=None)

    # Scale pixels to [0,1]
    X_train, X_test = X_train/255, X_test/255

    return X_train, y_train, X_test, y_test


def load_MNIST_28(flatten_images=False):

    # download MNIST dataset
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    assert X_train.shape == (60000, 28, 28)
    assert X_test.shape == (10000, 28, 28)
    assert y_train.shape == (60000,)
    assert y_test.shape == (10000,)

    # Flatten for the non-convolutional models
    if flatten_images:
        X_train = X_train.reshape(60000, 28*28)
        X_test  = X_test.reshape(10000, 28*28)

    y_train = np.eye(10)[y_train.reshape(60000)]
    y_test  = np.eye(10)[y_test.reshape(10000)]

    # Scale pixels to [0,1]
    X_train, X_test = X_train/255, X_test/255


    return X_train, y_train, X_test, y_test


##################################################
##################### Result storage
##################################################
def append_run_to_file(filepath, data):
    """
    Makes file if it does not exist.
    Dumps dictionary into file
    """

    with open(filepath, 'a+') as file:
        file.seek(0)  # Move the cursor to the beginning in case the file is empty

        for item in data:
            json_string = json.dumps(item, indent=2)
            file.write(json_string + '\n')


def load_run_from_file(filepath):
    """
    Reads dictionaries from file into list
    """

    with open(filepath, 'r') as file:
        data = [json.loads(line) for line in file]

    return data