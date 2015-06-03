import os
import cPickle
import gzip
from pylab import plt
import numpy as np

def load_spirals(N, plot=True):
    """
    Loads a 3 - class non-linearly seperable data set consisting of 'spirals'
    Code is adapted from the Spring 2015 Stanford cs231 course: 'Convolutional
    Neural Networks for Visual Recognition'. Link: http://cs231n.github.io/neural-networks-case-study/
    :param N: (int) Number of data points per class
    :param plot: (bool) Plot resulting 2-D dataset
    :return: (X, y) X is a NX2 matrix of inputs, y is an NX1 vector of class labels
    """
    D = 2 # dimensionality
    K = 3 # number of classes
    X = np.zeros((N*K,D)) # data matrix (each row = single example)
    y = np.zeros(N*K, dtype='uint8') # class labels

    for j in xrange(K):
      ix = range(N*j,N*(j+1))
      r = np.linspace(0.0,1,N) # radius
      t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
      X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
      y[ix] = j

    if plot:
        plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)

    return X, y

def load_mnist(path, filename='mnist.pkl.gz', plot=True):
    """
    Loads the MNIST dataset. Downloads the data if it doesn't already exist.
    This code is adapted from the deeplearning.net tutorial on classifying
    MNIST data with Logistic Regression: http://deeplearning.net/tutorial/logreg.html#logreg
    :param path: (str) Path to where data lives or should be downloaded too
    :param filename: (str) name of mnist file to download or load
    :return: train_set, valid_set, test_set
    """
    dataset = '{}/{}'.format(path, filename)
    data_dir, data_file = os.path.split(dataset)

    if data_dir == "" and not os.path.isfile(dataset):
        new_path = os.path.join(os.path.split(__file__)[0], "..", "data", dataset)
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib

        origin = ('http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz')
        print 'Downloading data from {}'.format(origin)
        urllib.urlretrieve(origin, dataset)

    print '... loading data'
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    X_train = train_set[0]
    y_train = train_set[1]
    if plot:
        for k in range(25):
            plt.subplot(5,5,k)
            plt.imshow(np.reshape(X_train[k,:], (28,28)))
            plt.axis('off')
            plt.title(y_train[k])

    return train_set, valid_set, test_set
