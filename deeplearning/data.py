import os
import cPickle
import gzip

def load_mnist(path, filename='mnist.pkl.gz'):
    """
    Loads the MNIST dataset. Downloads the data if it doesn't already exist.
    This code is adapted from the deeplearning.net tutorial on classifying
    MNIST data with Logistic Regression: http://deeplearning.net/tutorial/logreg.html#logreg
    :param path: (str) Path to where data lives or should be downloaded too
    :param filename: (str) name of mnist file to download or load
    :return:
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
    return train_set, valid_set, test_set
