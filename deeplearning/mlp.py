import numpy as np


class MultiLayerPerceptron(object):
    def __init__(self, inputs, num_hidden_units, labels):
        """

        :param inputs: Matrix of shape N X I, where N is the number of observations
        and I is the dimensionality of the input
        :param num_hidden_units: (int) Number of hidden units
        :param labels: An N vector of class labels
        """

        self.inputs = inputs
        self.labels = labels
        self.num_inputs, self.input_dim = np.shape(self.inputs)
        self.T = self.dummy_encode(labels)

        self.hidden_activation_fn = self.relu

        N, I = np.shape(inputs)
        N, K = np.shape(self.T)

        self.W1 = np.random.randn(num_hidden_units, I)
        self.W2 = np.random.randn(K, num_hidden_units)

    def dummy_encode(self, labels):
        T = np.zeros((self.num_inputs, np.max(labels) + 1))
        ind = 0
        for lbl in labels:
            T[ind, lbl] = 1
            ind = ind + 1
        return T

    def hidden_layer(self):
        return self.hidden_activation_fn(np.dot(self.inputs, self.W1.T))

    def output_layer(self, Y):
        return self.softmax(np.dot(Y, self.W2.T))

    def forward_pass(self):
        Z = self.output_layer(self.hidden_layer())
        return Z

    @staticmethod
    def relu(X):
        return np.maximum(0, X)

    @staticmethod
    def softmax(X):
        exp_scores = np.exp(X)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs

    def cross_entropy_loss(self, probs):
        corect_logprobs = -np.log(probs[range(self.num_inputs), self.labels])
        loss = np.sum(corect_logprobs) / self.num_inputs
        return loss

    def compute_gradients(self):
        Y = self.hidden_layer()
        Z = self.output_layer(Y)

        dloss = Z - self.T
        dloss/=self.num_inputs
        dW2 = np.dot(dloss.T, Y)
        dhidden = np.dot(dloss, self.W2)
        dhidden[Y <= 0] = 0
        dW1 = np.dot(dhidden.T, self.inputs)

        return dW1, dW2

    def fit(self, learning_rate, max_iter, iter_print):

        n_iter = 0

        while (n_iter < max_iter):
            prob = self.forward_pass()
            loss = self.cross_entropy_loss(prob)
            dW1, dW2 = self.compute_gradients()
            self.W1 -= learning_rate * dW1
            self.W2 -= learning_rate * dW2

            if n_iter % iter_print == 0:
                print "iteration {}: loss: {}".format(n_iter, loss)
            n_iter+= 1