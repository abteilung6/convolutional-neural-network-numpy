import numpy as np
from htw_nn_framework.loss_func import LossCriteria

class NeuralNetwork:
    ''' Creates a neural network from a given layer architecture

    This class is suited for fully connected network and
    convolutional neural network architectures. It connects
    the layers and passes the data from one end to another.
    '''
    def __init__(self, layers, score_func=LossCriteria.softmax):
        ''' Setup a global parameter list and initilize a
            score function that is used for predictions.

        Args:
            layer: neural network architecture based on layer and activation function objects
            score_func: function that is used as classifier on the output
        '''
        self.layers = layers
        self.params = []
        for layer in self.layers:
            self.params.append(layer.params)
        self.score_func = score_func

    def forward(self, X):
        ''' Pass input X through all layers in the network
        '''
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, dout):
        grads = []
        ''' Backprop through the network and keep a list of the gradients
            from each layer.
        '''
        for layer in reversed(self.layers):
            dout, grad = layer.backward(dout)
            grads.append(grad)
        return grads

    def predict(self, X):
        ''' Run a forward pass and use the score function to classify
            the output.
        '''
        X = self.forward(X)
        return np.argmax(self.score_func(X), axis=1)
