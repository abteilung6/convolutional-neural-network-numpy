import numpy as np

class LossCriteria:
    ''' Implements diffrent typs of loss and score functions for neural networks

    Todo:
        - Implement init that defines score and loss function
    '''
    def softmax(X):
        ''' Numeric stable calculation of softmax
        '''
        exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))
        return exp_X / np.sum(exp_X, axis=1, keepdims=True)

    def cross_entropy_softmax(X, y):
        ''' Computes loss and prepares dout for backprop

        https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
        '''
        m = y.shape[0]
        p = LossCriteria.softmax(X)
        log_likelihood = -np.log(p[range(m), y])
        loss = np.sum(log_likelihood) / m
        dout = p.copy()
        dout[range(m), y] -= 1
        return loss, dout
