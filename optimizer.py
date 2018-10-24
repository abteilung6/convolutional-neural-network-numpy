import numpy as np


class Optimizer():
    '''
    Optimizer class which implements various optimizing algorithms
    '''

    def get_minibatches(X, y, batch_size):
        ''' Decomposes data set into small subsets (batch)
        '''
        m = X.shape[0]
        batches = []
        for i in range(0, m, batch_size):
            X_batch = X[i:i + batch_size, :, :, :]
            y_batch = y[i:i + batch_size, ]
            batches.append((X_batch, y_batch))
        return batches

    def sgd(network, X_train, y_train, loss_function, batch_size=32, epoch=100, learning_rate=0.001, X_test=None,
            y_test=None, verbose=None):
        '''
        Optimize a given network with stochastical gradient descent
        :param X_train: trainings data
        :param y_train: trainings label
        :param loss_function: loss function
        :param batch_size: size of a single batch
        :param epoch: amount of epochs
        :param learning_rate: the rate which is going to be multiplied with the gradient
        :param X_test: trainings data if you want to test your model in each epcoh
        :param y_test: trainings labels
        :param verbose: if its set it prints out training accuracy and test accuracy
        :return: optimized network
        '''
        minibatches = Optimizer.get_minibatches(X_train, y_train, batch_size)
        for i in range(epoch):
            loss = 0
            if verbose:
                print('Epoch', i + 1)
            for X_mini, y_mini in minibatches:
                # calculate loss and derivation of the last layer
                loss, dout = loss_function(network.forward(X_mini), y_mini)
                # calculate gradients via backpropagation
                grads = network.backward(dout)
                # run vanilla sgd update for all learnable parameters in self.params
                for param, grad in zip(network.params, reversed(grads)):
                    for i in range(len(grad)):
                        param[i] += - learning_rate * grad[i]
            if verbose:
                train_acc = np.mean(y_train == network.predict(X_train))
                test_acc = np.mean(y_test == network.predict(X_test))
                print("Loss = {0} :: Training = {1} :: Test = {2}".format(loss, train_acc, test_acc))
        return network

    def sgd_momentum(network, X_train, y_train, loss_function, batch_size=32, epoch=100, learning_rate=0.001, mu=0.98,
                     nesterov=None, X_test=None, y_test=None, verbose=None):
        '''
        Optimizes a give network with sgd+momentum or sgd+nesterov momentum
        :param X_train: trainings data
        :param y_train: trainings label
        :param loss_function: loss function
        :param batch_size: size of a single batch
        :param epoch: amount of epochs
        :param learning_rate: the rate which is going to be multiplied with the gradient
        :param mu: velocity rate
        :param nesterov: if set nesterov momentum will be performed
        :param X_test: trainings data if you want to test your model in each epcoh
        :param y_test: trainings labels
        :param verbose: if its set it prints out training accuracy and test accuracy
        :return: optimized network
        '''
        minibatches = Optimizer.get_minibatches(X_train, y_train, batch_size)
        velocity = {}
        for i in range(epoch):
            loss = 0
            if verbose:
                print('Epoch', i + 1)
            for X_mini, y_mini in minibatches:
                # calculate loss and derivation of the last layer
                loss, dout = loss_function(network.forward(X_mini), y_mini)
                # calculate gradients via backpropagation
                grads = network.backward(dout)
                # run vanilla sgd update for all learnable parameters in self.params
                i = 0
                for param, grad in zip(network.params, reversed(grads)):
                    if len(grad) == 0:
                        continue
                    # stores temporarily the old velocity
                    temp = velocity.get(i, (0, 0))
                    # calculates the velocity term and
                    # updating the network parameters
                    if (nesterov):
                        velocity[i] = [mu * temp[0] - learning_rate * grad[0], mu * temp[1] + learning_rate * grad[1]]
                        for index in range(len(grad)):
                            if temp == (0, 0):
                                param[index] += (1 + mu) * velocity[i][index]
                            else:
                                param[index] += - mu * temp[index] + (1 + mu) * velocity[i][index]
                    else:
                        velocity[i] = [mu * temp[0] + grad[0], mu * temp[1] + grad[1]]
                        for index in range(len(grad)):
                            param[index] -= learning_rate * velocity[i][index]
                    i += 1
            if verbose:
                train_acc = np.mean(y_train == network.predict(X_train))
                test_acc = np.mean(y_test == network.predict(X_test))
                print("Loss = {0} :: Training = {1} :: Test = {2}".format(loss, train_acc, test_acc))
        return network

    def adagrad(network, X_train, y_train, loss_function, batch_size=32, epoch=100, learning_rate=0.001,
                epsilon=0.0000001, X_test=None, y_test=None, verbose=None):
        '''  Optimize a given network with adagrad
        '''
        minibatches = Optimizer.get_minibatches(X_train, y_train, batch_size)
        grad_squared = {}
        for i in range(epoch):
            loss = 0
            if verbose:
                print('Epoch', i + 1)
            for X_mini, y_mini in minibatches:
                # calculate loss and derivation of the last layer
                loss, dout = loss_function(network.forward(X_mini), y_mini)
                # calculate gradients via backpropagation
                grads = network.backward(dout)

                i = 0
                for param, grad in zip(network.params, reversed(grads)):
                    # if we have no gradient we don not need to optimize
                    if len(grad) == 0:
                        continue

                    # stores temporarily the old squared gradients
                    temp = grad_squared.get(i, (0, 0))

                    # calulates the new squared gradients and adds them with the gold ones
                    # g**2_t_1 = g**2_t + gradient**2
                    grad_squared[i] = [temp[0] + grad[0] ** 2, temp[1] + grad[1] ** 2]

                    # updating our parameters with the squared gradients
                    for index in range(len(grad)):
                        param[index] += - learning_rate * grad[index] / (np.sqrt(grad_squared[i][index]) + epsilon)
                    i += 1
            if verbose:
                train_acc = np.mean(y_train == network.predict(X_train))
                test_acc = np.mean(y_test == network.predict(X_test))
                print("Loss = {0} :: Training = {1} :: Test = {2}".format(loss, train_acc, test_acc))
        return network

    def rmsprop(network, X_train, y_train, loss_function, batch_size=32, epoch=100, learning_rate=0.001, mu=0.9,
                epsilon=0.0000001, X_test=None, y_test=None, verbose=None):
        '''  Optimize a given network with adam
        '''
        minibatches = Optimizer.get_minibatches(X_train, y_train, batch_size)
        grad_squared = {}
        for i in range(epoch):
            loss = 0
            if verbose:
                print('Epoch', i + 1)
            for X_mini, y_mini in minibatches:
                # calculate loss and derivation of the last layer
                loss, dout = loss_function(network.forward(X_mini), y_mini)
                # calculate gradients via backpropagation
                grads = network.backward(dout)
                i = 0
                for param, grad in zip(network.params, reversed(grads)):
                    # if gradient does not exis, we don not need to optimize
                    if len(grad) == 0:
                        continue

                    # stores the old squared gradients
                    temp = grad_squared.get(i, (0, 0))

                    # calculates the new squared gradients and add them with the old one
                    # we also use now our hyperparameter mu now
                    # g**2_t_1 = g**2_t * mu + (1-mu)*gradient**2
                    grad_squared[i] = [mu * temp[0] + (1 - mu) * grad[0] ** 2, mu * temp[1] + (1 - mu) * grad[1] ** 2]

                    # updating our parameter
                    for index in range(len(grad)):
                        param[index] += - learning_rate * grad[index] / (np.sqrt(grad_squared[i][index]) + epsilon)
                    i += 1
            if verbose:
                train_acc = np.mean(y_train == network.predict(X_train))
                test_acc = np.mean(y_test == network.predict(X_test))
                print("Loss = {0} :: Training = {1} :: Test = {2}".format(loss, train_acc, test_acc))
        return network

    def adadelta(network, X_train, y_train, loss_function, batch_size=32, epoch=100, mu=0.95,
                 epsilon=0.0000001, learning_rate=1.0, X_test=None,
                 y_test=None, verbose=None):
        ''' Optimize a given network with adadelta
        '''
        minibatches = Optimizer.get_minibatches(X_train, y_train, batch_size)
        gradient_squared = {}
        delta_squared = {}
        delta = {}
        for i in range(epoch):
            loss = 0
            if verbose:
                print('Epoch', i + 1)
            for X_mini, y_mini in minibatches:
                # calculate loss and derivation of the last layer
                loss, dout = loss_function(network.forward(X_mini), y_mini)
                # calculate gradients via backpropagation
                grads = network.backward(dout)

                i = 0
                # runs adadelta optimization
                for param, grad in zip(network.params, reversed(grads)):
                    if len(grad) == 0:
                        continue
                    # stores the old delta, delta squared and gradient squared values
                    temp1 = delta.get(i, (0, 0))
                    temp2 = delta_squared.get(i, (0, 0))
                    temp3 = gradient_squared.get(i, (0, 0))

                    # calculates the squared gradients
                    gradient_squared[i] = [mu * temp3[0] + (1 - mu) * grad[0] ** 2,
                                           mu * temp3[1] + (1 - mu) * grad[1] ** 2]
                    # calculates the squared delta
                    delta_squared[i] = [mu * temp2[0] + (1 - mu) * temp1[0] ** 2,
                                        mu * temp2[1] + (1 - mu) * temp1[1] ** 2]

                    # calculates the single delta values
                    delta_0 = (np.sqrt(delta_squared[i][0]) + epsilon) * learning_rate * grad[0] / (
                            np.sqrt(gradient_squared[i][0]) + epsilon)
                    delta_1 = (np.sqrt(delta_squared[i][1]) + epsilon) * learning_rate * grad[1] / (
                            np.sqrt(gradient_squared[i][1]) + epsilon)

                    # stores the single delta values into a list
                    delta[i] = [delta_0, delta_1]

                    # updating our parameters with delta
                    for index in range(len(grad)):
                        param[index] -= delta[i][index]
                    i += 1
            if verbose:
                train_acc = np.mean(y_train == network.predict(X_train))
                test_acc = np.mean(y_test == network.predict(X_test))
                print("Loss = {0} :: Training = {1} :: Test = {2}".format(loss, train_acc, test_acc))
        return network

    def adam(network, X_train, y_train, loss_function, batch_size=32, epoch=100, learning_rate=0.001, beta1=0.9,
             beta2=0.999, epsilon=0.0000001, X_test=None, y_test=None, verbose=None):
        '''  Optimize a given network with adam
        '''
        minibatches = Optimizer.get_minibatches(X_train, y_train, batch_size)
        second_moment = {}
        first_moment = {}
        for i in range(epoch):
            loss = 0
            iteration = 0
            if verbose:
                print('Epoch', i + 1)
            for X_mini, y_mini in minibatches:
                iteration += 1
                # calculate loss and derivation of the last layer
                loss, dout = loss_function(network.forward(X_mini), y_mini)
                # calculate gradients via backpropagation
                grads = network.backward(dout)
                # run vanilla sgd update for all learnable parameters in self.params
                i = 0
                for param, grad in zip(network.params, reversed(grads)):
                    temp1 = first_moment.get(i, (0, 0))
                    temp2 = second_moment.get(i, (0, 0))
                    if len(grad) == 0:
                        continue
                    first_moment[i] = [beta1 * temp1[0] + (1 - beta1) * grad[0],
                                       beta1 * temp1[1] + (1 - beta1) * grad[1]]
                    second_moment[i] = [beta2 * temp2[0] + (1 - beta2) * grad[0] ** 2,
                                        beta2 * temp2[1] + (1 - beta2) * grad[1] ** 2]
                    beta1_term = (1 - beta1 ** iteration)
                    beta2_term = (1 - beta2 ** iteration)
                    first_unbias = [first_moment[i][0] / beta1_term, first_moment[i][1] / beta1_term]
                    second_unbias = [second_moment[i][0] / beta2_term, second_moment[i][1] / beta2_term]
                    for index in range(len(grad)):
                        param[index] += - learning_rate * first_unbias[index] / (
                                np.sqrt(second_unbias[index]) + epsilon)
                    i += 1
            if verbose:
                train_acc = np.mean(y_train == network.predict(X_train))
                test_acc = np.mean(y_test == network.predict(X_test))
                print("Loss = {0} :: Training = {1} :: Test = {2}".format(loss, train_acc, test_acc))
        return network
