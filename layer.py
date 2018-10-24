import numpy as np


class Flatten():
    ''' Flatten layer used to reshape inputs into vector representation

    Layer should be used in the forward pass before a dense layer to
    transform a given tensor into a vector.
    '''

    def __init__(self):
        self.params = []

    def forward(self, X):
        ''' Reshapes a n-dim representation into a vector
            by preserving the number of input rows.

        Examples:
            [10000,[1,28,28]] -> [10000,784]
        '''
        self.X_shape = X.shape
        self.out_shape = (self.X_shape[0], -1)
        out = X.reshape(-1).reshape(self.out_shape)
        return out

    def backward(self, dout):
        ''' Restore dimensions before flattening operation
        '''
        out = dout.reshape(self.X_shape)
        return out, []


class FullyConnected():
    ''' Fully connected layer implemtenting linear function hypothesis
        in the forward pass and its derivation in the backward pass.
    '''

    def __init__(self, in_size, out_size):
        ''' Initilize all learning parameters in the layer

        Weights will be initilized with modified Xavier initialization.
        Biases will be initilized with zero.
        '''
        self.W = np.random.randn(in_size, out_size) * np.sqrt(2. / in_size)
        self.b = np.zeros((1, out_size))
        self.params = [self.W, self.b]

    def forward(self, X):
        self.X = X
        out = np.add(np.dot(self.X, self.W), self.b)
        return out

    def backward(self, dout):
        dX = np.dot(dout, self.W.T)
        dW = np.dot(self.X.T, dout)
        db = np.sum(dout, axis=0)
        return dX, [dW, db]


class Conv():
    ''' This layer creates a convolution kernel that
    is convolved with the layer input to produce a tensor of outputs.
    '''

    def __init__(self, input_channels=1, filter_num=32, filter_dim=(3, 3), stride=1, padding=1):
        '''
        Initilize all parameters of our convolution layer

        Weights will be initilized with modified Xavier initialization.
        Biases will be initilized with zero.

        :param input_channels: if input=(3,5,2,1) then input_channels=5
        :param filter_num: amount of filters will be used
        :param filter_dim: the dimension of our filter
        :param stride: stride size
        :param padding: padding size
        '''
        self.input_channels = input_channels
        self.filter_num = filter_num
        self.filter_dim = filter_dim
        self.stride = stride
        self.padding = padding
        self.W = np.random.randn(filter_num, self.input_channels, self.filter_dim[0], self.filter_dim[1]) * np.sqrt(
            2. / self.filter_dim[0])
        self.b = np.zeros((1, filter_num)).T
        self.params = [self.W, self.b]

    def forward(self, X):
        '''
        Calculates the activation map of our conv layer

        :param X: images
        :return: activation map
        '''

        # getting every parameter
        n_filters, d_filter, h_filter, w_filter = self.W.shape
        n_x, d_x, h_x, w_x = X.shape

        # calculating the output size
        size_out = (h_x - h_filter + 2 * self.padding) / self.stride + 1
        size_out = int(size_out)

        # creates a new column vector out of our input with the size of our output image
        # e.g. if input=(3534,1,28,28) and kernel=(32,1,3,3) with stride 1 and padding 0
        # then we know that our new output size has to be (28-3)/1 +1 = 26
        #
        # im2col_indices makes a new matrix with out=(9,3534*28*28)
        x_col = Helper().im2col_indices(X, h_filter, w_filter, padding=self.padding, stride=self.stride)

        # reshaping our filter matrix to a column vector
        # e.g if kernel=(32,1,3,3) then col=(32,3*3)=(32,9) with 64 standing for the
        # amount of kernel used
        w_col = self.W.reshape(n_filters, -1)

        # (32,9) x (9,3534*28*28) = (32,3534*28*28)
        out = np.add(np.dot(w_col, x_col), self.b)

        # reshaping (32,3534*28*28) to (3534,32,28,28)
        #out = out.reshape(n_x, n_filters, size_out, size_out)
        out = out.reshape(n_filters, size_out, size_out, n_x)
        out = out.transpose(3, 0, 1, 2)
        # storing in self for future use in backward path
        self.X = X
        self.x_col = x_col

        return out

    def backward(self, dout):
        '''
        backward path of our convolution layer
        :param dout: gradient
        :return: dx, [dw,db]
        '''
        n_filter, d_filter, h_filter, w_filter = self.W.shape

        # calculates the bias gradient
        db = np.sum(dout, axis=(0, 2, 3))
        db = db.reshape(n_filter, -1)

        # (3534,32,28,28) reshape into (32,3534*28*28)
        dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(n_filter, -1)

        # (32,3534*28*28) * (9,3534*28*28).T = (32,3534*28*28) * (3534*28*28,9) = (32,9)
        dW = np.dot(dout_reshaped, self.x_col.T)

        # reshapes gradient of the weights
        # if the gradient is (32,9) then it will be reshaped into (32,1,3,3)
        dW = dW.reshape(self.W.shape)

        # reshapes the weights to a column vector again
        # e.g if weights = (32,1,3,3) then it will be reshaped into (32,9)
        w_reshape = self.W.reshape(n_filter, -1)

        # (32,9).T * (32,3534*28*28) = (9,32) * (32,3534*28*28) = (9,3534*28*28)
        dX_col = np.dot(w_reshape.T, dout_reshaped)

        # stretches out the column vector back to a image matrix
        # (9,3534*28*28) => (3534, 1, 28, 28)
        dX = Helper().col2im_indices(dX_col, self.X.shape, h_filter, w_filter, padding=self.padding,
                                     stride=self.stride)
        return dX, [dW, db]
    

class Pool():
    ''' Pooling layer
    '''

    def __init__(self, func='max', filter_dim=(2, 2), stride=2):
        '''
        Initialize the pooling layer

        :param func: max,mean or sum - max is set as default
        :param filter_dim: pooling size
        :param stride: stride size
        '''
        self.func = func
        self.filter_dim = filter_dim
        self.stride = stride
        self.params = []

    def forward(self, X):
        '''
        calculates the activation map of our pooling layer
        :param X: input
        :return: activation map
        '''
        # reshapes the images to that the depth channel is 1
        n_x, d_x, h_x, w_x = X.shape
        x_reshaped = X.reshape(n_x * d_x, 1, h_x, w_x)

        # uses helper function to get a column matrix
        x_col = Helper().im2col_indices(x_reshaped, self.filter_dim[0], self.filter_dim[1], padding=0,
                                        stride=self.stride)
        # calculates the output
        if self.func == 'sum':
            out_col = np.sum(x_col, axis=0)
        elif self.func == 'mean':
            out_col = np.mean(x_col, axis=0)
        else:
            out_col = np.max(x_col, axis=0)

        # reshapes the column matrix to a image matrix
        out_size = int((h_x - self.filter_dim[0]) / self.stride + 1)
        #out = out_col.reshape(n_x, d_x, out_size, out_size)
        out = out_col.reshape(out_size, out_size, n_x, d_x)
        out = out.transpose(2, 3, 0, 1)

        # saves input and input col for future use
        self.x_col = x_col
        self.X = X
        return out

    def backward(self, dout):
        '''
        backward path of our pooling layers
        :param dout: gradient
        :return: dX, no gradient on pooling layers
        '''
        # create a column vector out of our gradient
        dout_flat = dout.transpose(2, 3, 0, 1).ravel()

        # replace the values that are not equal to our gradient with 0
        temp_x_col = self.x_col.copy()
        temp_x_col[temp_x_col != dout_flat] = 0

        # since our matrix is still stretched out we need to revert it back to a image matrix
        n_x, d_x, h_x, w_x = self.X.shape
        dX = Helper().col2im_indices(temp_x_col, (n_x * d_x, 1, h_x, w_x), self.filter_dim[0], self.filter_dim[1],
                                     padding=0, stride=self.stride)
        dX = dX.reshape(self.X.shape)
        return dX, []

class Test():
    
    def __init__(self, ):
        None
    
    def forward(self, X):
        return None
    
    def backward(self, dout):
        return None

class Batchnorm():
    ''' Batchnorm layer
    '''

    def __init__(self, X_dim):
        None

    def forward(self, X):
        return None

    def backward(self, dout):
        return None


class Dropout():
    ''' Dropout layer
    '''

    def __init__(self, prob=0.5):
        self.prob = prob
        self.params = []

    def forward(self, X):
        self.X = X
        drop = (np.random.rand(*X.shape) < self.prob) / self.prob
        
        self.drop = drop
        out = X * drop
        return out

    def backward(self, dout):
        dx = dout * self.drop
        return dx, []


class Helper():

    def __init__(self):
        None

    def get_im2col_indices(self, x_shape, field_height, field_width, padding=1, stride=1):
        # First figure out what the size of the output should be
        N, C, H, W = x_shape
        out_size = (H + 2 * padding - field_height) / stride + 1
        out_size = int(out_size)

        i0 = np.repeat(np.arange(field_height), field_width)
        i0 = np.tile(i0, C)
        i1 = stride * np.repeat(np.arange(out_size), out_size)

        j0 = np.tile(np.arange(field_width), field_height * C)
        j1 = stride * np.tile(np.arange(out_size), out_size)

        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)

        k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

        return (k, i, j)

    def im2col_indices(self, x, field_height, field_width, padding=1, stride=1):
        '''
        Helper functions. Image matrix to column matrix
        :param x: input data
        :param field_height: kernel height
        :param field_width: kernel width
        :param padding: padding size
        :param stride: stride size
        :return: column matrixs
        '''
        # Zero-pad the input
        p = padding
        x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

        k, i, j = self.get_im2col_indices(x.shape, field_height, field_width, padding,
                                          stride)

        cols = x_padded[:, k, i, j]
        C = x.shape[1]
        cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
        return cols

    def col2im_indices(self, cols, x_shape, field_height=3, field_width=3, padding=1,
                       stride=1):
        '''
        Helper function. Column matrix to Image matrix
        :param cols: colmn matrix of a input
        :param x_shape: the original input shape
        :param field_height: kernel height
        :param field_width: kernel width
        :param padding: padding size
        :param stride: stride size
        :return: image matrix
        '''
        N, C, H, W = x_shape
        H_padded, W_padded = H + 2 * padding, W + 2 * padding
        x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
        k, i, j = self.get_im2col_indices(x_shape, field_height, field_width, padding,
                                          stride)
        cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
        cols_reshaped = cols_reshaped.transpose(2, 0, 1)
        np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
        if padding == 0:
            return x_padded
        return x_padded[:, :, padding:-padding, padding:-padding]
