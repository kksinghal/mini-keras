import numpy as np

"""
Attributes:
n_filters: int
    Number of filters in the layer
padding: string
    'VALID', 'SAME'
stride: int
W: filter_height * filter_width * n_filters numpy array 
    Weights of the filters
b: n_filters * 1 numpy array 
    biases of each filter
activation: activation object
activation_name: string
    Name of activation function
output_height: int
output_width: int
"""

class Conv2D:
    """
    Parameters:
    activation: string
        "relu", "sigmoid", "linear"
    """
    def __init__(self, n_filters=1, padding="VALID", input_size=None, filter_size=[3,3], stride=1, activation="linear", regularizer="no_regularizer"):
        self.n_filters = n_filters
        self.stride = stride
        self.filter_size= filter_size
        self.input_size=input_size
        
        if padding == "VALID":
            self.padding = [0,0]
        else:: (It is like sliding one matrix over another from right to left, bottom to top)
            self.padding = [np.floor([(self.filter_size[0] - 1)/2])[0], np.floor([(self.filter_size[0] - 1)/2])[0]]
            self.stride = 1
        
        if activation == "relu":
            from keras.activations.relu import relu
            self.activation_name = "relu"
            self.activation = relu()
        
        elif activation == "softmax":
            from keras.activations.softmax import softmax
            self.activation_name = "softmax"
            self.activation = softmax()
        
        elif activation == "sigmoid":
            from keras.activations.sigmoid import sigmoid
            self.activation_name = "sigmoid"
            self.activation = sigmoid()
        elif activation == "linear":
            from keras.activations.linear import linear
            self.activation_name = "linear"
            self.activation = linear()
            
    
    def initialise_weights(self, input_size):
        self.W = np.random.randn(self.filter_size[0], self.filter_size[1], input_size[2]) * 0.01

        self.b = np.zeros((self.n_filters, 1))
        self.output_height = int( np.floor([self.input_size[0] + 2*self.padding[0] - self.filter_size[0]])[0] + 1 )
        self.output_width = int( np.floor([self.input_size[1] + 2*self.padding[1] - self.filter_size[1]])[0] + 1 )
        return [self.output_height, self.output_width, self.n_filters]
    
    
    """
    Attributes:
    activation_prev_layer:  input to this layer
    """
    def forward_propagation(self, activation_prev_layer):
        output = np.zeros((self.output_height, self.output_width, self.n_filters, activation_prev_layer.shape[-1]))
        for i in range(0, self.output_height):
            for j in range(0, self.output_width):
                y = i*self.stride
                x = j*self.stride
                A = activation_prev_layer[y:y+self.filter_size[0], x:x+self.filter_size[1], :]
                B = self.W
                output[i,j,:] = np.sum(np.multiply(A, B), axis=(0,1,2))
        self.A = output
        return output
    
    
    def backward_propagation(self, dz, activation_prev_layer, activation_derivative_prev_layer, learning_rate):
        filter = np.fliplr(np.flipud(self.W))
        padding = (self.filter_size - 1)/2
        
        prev_dz_size = dz.shape
        
        
        top_bottom_padding_size = dz.shape
        top_bottom_padding_size[0] = padding[0]
        top_bottom_padding = np.zeros(top_bottom_padding_size)
        dz = np.vstack((top_bottom_padding, dz, top_bottom_padding))
        
        left_right_padding_size = dz.shape
        left_right_padding_size[1] = padding[1]
        left_right_padding = np.zeros(left_right_padding_size)
        dz = np.hstack((left_right_padding, dz, left_right_padding))
        
        vertical_padd_size = dz.shape
        vertical_padd_size[1] = self.stride -1 
        vertical_padd = np.zeros(vertical_padd_size)
        
        for i in range(padding[1]+1, padding[1] + prev_dz_shape[1]):
            dz = np.hstack((dz[:, :i, ...], vertical_padd, dz[:, i:, ...]))
            
        horizontal_padd_size = dz.shape
        horizontal_padd_size[0] = self.stride -1 
        horizontal_padd = np.zeros(horizontal_padd_size)
        
        for i in range(padding[0]+1, padding[0] + prev_dz_shape[0]):
            dz = np.vstack((dz[:i, ...], horizontal_padd, dz[i:, ...]))
        
        
        dz_height = int( np.floor([dz.shape[0] - self.filter_size[0]])[0] + 1 )
        dz_width = int( np.floor([dz.shape[1] - self.filter_size[1]])[0] + 1 )
        new_dz = np.zeros((dz_height, dz_width))
        for i in range(dz_height):
            for j in range(dz_width):
                A = dz[i:i+self.filter_size[0], j:j+self.filter_size[1], :]
                B = self.W
                new_dz[i,j] = np.sum(np.multiply(A, B), axis=(0,1,2))
                new_dz[i,j] = 
        """
        dW = np.dot(dz, activation_prev_layer.T) + self.regularizer.calculateDerivative(self.W)
        db = dz.sum(axis=1, keepdims=True)
        
        self.W -= learning_rate * dW
        self.b -= learning_rate * db
        
        dz = np.multiply(np.dot(self.W.T, dz), \
                         activation_derivative_prev_layer)
        return dz
        """
        
    def predict(self, X):
        return self.forward_propagation(X)
    