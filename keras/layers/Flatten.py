import numpy as np

"""
Attributes:
input_shape: tuple
A: (temporarily used during backprop): n_units * m dimensional array
    activation of Z
"""

class Flatten:
    def __init__(self, keep_prob=1, input_size=None):
        self.keep_prob=keep_prob
        self.input_size= input_size
        from keras.activations.linear import linear
        self.activation_name = "linear"
        self.activation = linear()
    
    
    """
    Returns output of this layer given an input
    Also stores the input_size as cache to be used during back prop
    
    """
    
    def initialise_weights(self, input_size):
        return [np.prod(input_size)]
        
    def forward_propagation(self, activation_prev_layer):
        self.input_shape = activation_prev_layer.shape
        keep_prob = self.keep_prob
        a = activation_prev_layer.reshape(-1,activation_prev_layer.shape[-1])
        self.active_neurons = np.random.choice([1,0], size=(len(a),1), p=[keep_prob, 1-keep_prob])
        a = np.multiply(a, self.active_neurons)
        self.A = a
        return a
    
    
    def backward_propagation(self, dz, activation_prev_layer, activation_derivative_prev_layer, learning_rate):
        dz = dz.reshape(self.input_shape)
        return dz
    
    
    def predict(self, X): 
        return X.reshape(-1,X.shape[-1])