import numpy as np

"""
Attributes:
n_units: int
    Number of units in the layer
W: n_units_in_prev_layer * n_units_in_this_layer numpy array 
    Weights of this layer
b: n_units_in_current_layer * 1 numpy array 
    biases of this layer
activation: activation object
activation_name: string
    Name of activation function
A: (temporarily used during backprop): n_units * m dimensional array
    activation of Z
active_neurons: n_units*1 numpy array
    Value of each element is binary. Used for dropout
"""

class Dense:
    """
    Parameters:
    activation: string
        "relu", "sigmoid", "linear"
    regularizer(optional): tuple(string name_of_regularizer, int labmda)
    keep_prob: 0< float <1
        For dropouts, probabilty of keeping a neuron active
    """
    def __init__(self, n_units, keep_prob=1, activation="linear", regularizer="no_regularizer", input_size=None):
        self.n_units = n_units
        self.keep_prob=keep_prob
        self.input_size= input_size
        
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
            
        
        if regularizer[0] == "l2":
            from keras.regularizers.l2 import l2
            self.regularizer = l2(regularizer[1])
        else:
            from keras.regularizers.no_regularizer import no_regularizer
            self.regularizer = no_regularizer()
    
    
    def initialise_weights(self, input_size):
        activation_name = self.activation_name
        # Check in case someone applies activation on input layer
        
        if activation_name == "relu":
            self.W = np.random.randn(self.n_units, input_size[0]) * np.sqrt(2/input_size[0])

        elif activation_name == "sigmoid":
            self.W = np.random.randn(self.n_units, input_size[0]) * np.sqrt(1/input_size[0])

        else:
            self.W = np.random.randn(self.n_units, input_size[0])

        self.b = np.zeros((self.n_units, 1))
        return [self.n_units]
    
    
    
    """
    Returns output of this layer given an input
    Also stores the activation as cache to be used during back prop
    
    Attributes:
    activation_prev_layer:  input to this layer
    """
    def forward_propagation(self, activation_prev_layer):
        keep_prob = self.keep_prob
        self.active_neurons = np.random.choice([1,0], size=(self.n_units,1), p=[keep_prob, 1-keep_prob])
        z = np.dot(self.W, activation_prev_layer) + self.b
        print(z)
        a = self.activation.activate(z)
        self.A = a
        a = np.multiply(a, self.active_neurons)
        return a
    
    
    def backward_propagation(self, dz, activation_prev_layer, activation_derivative_prev_layer, learning_rate):
        dW = np.dot(dz, activation_prev_layer.T) + self.regularizer.calculateDerivative(self.W)
        db = dz.sum(axis=1, keepdims=True)
        self.W -= learning_rate * dW
        self.b -= learning_rate * db
        
        dz = np.multiply(np.dot(self.W.T, dz), \
                         activation_derivative_prev_layer)
        return dz
    
    
    def predict(self, X):
        z = np.dot(self.W, X) + self.b
        a = self.activation.activate(z)  
        return a