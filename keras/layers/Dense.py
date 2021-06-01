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
    def __init__(self, n_units, keep_prob=1, activation="linear", regularizer="no_regularizer"):
        self.n_units = n_units
        self.keep_prob=keep_prob
        
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