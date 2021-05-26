"""
Attributes:
W: n_units_in_prev_layer * n_units_in_this_layer numpy array 
    Weights of this layer
b: n_units_in_current_layer * 1 numpy array 
    biases of this layer
Z: n_units * m dimensional array
    Z = WX + b for each unit for each data point
A: n_units * m dimensional array
    activation of Z
"""
class Dense:
    """
    Parameters:
    activation: string
        "relu", "sigmoid"
    """
    def __init__(self, n_units, activation="linear"):
        self.n_units = n_units
        if(activation == "relu"):
            from keras.activations.relu import relu
            self.activation = relu()
        elif(activation == "sigmoid"):
            from keras.activations.sigmoid import sigmoid
            self.activation = sigmoid()
        elif(activation == "linear"):
            from keras.activations.linear import linear
            self.activation = linear()