import numpy as np


"""
Rectified Linear Unit activation function class
"""
class relu:
    """
    Activate:
    Activate output of neuron using RELU function
    ----------
    Parameters:
    Z: n*m dimensional numpy matrix
        n = number of units in layer, m = number of datapoints
        Z = WX + b
    Returns:
    A: n*m dimensional numpy matrix
        n = number of units in layer, m = number of datapoints
        A = relu activated output of neuron
    """
    def activate(self, Z):
        A = np.maximum(Z, 0)
        return A
    
    """
    Derivative:
    ------------
    Returns derivative with respect to the Z=WX+b
    
    Parameters:
    A: n*m dimensional numpy matrix
        n = number of units in layer, m = number of datapoints
        Activations of units in layers
    """
    def derivative(self, A):
        return np.where(A>0, 1, 0)