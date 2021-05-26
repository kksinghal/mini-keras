import numpy as np


"""
Linear activation function class
"""
class linear:
    """
    Activate:
    Activate output of neuron using identity function
    ----------
    Parameters:
    Z: n*m dimensional numpy matrix
        n = number of units in layer, m = number of datapoints
        Z = WX + b
    Returns:
    A: n*m dimensional numpy matrix
        n = number of units in layer, m = number of datapoints
        A = linear activated output of neuron
    """
    def activate(self, Z):
        return Z
    
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
        return np.ones(A.shape)