import numpy as np

"""
Softmax activation function class
"""
class softmax:
    """
    Activate:
    Activate output of neuron using softmax function
    ----------
    Parameters:
    Z: n*m dimensional numpy matrix
        n = number of units in layer, m = number of datapoints
        Z = WX + b
    Returns:
    A: n*m dimensional numpy matrix
        n = number of units in layer, m = number of datapoints
        A = softmax activated output of neuron
    """
    def activate(self, Z):
        Z = np.exp(Z)
        A = Z / np.sum(Z, axis=0)
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
        return A - np.multiply(A,A)