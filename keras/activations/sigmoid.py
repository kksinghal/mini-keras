import numpy as np

"""
Sigmoid activation function class
"""
class sigmoid:
    """
    Activate:
    Activate output of neuron using sigmoid function
    ----------
    Parameters:
    Z: n*m dimensional numpy matrix
        n = number of units in layer, m = number of datapoints
        Z = WX + b
    Returns:
    A: n*m dimensional numpy matrix
        n = number of units in layer, m = number of datapoints
        A = sigmoid activated output of neuron
    """
    def activate(self, Z):
        return 1/(1+np.exp(-1*Z))
    
    
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
        return np.multiply(A, 1-A)