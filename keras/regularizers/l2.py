import numpy as np


"""
L2 Regularizer
---------------
Attributes:
lambd: float
    Regularization parameter
"""

class l2:
    def __init__(self, lambd):
        self.lambd = lambd
        
    def calculateCost(self, W):
        return self.lambd * np.dot(W.T, W).mean() / 2 
        
    def calculateDerivative(self, W):
        return self.lambd* np.sum(W, axis=1, keepdims=True)