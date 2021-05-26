import numpy as np

class binary_crossentropy:
    """
    calculateLoss
    --------------
    Returns average loss over each data point
    Parameters:
    h: m*1 dimensional numpy array, m=number of datapoints
        Predicted value of each data point
    Y: m*1 dimensional numpy array, m=number of datapoints
        True class of each data point
    """
    def calculateLoss(self, h, Y):
        l = (-Y * np.log(h) - (1-Y) * np.log(1-h)).mean()
        return l
    
    """
    derivative
    --------------
    Returns 1*m numpy array
        derivative of loss with respect to h
    Parameters:
    h: m*1 dimensional numpy array, m=number of datapoints
        Predicted value of each data point
    Y: m*1 dimensional numpy array, m=number of datapoints
        True class of each data point
    """
    def derivative(self, h, Y):
        return (-Y/h + (1-Y)/(1-h))/len(Y)