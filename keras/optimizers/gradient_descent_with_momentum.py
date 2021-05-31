import numpy as np

"""
Attribute:
V: momentum
    starting with momentum of last layer to first layer
"""
class gradient_descent_with_momentum:
    def __init__(self):
        self.Vw = []
        self.Vb = []
    
    def update_weights(self, layers, dW, db, learning_rate):
        n_layers = len(layers)
        for i in range(n_layers - 1, 0, -1):
            if n_layers-1 -i < len(self.Vw):
                self.Vw[n_layers-1 -i]= ( 0.9 * self.Vw[n_layers-1 -i] ) + ( (1-0.9) * dW[n_layers-1 -i] )
                self.Vb[n_layers-1 -i]= ( 0.9 * self.Vb[n_layers-1 -i] ) + ( (1-0.9) * db[n_layers-1 -i] )
                
            else:
                self.Vw.append(0)
                self.Vb.append(0)
                
            layers[i].W -= learning_rate * np.multiply(self.Vw[n_layers-1 -i], layers[i].active_neurons)
            layers[i].b -= learning_rate * self.Vb[n_layers-1 -i]           