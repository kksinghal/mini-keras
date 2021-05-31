import numpy as np

"""
Attribute:
S: starting with momentum of last layer to first layer
V: momentum
    starting with momentum of last layer to first layer
"""
class adam:
    def __init__(self):
        self.Sw = []
        self.Sb = []
        
        self.Vw = []
        self.Vb = []
        
        self.beta = 0.9
    
    def update_weights(self, layers, dW, db, learning_rate):
        epsilon = 1e-8
        n_layers = len(layers)
        for i in range(n_layers - 1, 0, -1):
            if n_layers-1 -i < len(self.Sw):
                self.Sw[n_layers-1 -i]=  self.beta * self.Sw[n_layers-1 -i]  + (1-self.beta) * np.square(dW[n_layers-1 -i]) 
                self.Sb[n_layers-1 -i]= ( self.beta * self.Sb[n_layers-1 -i] ) + ( (1-self.beta) * np.square(db[n_layers-1 -i]) )

                self.Vw[n_layers-1 -i]= ( 0.9 * self.Vw[n_layers-1 -i] ) + ( (1-0.9) * dW[n_layers-1 -i] )
                self.Vb[n_layers-1 -i]= ( 0.9 * self.Vb[n_layers-1 -i] ) + ( (1-0.9) * db[n_layers-1 -i] )
                
            else:
                self.Sw.append(0)
                self.Sb.append(0)
                
                self.Vw.append(0)
                self.Vb.append(0)
            
            
            layers[i].W -= learning_rate * np.multiply(self.Vw[n_layers-1 -i], layers[i].active_neurons)\
                                                    / np.sqrt(self.Sw[n_layers-1 -i] + epsilon)
            layers[i].b -= learning_rate * self.Vb[n_layers-1 -i] / np.sqrt(self.Sb[n_layers-1 -i] + epsilon)
            