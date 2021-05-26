import numpy as np

"""
Attributes:
layers : 1D numpy array containing layers
loss: loss function
learning_rate: float
n_layers: int, number of layers including input layer
loss_history: 1D array, loss after each epoch
"""

class Sequential:
    def __init__(self, layers):
        self.layers = layers
        self.n_layers = len(self.layers)
        self.loss_history = np.array([])
    
    """
    compile
    --------
    Parameters:
    learning_rate: float
    loss: loss functions
    """
    def compile(self, loss, learning_rate=0.1):
        #initialise weights
        for i in range(1, len(self.layers)):
            self.layers[i].W = np.random.randn(self.layers[i].n_units, self.layers[i-1].n_units) * 0.01
            self.layers[i].b = np.random.randn(self.layers[i].n_units, 1) * 0.01
        
        self.learning_rate = learning_rate
        if(loss == "binary_crossentropy"):
            from keras.losses.binary_crossentropy import binary_crossentropy
            self.loss = binary_crossentropy()
            
            
    def forward_propagation(self, X):
        a = X
        self.layers[0].Z = X
        self.layers[0].A = X
        for i in range(1, self.n_layers):
            z = np.dot(self.layers[i].W, a) + self.layers[i].b
            a = self.layers[i].activation.activate(z)
            self.layers[i].Z = z
            self.layers[i].A = a   
        return a
    
    def backward_propagation(self, Y):
        activation_prev_layer = self.layers[self.n_layers - 1].A
        dz = np.multiply(self.loss.derivative(activation_prev_layer, Y), \
                         self.layers[self.n_layers - 1].activation.derivative(activation_prev_layer))
        
        for i in range(self.n_layers-1, 0, -1):
            activation_prev_layer = self.layers[i - 1].A
            dW = np.dot(dz, activation_prev_layer.T)
            self.layers[i].W -= self.learning_rate * dW
            db = dz.sum(axis=1, keepdims=True)
            self.layers[i].b -= self.learning_rate * db
            dz = np.multiply(np.dot(self.layers[i].W.T, dz), \
                             self.layers[i - 1].activation.derivative(activation_prev_layer))
             
    def fit(self, X, Y, epochs=100):
        for i in range(epochs):
            self.forward_propagation(X)
            self.backward_propagation(Y)
            output = self.layers[self.n_layers - 1].A
            self.loss_history = np.append(self.loss_history, self.loss.calculateLoss(output, Y))
            
    def predict(self, X):
        return self.forward_propagation(X)
