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
        self.history = {}
    
    """
    compile
    --------
    Parameters:
    learning_rate: float
    loss: loss functions
    """
    def compile(self, loss, learning_rate=0.1):
        np.random.seed(0)
        #initialise weights
        for i in range(1, len(self.layers)):
            activation_name = self.layers[i].activation_name
            # Check in case someone applies activation on input layer
            if i>0:
                n_units_in_prev_layer = self.layers[i-1].n_units
                
            if activation_name == "relu":
                self.layers[i].W = np.random.randn(self.layers[i].n_units, self.layers[i-1].n_units)* \
                                                                            np.sqrt(2/n_units_in_prev_layer)
                
            elif activation_name == "sigmoid":
                self.layers[i].W = np.random.randn(self.layers[i].n_units, self.layers[i-1].n_units) * \
                                                                            np.sqrt(1/n_units_in_prev_layer)
                
            elif activation_name == "linear":
                self.layers[i].W = np.random.randn(self.layers[i].n_units, self.layers[i-1].n_units)
                
            self.layers[i].b = np.zeros((self.layers[i].n_units, 1))
        
        self.learning_rate = learning_rate
        if(loss == "binary_crossentropy"):
            from keras.losses.binary_crossentropy import binary_crossentropy
            self.loss = binary_crossentropy()
            
            
    def forward_propagation(self, X):
        a = X
        self.layers[0].A = X
        for i in range(1, self.n_layers):
            keep_prob = self.layers[i].keep_prob
            self.layers[i].active_neurons = np.random.choice([1,0], size=(self.layers[i].n_units,1), p=[keep_prob, 1-keep_prob])
            z = np.dot(self.layers[i].W, a) + self.layers[i].b
            a = self.layers[i].activation.activate(z)
            self.layers[i].A = a
            a = np.multiply(a, self.layers[i].active_neurons)
        return a
    
    def backward_propagation(self, Y):
        activation_prev_layer = self.layers[self.n_layers - 1].A
        dz = np.multiply(self.loss.derivative(activation_prev_layer, Y), \
                         self.layers[self.n_layers - 1].activation.derivative(activation_prev_layer))
        
        for i in range(self.n_layers-1, 0, -1):
            activation_prev_layer = self.layers[i - 1].A
            dW = np.dot(dz, activation_prev_layer.T) + self.layers[i].regularizer.calculateDerivative(self.layers[i].W)
            self.layers[i].W -= self.learning_rate * np.multiply(dW, self.layers[i].active_neurons)                    
            db = dz.sum(axis=1, keepdims=True)
            self.layers[i].b -= self.learning_rate * db
            
            dz = np.multiply(np.dot(self.layers[i].W.T, dz), \
                             self.layers[i - 1].activation.derivative(activation_prev_layer))
            
             
    def fit(self, X, Y, X_val=np.array([]), Y_val=np.array([]), epochs=100, verbose=1):
        self.history["train_loss"] = np.array([])
        if X_val.size != 0:
            self.history["val_loss"] = np.array([])
            
        for i in range(epochs):
            self.forward_propagation(X)
            self.backward_propagation(Y)
            train_output = self.layers[self.n_layers - 1].A
            self.history["train_loss"] = np.append(self.history["train_loss"]\
                                , self.loss.calculateLoss(train_output, Y, self.layers[1:]))
            
            if X_val.size != 0:
                val_output = self.forward_propagation(X_val)
                self.history["val_loss"] = np.append(self.history["val_loss"]\
                                ,self.loss.calculateLoss(val_output, Y_val, self.layers[1:]))

        
    def predict(self, X):
        a = X
        self.layers[0].A = X
        for i in range(1, self.n_layers):
            z = np.dot(self.layers[i].W, a) + self.layers[i].b
            a = self.layers[i].activation.activate(z)
            self.layers[i].A = a   
        return a
