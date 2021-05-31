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
    loss: string, name of loss function
    optimizer (default: adam): string, name of optimizer
    """
    def compile(self, loss, optimizer='adam', learning_rate=0.1):
        self.learning_rate = learning_rate
        
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
        
        if loss == "binary_crossentropy":
            from keras.losses.binary_crossentropy import binary_crossentropy
            self.loss = binary_crossentropy()
            
        if optimizer == 'gradient_descent':
            from keras.optimizers.gradient_descent import gradient_descent
            self.optimizer = gradient_descent()
        elif optimizer == 'gradient_descent_with_momentum':
            from keras.optimizers.gradient_descent_with_momentum import gradient_descent_with_momentum
            self.optimizer = gradient_descent_with_momentum() 
        elif optimizer == 'adam':
            from keras.optimizers.adam import adam
            self.optimizer = adam() 
        
            
            
    def forward_propagation(self, X):
        #a = output of current layer
        a = X
        # A: list of activations of each layer
        A = [X] # Activation of 0th layer is input feature vector
        for i in range(1, self.n_layers):
            keep_prob = self.layers[i].keep_prob
            self.layers[i].active_neurons = np.random.choice([1,0], size=(self.layers[i].n_units,1), p=[keep_prob, 1-keep_prob])
            z = np.dot(self.layers[i].W, a) + self.layers[i].b
            a = self.layers[i].activation.activate(z)
            A.append(a)
            a = np.multiply(a, self.layers[i].active_neurons)
        
        return a, A
    
    def compute_gradients(self, Y, A):

        activation_prev_layer = A[self.n_layers - 1]

        dz = np.multiply(self.loss.derivative(activation_prev_layer, Y), \
                         self.layers[self.n_layers - 1].activation.derivative(activation_prev_layer))
        # List of gradients of all W starting from last layer to first
        dW = []
        # List of gradients of all b starting from last layer to first
        db = []
        for i in range(self.n_layers-1, 0, -1):
            activation_prev_layer = A[i - 1]
            dWi = np.dot(dz, activation_prev_layer.T) + self.layers[i].regularizer.calculateDerivative(self.layers[i].W)
            dW.append(dWi)
            
            dbi = dz.sum(axis=1, keepdims=True)
            db.append(dbi)

            dz = np.multiply(np.dot(self.layers[i].W.T, dz), \
                             self.layers[i - 1].activation.derivative(activation_prev_layer))
            
        return dW, db
    
    """
    X: n*m numpy array, n=number of features, m=number of data points
    Y: Ny*m numpy array, Ny= number of output units, m = number of data points
    """
    def fit(self, X, Y, X_val=np.array([]), Y_val=np.array([]), epochs=100, batch_size=None, verbose=1):
        n_training_pts = Y.shape[1]
        if batch_size == None:
            batch_size = n_training_pts
        self.history["train_loss"] = np.array([])
        if X_val.size != 0:
            self.history["val_loss"] = np.array([])
            
        for i in range(epochs):
            from sklearn.utils import shuffle
            X, Y = shuffle(X.T, Y.T, random_state=123)
            X, Y = X.T, Y.T
            n_batches = int(np.ceil([n_training_pts / batch_size])[0])
            for j in range(n_batches):
                if batch_size*(j+1) < n_training_pts:
                    X_mini = X[:, range(batch_size*j, batch_size*(j+1))]
                    Y_mini = Y[:, range(batch_size*j, batch_size*(j+1))]
                else:
                    X_mini = X[:, range(batch_size*j,n_training_pts)]
                    Y_mini = Y[:, range(batch_size*j, n_training_pts)]
                #print(X_mini.shape)
                #a = output of last layer, A = list of activations of each layer
                a, A = self.forward_propagation(X_mini)
                dW, db = self.compute_gradients(Y_mini, A)
                self.optimizer.update_weights(self.layers, dW, db, self.learning_rate)

                
                train_output, _ = self.forward_propagation(X_mini)

                self.history["train_loss"] = np.append(self.history["train_loss"]\
                                    , self.loss.calculateLoss(train_output, Y_mini, self.layers[1:]))
                if X_val.size != 0:
                    val_output, _ = self.forward_propagation(X_val)
                    self.history["val_loss"] = np.append(self.history["val_loss"]\
                                    ,self.loss.calculateLoss(val_output, Y_val, self.layers[1:]))

        
    def predict(self, X):
        a = X
        for i in range(1, self.n_layers):
            z = np.dot(self.layers[i].W, a) + self.layers[i].b
            a = self.layers[i].activation.activate(z)  
        return a
