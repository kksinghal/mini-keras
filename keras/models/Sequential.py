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
        input_size = self.layers[0].input_size
        #input_size = [int(i) for i in input_size]
        for i in range(self.n_layers):
            input_size = self.layers[i].initialise_weights(input_size)

        if loss == "binary_crossentropy":
            from keras.losses.binary_crossentropy import binary_crossentropy
            self.loss = binary_crossentropy()
        elif loss == "categorical_crossentropy":
            from keras.losses.categorical_crossentropy import categorical_crossentropy
            self.loss = categorical_crossentropy()
            
        if optimizer == 'gradient_descent':
            from keras.optimizers.gradient_descent import gradient_descent
            self.optimizer = gradient_descent()
        elif optimizer == 'gradient_descent_with_momentum':
            from keras.optimizers.gradient_descent_with_momentum import gradient_descent_with_momentum
            self.optimizer = gradient_descent_with_momentum() 
        elif optimizer == 'adam':
            from keras.optimizers.adam import adam
            self.optimizer = adam() 
        
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
                    X_mini = X[..., range(batch_size*j, batch_size*(j+1))]
                    Y_mini = Y[..., range(batch_size*j, batch_size*(j+1))]
                else:
                    X_mini = X[..., range(batch_size*j,n_training_pts)]
                    Y_mini = Y[..., range(batch_size*j, n_training_pts)]
                
                a = X_mini
                for k in range(self.n_layers):
                    a = self.layers[k].forward_propagation(a)#also store activation in layer for case of dense layer
                    
                
                dz = np.multiply(self.loss.derivative(a, Y), \
                         self.layers[self.n_layers - 1].activation.derivative(a))
                
                for k in range(self.n_layers-1, -1, -1):
                    if k>0:
                        activation_prev_layer = self.layers[k-1].A
                        activation_derivative_prev_layer = self.layers[k-1].activation.derivative(activation_prev_layer)
                    else:
                        activation_prev_layer = X_mini
                        activation_derivative_prev_layer = 1
                        
                    # update the weights and return the new dz
                    # A is activation of a layer stored in layer object
                    dz = self.layers[k].backward_propagation(dz, activation_prev_layer \
                                                        , activation_derivative_prev_layer, self.learning_rate) 
                    
                
                    
                
                train_output = a

                self.history["train_loss"] = np.append(self.history["train_loss"]\
                                    , self.loss.calculateLoss(train_output, Y_mini, self.layers))
                if X_val.size != 0:
                    val_output = X_val
                    for i in range(0, self.n_layers):
                        val_output = self.layers[i].forward_propagation(val_output)

                    self.history["val_loss"] = np.append(self.history["val_loss"]\
                                    ,self.loss.calculateLoss(val_output, Y_val, self.layers))


    def predict(self, X):
        a = X
        for i in range(self.n_layers):
            a = self.layers[i].predict(a) 
        return a
