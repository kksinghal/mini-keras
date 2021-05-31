import numpy as np

class gradient_descent:
    def update_weights(self, layers, dW, db, learning_rate):
        n_layers = len(layers)
        for i in range(n_layers - 1, 0, -1):
            layers[i].W -= learning_rate * np.multiply(dW[n_layers-1 -i], layers[i].active_neurons)
            layers[i].b -= learning_rate * db[n_layers-1 -i]            