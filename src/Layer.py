import numpy as np

class Layer:
    def __init__(self, input_size, output_size, activation_type, name):
        #name must be specified
        if not name:
            raise ValueError("Name must be specified")
        self.name = name

        #each column of both represents the w and b of a neuron in the network
        #make an input size by output size array of weights, each weight randomized 
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size) #He initialization
        #make an 1 by output size array of biases
        self.biases = np.random.randn(1, output_size)

        #set activation type based on number (2 types for now: relu and softmax)
        if activation_type == "relu":
            self.activation_type = 1
        elif activation_type == "softmax":
            self.activation_type = 2
        else:
            raise ValueError("Invalid activation type")

    def activation(self, z):
        if self.activation_type == 1:
            #relu
            return np.maximum(0, z)
        else:
            #softmax
            #subtract by max in order to avoid large number issues with exponential
            #output values remain the same as relative proportions are close
            exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
            return exp_z / np.sum(exp_z, axis=1, keepdims=True) #column wise softmax (row by row)
        
    def activation_derivative(self, z):
        if self.activation_type == 1:
            #derivative of relu with respect to z
            return z > 0
        else:
            #not used in training
            #derivative of softmax with respect to z
            softmax = self.activation(z)
            return softmax * (1 - softmax)
    
    def forward(self, A_in, activation_on=True):
        #for the first layer, A_in is a matrix with rows = training examples and cols = input size 
        Z = np.dot(A_in, self.weights) + self.biases #biases are being expanded to add to all training examples, row by row

        if activation_on:
            return self.activation(Z)
        else:
            return Z

