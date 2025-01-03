import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Layer import Layer

def main():
    X_train, y_train = load_data("train")

    #normalize X to have values between 0 and 1
    X_train = X_train / 255.0

    #define layers using layer class, store in a neural network array
    neural_network = [
        Layer(name="1", input_size=28*28, output_size=25, activation_type="relu"),
        Layer(name="2", input_size=25, output_size=20, activation_type="relu"),
        Layer(name="4", input_size=20, output_size=10, activation_type="softmax")
    ]

    # train network
    train(neural_network, X_train, y_train, epochs=100, batch_size=1024, alpha=0.1, show_progress=True, show_plots=True)

    # cross_validation_test(neural_network, X_train, y_train, folds=5, epochs=75) # 5 folds = 20% per fold

    # #test network
    X_test, y_test = load_data("test")
    X_train = X_train / 255.0 #normalize
    test(neural_network, X_test, y_test)

    while input("Type enter to show more examples (STOP to end): ") != "STOP":
        show_n_examples(neural_network, X_test, y_test, n_examples=20, k_per_row=10)

def test(neural_network, X, y):
    print("")
    print("Evaluating model on test data...")

    accuracy, cost, num_correct = compute_accuracy_and_cost(neural_network, X, y)
    m = X.shape[0]

    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print(f"Model Cost: {cost:.2f}")
    print(f"#Correct out of test data: {num_correct}/{m}")
    print("")

def show_n_examples(neural_network, X, y, n_examples, k_per_row=5):

    #get n random examples
    random_indices = np.random.permutation(X.shape[0])[:n_examples]
    X_random = X[random_indices]
    y_random = y[random_indices]

    #show images, k per row
    fig, axes = plt.subplots((n_examples // k_per_row) + ((n_examples % k_per_row) != 0), k_per_row)
    axes = axes.flatten()
    for i in range(n_examples):
        ax = axes[i]
        img = X_random[i].reshape(28, 28)
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        
        #pass through network, get argmax for number prediction (reshape(1, -1) returns a (1,784) shape)
        prediction = np.argmax(forward_pass(neural_network, X_random[i].reshape(1, -1)))
        ax.set_title(f"Label: {y_random[i]}\nPred: {prediction}")

    plt.show()

def train(neural_network, X, y, epochs=100, batch_size=1024, alpha=0.1, show_progress=False, show_plots=False):
    m = X.shape[0]

    if show_plots:
        #initialize plot info
        epoch_list, accuracy_list, cost_list = [], [], []  

        plt.ion() #interactive mode
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy [%, Training]")
        
        ax2.set_xlabel("Epoch") 
        ax2.set_ylabel("Cost [Training]")

        accuracy_plot, = ax1.plot(epoch_list, accuracy_list, 'g-', label="Accuracy")
        cost_plot, = ax2.plot(epoch_list, cost_list, 'r-', label="Cost")

        ax1.legend()
        ax2.legend()

        #show plot
        plt.show()

    # (using mini batch gradient descent)
    for epoch in range(epochs):
        #shuffle the batches for each epoch 
        permutation = np.random.permutation(m) #returns a shuffled index array of size m applied to both
        X_shuffled = X[permutation]
        y_shuffled = y[permutation]

        for i in range(0, m, batch_size):
            #get batches
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            #get derivatives
            derivatives = backprop(neural_network, X_batch, y_batch)

            #update params based on derivatives
            update_params(neural_network, derivatives, alpha)
        
        accuracy, cost, num_correct = compute_accuracy_and_cost(neural_network, X, y)

        if show_progress:
            print(f"Epoch {epoch + 1}: Accuracy = {accuracy * 100:.2f}%, Cost = {cost:.2f}")

        if show_plots:
            #update plot data
            epoch_list.append(epoch + 1)
            accuracy_list.append(accuracy * 100)
            cost_list.append(cost)

            accuracy_plot.set_data(epoch_list, accuracy_list)
            cost_plot.set_data(epoch_list, cost_list)

            ax1.relim()
            ax1.autoscale_view()
            ax2.relim()
            ax2.autoscale_view()

            plt.draw()
            plt.pause(0.01)

    if show_plots:
        #turn off plot interactive mode
        plt.ioff()
        plt.show()


# apply a gradient descent step using derivatives
def update_params(neural_network, derivatives, alpha=0.1):
    for i, layer in enumerate(neural_network):
        dJ_dW = derivatives[i][0]
        dJ_db = derivatives[i][1]

        layer.weights -= alpha * dJ_dW
        layer.biases -= alpha * dJ_db

#returns the derivatives needed to apply gradient descent step
def backprop(neural_network, X, y):
    derivatives = []

    #X contains training examples
    m = X.shape[0] 

    #get output from a forward pass
    output, intermediate_values = forward_pass(neural_network, X, save_intermediate_values=True)

    #initialize dJ_dZ 
    dJ_dZ = 0

    #reverse loop for backprop
    for i in range(len(neural_network) - 1, -1, -1):
        layer = neural_network[i]

        if i == len(neural_network) - 1: #output layer
            #better to not include
            dJ_dZ = (output - one_hot_encode(y)) / m #if the output is close to the ideal value (one hot y), then the cost for the digit in that example is lower (subtracting)
        else:  #hidden layers
            next_layer = neural_network[i + 1]
            dJ_dA = np.dot(dJ_dZ, next_layer.weights.T) #uses next layer's dJ_dZ | goes weight by weight per training example, outputs a dot product sum 
            dJ_dZ = dJ_dA * layer.activation_derivative(intermediate_values["Z" + layer.name])  #not a dot product because initial function was element wise

        if i == 0: #first hidden layer
            dJ_dW = np.dot(X.T, dJ_dZ) #use x input (X = dZ_dW)
            dJ_db = np.sum(dJ_dZ, axis=0, keepdims=True)
        else: #other hidden layers
            prev_layer = neural_network[i - 1]
            dJ_dW = np.dot(intermediate_values["A" + prev_layer.name].T, dJ_dZ) #use activations for prev layer
            dJ_db = np.sum(dJ_dZ, axis=0, keepdims=True)

        derivatives.append((dJ_dW, dJ_db))

    #reversed derivative list since we worked backward to get them
    derivatives.reverse()
    return derivatives

def forward_pass(neural_network, X, save_intermediate_values=False):
    if not save_intermediate_values:
        output = X
        for layer in neural_network:
            output = layer.forward(output)
        return output
    else:
        #save all z and a for each layer
        intermediate_values = {}

        output = X
        for layer in neural_network:
            Z = layer.forward(output, activation_on=False)
            A = layer.activation(Z)
            output = A

            intermediate_values.update({
                "Z" + layer.name: Z,
                "A" + layer.name: A
            })

        return output, intermediate_values

def one_hot_encode(y):
    #np.eye is an identity matrix, indexing at position y gives an array with zeros in all pos except for pos y
    one_hot_y = np.eye(np.max(y) + 1)[y] #np.max + 1 = 10 for 10 classes (0-9 digits)
    return one_hot_y

#compute cost/accuracy given params
def compute_accuracy_and_cost(neural_network, X, y):
    # m = number of training examples
    m = X.shape[0]

    #compute a forward pass
    y_hat = forward_pass(neural_network, X)
    
    #multiply by hot encoded y to only consider yth cost calculation
    cost = np.sum(one_hot_encode(y) * -np.log(y_hat + 1e-8)) / m #add 1e-8 to avoid log of zero

    predictions = np.argmax(y_hat, axis=1) #returns indices of max numbers across training examples (column wise) 
    correct = predictions == y

    accuracy = np.mean(correct)
    num_correct = np.sum(correct)

    return accuracy, cost, num_correct

#gives insight into how the architecture (layers, nodes) performs on newly seen data
#returns avg. of 
def cross_validation_test(neural_network, X, y, folds=5, epochs=50, batch_size=1024, alpha=0.1):
    m = X.shape[0]
    fold_size = m // folds #folds are cross validation data subsets
    accuracies = []

    print("")
    print("Cross validation testing...")

    #pick fold subsets, train on non-fold training set, test accuracy on fold set
    for i in range(folds):
        print(f"Training fold {i + 1}...")
        start = i * fold_size
        end = start + fold_size

        X_val = X[start:end]
        y_val = y[start:end]

        X_train = np.concatenate((X[:start], X[end:]), axis=0)
        y_train = np.concatenate((y[:start], y[end:]), axis=0)

        train(neural_network, X_train, y_train, epochs=epochs, batch_size=batch_size, alpha=alpha)

        print(f"Testing fold {i + 1}...")
        accuracy, _, _ = compute_accuracy_and_cost(neural_network, X_val, y_val)
        accuracies.append(accuracy)

    #return avg accuracy for the model architecture
    avg_accuracy = np.mean(accuracies)
    print(f"Cross-validation Accuracy: {avg_accuracy * 100:.2f}%")

# loads x and y training data
def load_data(type):
    if type != "train" and type != "test":
        raise ValueError("Invalid argument for data")

    #read csv, convert to numpy
    data = pd.read_csv(f"data/mnist_{type}.csv").to_numpy()

    #y is first column, X is rest
    X = data[:,1:]
    y = data[:,0]

    # print(X.shape)
    # print(y.shape)

    return X, y


main()