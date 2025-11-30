from .interfaces import Model
from sklearn.metrics import classification_report
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class MLP(Model):

    @staticmethod
    def _init_params(sizes):
        rng = np.random.default_rng()
        # Params is a list of dictionaries with each layer being one element
        params = []
        for i in range(len(sizes) - 1):
            # weights are initialized using size of previous + current layer (output neurons have unique weights for each input neurons)
            W = rng.normal(0, 0.01, (sizes[i], sizes[i+1])).astype(float)
            # biases are initialized using just the current layer (output neurons have one bias used for every input neuron)
            b = np.zeros(sizes[i+1], dtype=float)
            params.append({"W": W, "b": b})
        return params

    @staticmethod
    def _relu(x):
        # Activation function - Rectified Linear Unit (return 0 if x < 0)
        x = np.asarray(x, dtype=float)
        return np.maximum(0, x)

    @staticmethod
    def _relu_grad(x):
        # Derivative of ReLU for backprop
        x = np.asarray(x, dtype=float)
        return (x > 0).astype(float)

    @staticmethod
    def _softmax(x):
        # Another activation function, used for the final output layer as it helps with classification
        x = np.asarray(x, dtype=float)
        x = x - np.max(x, axis=1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=1, keepdims=True)

    @staticmethod
    def _one_hot(y, num_classes):
        oh = np.zeros((len(y), num_classes), dtype=float)
        oh[np.arange(len(y)), y] = 1.0
        return oh

    @staticmethod
    def train(X_train, y_train, params):
        # enforce numpy float arrays (it is boolean in the Pandas Series)
        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train).reshape(-1).astype(int)

        params: dict = params.get("mlp_settings")
        hidden = params.get("hidden_layers", [64])
        lr = params.get("lr", 0.01)
        epochs = params.get("epochs", 50)

        input_dim = X_train.shape[1]
        output_dim = int(np.max(y_train)) + 1

        # Number of layers = input (1) + number of hidden (n) + output (1)
        layer_sizes = [input_dim] + hidden + [output_dim]
        p = MLP._init_params(layer_sizes)
        Y = MLP._one_hot(y_train, output_dim)

        for _ in range(epochs):

            activations = [X_train]
            preacts = []

            # forward
            for li, layer in enumerate(p):
                # Matrix multiplication of layer with the next layer's weights and biases
                z = np.asarray(activations[-1] @ layer["W"] + layer["b"], dtype=float)
                # Save result for backprop
                preacts.append(z)

                if li == len(p) - 1:
                    # Softmax if output layer
                    a = MLP._softmax(z)
                else:
                    # ReLU for hidden layers
                    a = MLP._relu(z)

                activations.append(a)

            # backward
            grads = [None] * len(p)
            dz = activations[-1] - Y

            for i in reversed(range(len(p))):
                a_prev = activations[i]
                layer = p[i]

                grads[i] = {
                    "dW": (a_prev.T @ dz) / len(X_train),
                    "db": dz.mean(axis=0)
                }

                if i > 0:
                    da_prev = dz @ layer["W"].T
                    dz = da_prev * MLP._relu_grad(preacts[i - 1])

            # update
            for i in range(len(p)):
                p[i]["W"] -= lr * grads[i]["dW"]
                p[i]["b"] -= lr * grads[i]["db"]

        return p, plt.figure()

    @staticmethod
    def eval(model, X_test, y_test, params):
        X_test = np.asarray(X_test, dtype=float)
        y_test = np.asarray(y_test).reshape(-1).astype(int)

        a = X_test
        for i, layer in enumerate(model):
            z = np.asarray(a @ layer["W"] + layer["b"], dtype=float)
            if i == len(model) - 1:
                a = MLP._softmax(z)
            else:
                a = MLP._relu(z)

        y_pred = np.argmax(a, axis=1)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Creating classification report as matplotlib plot
        fig, ax = plt.subplots()
        plt.figure(figsize=(8, 6))
        sns.heatmap(pd.DataFrame(report).transpose()[['precision', 'recall', 'f1-score']], annot=True, cmap='viridis', fmt=".2f", ax=ax)
        ax.set_title('Classification Report Heatmap for Decision Tree')

        return report, fig
