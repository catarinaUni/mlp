import numpy as np
import pandas as pd

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def initialize_weights(input_size, hidden_size, output_size):
    np.random.seed(42)
    W1 = np.random.randn(input_size, hidden_size)
    b1 = np.zeros(hidden_size)
    W2 = np.random.randn(hidden_size, output_size)
    b2 = np.zeros(output_size)
    return W1, b1, W2, b2

def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = Z2 
    return A1, A2

def backward_propagation(X, Y, A1, A2, W1, W2, b1, b2, learning_rate):
    m = X.shape[0]
    dZ2 = A2 - Y
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0) / m
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * sigmoid_derivative(A1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0) / m
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

def train(X, Y, learning_rate, epochs):
    input_size = X.shape[1]
    hidden_size = 10 
    output_size = 1
    W1, b1, W2, b2 = initialize_weights(input_size, hidden_size, output_size)
    for epoch in range(epochs):
        A1, A2 = forward_propagation(X, W1, b1, W2, b2)
        backward_propagation(X, Y, A1, A2, W1, W2, b1, b2, learning_rate)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, loss: {np.mean(np.square(A2 - Y))}")
    return W1, b1, W2, b2

df = pd.read_csv('WHR2024.csv')

df = df.replace({',': '.'}, regex=True)

df = df.dropna()

print(df.columns)

X = df.iloc[:, -7:].values
X = X.astype(float)

Y = df['Ladder score'].values.reshape(-1, 1)
Y = Y.astype(float)

W1, b1, W2, b2 = train(X, Y, 0.01, 10000)

new_input = np.array([[1.844, 1.572, 0.695, 0.859, 0.142, 0.546, 2.082],
        [1.908, 1.520, 0.699, 0.823, 0.204, 0.0, 1.881],
        [1.881, 1.617, 0.718, 0.819, 0.258, 0.182, 2.050],
        [1.878, 1.501, 0.724, 0.838, 0.221, 0.524, 1.658],
        [0.629, 0.001, 0.272, 0.0, 0.101, 0.108, 0.677]])
A1, prediction = forward_propagation(new_input, W1, b1, W2, b2)
print("Predição:", prediction)
