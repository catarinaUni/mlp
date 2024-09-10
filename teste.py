import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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

def train(X_train, Y_train, learning_rate, epochs):
    input_size = X_train.shape[1]
    hidden_size = 10 
    output_size = 1
    W1, b1, W2, b2 = initialize_weights(input_size, hidden_size, output_size)
    for epoch in range(epochs):
        A1, A2 = forward_propagation(X_train, W1, b1, W2, b2)
        backward_propagation(X_train, Y_train, A1, A2, W1, W2, b1, b2, learning_rate)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, loss: {np.mean(np.square(A2 - Y_train))}")
    return W1, b1, W2, b2

df = pd.read_csv('WHR2024.csv')

df = df.replace({',': '.'}, regex=True)
df = df.dropna()

print(df.columns)

X = df.iloc[:, -7:].values
X = X.astype(float)

Y = df['Ladder score'].values.reshape(-1, 1)
Y = Y.astype(float)

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=42)

# Treinar a MLP
W1, b1, W2, b2 = train(X_train, Y_train, 0.01, 10000)

# Avaliar a MLP no conjunto de teste
_, Y_pred = forward_propagation(X_test, W1, b1, W2, b2)
loss = np.mean(np.square(Y_pred - Y_test))
print(f"Loss no conjunto de teste: {loss}")

def mean_absolute_error(Y_test, Y_pred):
    return np.mean(np.abs(Y_test - Y_pred))

def mean_squared_error(Y_test, Y_pred):
    return np.mean(np.square(Y_test - Y_pred))

mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")


plt.figure(figsize=(10, 6))
plt.plot(Y_test, label='Valores reais')
plt.plot(Y_pred, label='Previsões')
plt.xlabel('Índice')
plt.ylabel('Valor')
plt.title('Previsões vs Valores reais')
plt.legend()
plt.show()
