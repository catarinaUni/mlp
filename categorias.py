import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def preprocess_data(file_path):
    df = pd.read_csv(file_path, delimiter=',')
    df = df.replace({',': '.'}, regex=True)
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    return df

def categorize_data(df, column_name, bins, labels):
    df[column_name + '_category'] = pd.cut(df[column_name], bins=bins, labels=False)  # Use labels=False para obter códigos numéricos
    return df

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
    X = np.array(X, dtype=float)  # Garantir que X é um array NumPy
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
    X_train = np.array(X_train, dtype=float)
    Y_train = np.array(Y_train, dtype=float)
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

def plot_predictions(Y_test, Y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(Y_test, Y_pred, alpha=0.5)
    plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], 'r--', lw=2)
    plt.xlabel('Valores Reais')
    plt.ylabel('Previsões')
    plt.title('Previsões vs Valores Reais')
    plt.show()

# Definir intervalos e rótulos
bins = [0, 3, 5, 7]
labels = ['baixo', 'Médio', 'Alto']

# Processar os dados
df = preprocess_data('WHR2024.csv')
df = categorize_data(df, 'Ladder score', bins, labels)

# Usar apenas as características numéricas para o treinamento
X = df.iloc[:, -7:].values
Y = df['Ladder score_category'].values.reshape(-1, 1)  # Usar categorias numéricas como Y

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=42)

# Treinar a MLP
W1, b1, W2, b2 = train(X_train, Y_train, 0.01, 10000)

# Avaliar a MLP no conjunto de teste
_, Y_pred = forward_propagation(X_test, W1, b1, W2, b2)
loss = np.mean(np.square(Y_pred - Y_test))
print(f"Loss no conjunto de teste: {loss}")

# Plotar previsões vs valores reais
plot_predictions(Y_test, Y_pred)
