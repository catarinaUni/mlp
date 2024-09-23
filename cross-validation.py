import numpy as np
import pandas as pd
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
    return W1, b1, W2, b2

def mean_absolute_error(Y_test, Y_pred):
    return np.mean(np.abs(Y_test - Y_pred))

def mean_squared_error(Y_test, Y_pred):
    return np.mean(np.square(Y_test - Y_pred))

def root_mean_squared_error(Y_test, Y_pred):
    return np.sqrt(np.mean(np.square(Y_test - Y_pred)))

def split_data(X, Y, n_splits):
    fold_size = len(X) // n_splits
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    folds = []
    
    for i in range(n_splits):
        test_indices = indices[i * fold_size: (i + 1) * fold_size]
        train_indices = np.concatenate((indices[:i * fold_size], indices[(i + 1) * fold_size:]))
        folds.append((train_indices, test_indices))
    
    return folds

def cross_validate(X, Y, learning_rate, epochs, n_splits=5):
    folds = split_data(X, Y, n_splits)
    losses = []
    maes = []
    mses = []
    rmses = []
    
    for i, (train_index, test_index) in enumerate(folds):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        
        W1, b1, W2, b2 = train(X_train, Y_train, learning_rate, epochs)
        
        _, Y_pred = forward_propagation(X_test, W1, b1, W2, b2)
        loss = np.mean(np.square(Y_pred - Y_test))
        mae = mean_absolute_error(Y_test, Y_pred)
        mse = mean_squared_error(Y_test, Y_pred)
        rmse =  root_mean_squared_error(Y_test, Y_pred)
        
        losses.append(loss)
        maes.append(mae)
        mses.append(mse)
        rmses.append(rmse)
        
        print(f"Loss na dobra {i + 1}: {loss}")
        print(f"MAE na dobra {i + 1}: {mae}")
        print(f"MSE na dobra {i + 1}: {mse}")
        print(f"RMSE na dobra {i + 1}: {rmse}")
    
    mean_loss = np.mean(losses)
    std_loss = np.std(losses)
    mean_mae = np.mean(maes)
    std_mae = np.std(maes)
    mean_mse = np.mean(mses)
    std_mse = np.std(mses)
    mean_rmse = np.mean(rmses)
    std_rmse =  np.std(rmses)

    print(f"\nPerda média: {mean_loss} ± {std_loss}")
    print(f"MAE médio: {mean_mae} ± {std_mae}")
    print(f"MSE médio: {mean_mse} ± {std_mse}")
    print(f"RMSE médio:  {mean_rmse} ± {std_rmse}")
    
    return losses, maes, mses

df = pd.read_csv('WHR2024.csv')
df = df.replace({',': '.'}, regex=True)
df = df.dropna()

X = df.iloc[:, -7:].values.astype(float)
Y = df['Ladder score'].values.reshape(-1, 1).astype(float)

losses, maes, mses = cross_validate(X, Y, learning_rate=0.01, epochs=10000, n_splits=5)

W1, b1, W2, b2 = train(X, Y, 0.01, 10000)
_, Y_pred = forward_propagation(X, W1, b1, W2, b2)

loss = np.mean(np.square(Y_pred - Y))
mae = mean_absolute_error(Y, Y_pred)
mse = mean_squared_error(Y, Y_pred)
rmse =   root_mean_squared_error(Y, Y_pred)

print(f"\nLoss no conjunto completo: {loss}")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.boxplot([losses, maes, mses], labels=['Loss', 'MAE', 'MSE'])
plt.title('Boxplot das Métricas de Desempenho')
plt.ylabel('Valores')
plt.grid()

plt.subplot(1, 2, 2)
plt.hist(Y, bins=30, color='blue', alpha=0.7)
plt.title('Distribuição das Classes (Ladder Score)')
plt.xlabel('Ladder Score')
plt.ylabel('Frequência')

plt.tight_layout()
plt.show()
