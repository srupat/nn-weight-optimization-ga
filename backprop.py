import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score

# Load and preprocess the dataset
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_onehot, test_size=0.2, random_state=42)

# Activation functions
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

def softmax(x):
    exp_vals = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

# Initialize weights and biases
np.random.seed(42)
input_dim = 4
hidden_dim = 6
output_dim = 3

W1 = np.random.randn(input_dim, hidden_dim) * 0.1
b1 = np.zeros((1, hidden_dim))

W2 = np.random.randn(hidden_dim, output_dim) * 0.1
b2 = np.zeros((1, output_dim))

# Training parameters
epochs = 200
learning_rate = 0.05

# Training loop
for epoch in range(epochs):
    # Forward pass
    z1 = np.dot(X_train, W1) + b1
    a1 = leaky_relu(z1)
    z2 = np.dot(a1, W2) + b2
    y_pred = softmax(z2)

    # Loss (cross-entropy)
    loss = -np.mean(np.sum(y_train * np.log(y_pred + 1e-8), axis=1))

    # Backpropagation
    error_output = y_pred - y_train
    dW2 = np.dot(a1.T, error_output)
    db2 = np.sum(error_output, axis=0, keepdims=True)

    error_hidden = np.dot(error_output, W2.T) * leaky_relu_derivative(z1)
    dW1 = np.dot(X_train.T, error_hidden)
    db1 = np.sum(error_hidden, axis=0, keepdims=True)

    # Update weights
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

# Evaluate on test set
z1 = np.dot(X_test, W1) + b1
a1 = leaky_relu(z1)
z2 = np.dot(a1, W2) + b2
y_test_pred = softmax(z2)
y_test_labels = np.argmax(y_test_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_true_labels, y_test_labels)
print(f"\nTest Accuracy: {accuracy:.4f}")
