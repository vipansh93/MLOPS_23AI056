import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Simple neural network implementation from scratch
class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        # Forward propagation
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2
    
    def compute_loss(self, y_true, y_pred):
        # Cross-entropy loss
        m = y_true.shape[0]
        log_probs = -np.log(y_pred[range(m), np.argmax(y_true, axis=1)])
        loss = np.sum(log_probs) / m
        return loss
    
    def backward(self, X, y_true, y_pred, learning_rate=0.1):
        m = X.shape[0]
        
        # Backward propagation
        dz2 = y_pred - y_true
        dW2 = (1/m) * np.dot(self.a1.T, dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        dz1 = np.dot(dz2, self.W2.T) * self.a1 * (1 - self.a1)
        dW1 = (1/m) * np.dot(X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        # Update weights and biases
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def train(self, X, y, epochs=2000, learning_rate=0.01):
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X)
            
            # Compute loss
            loss = self.compute_loss(y, y_pred)
            
            # Add L2 regularization
            reg_loss = loss + 0.001 * (np.sum(self.W1**2) + np.sum(self.W2**2))
            
            # Backward pass
            self.backward(X, y, y_pred, learning_rate)
            
            if epoch % 200 == 0:
                print(f"Epoch {epoch}, Loss: {reg_loss:.4f}")
    
    def predict(self, X):
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)
    
    def accuracy(self, X, y_true):
        y_pred = self.predict(X)
        y_true_labels = np.argmax(y_true, axis=1)
        return np.mean(y_pred == y_true_labels)

# Load and prepare the Iris dataset
print("Loading Iris dataset...")
data = load_iris()
X = data.data
y = data.target.reshape(-1, 1)

# One-hot encode the target variable
encoder = OneHotEncoder()
y_encoded = encoder.fit_transform(y).toarray()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Normalize the features
X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
X_test = (X_test - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)

# Create and train the neural network
print("Creating neural network...")
nn = SimpleNeuralNetwork(input_size=4, hidden_size=10, output_size=3)

print("Training neural network...")
nn.train(X_train, y_train, epochs=2000, learning_rate=0.01)

# Evaluate the model
train_accuracy = nn.accuracy(X_train, y_train)
test_accuracy = nn.accuracy(X_test, y_test)

print(f"\nTraining accuracy: {train_accuracy:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

# Make some predictions
print("\nSample predictions:")
sample_indices = [0, 1, 2]
for i in sample_indices:
    prediction = nn.predict(X_test[i:i+1])
    actual = np.argmax(y_test[i:i+1], axis=1)
    print(f"Sample {i}: Predicted={prediction[0]}, Actual={actual[0]}")
