# main.py

import numpy as np
from preprocess import preprocess_training
from model import NeuralNetwork
from evaluate import evaluate_classification
from config import TRAIN_PATH, EPOCHS, VALIDATION_SPLIT
from layers import sigmoid

from sklearn.model_selection import train_test_split

# === Step 1: Preprocess the training data ===
X, y = preprocess_training(TRAIN_PATH)

# === Step 2: Train-test split (for validation) ===
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=VALIDATION_SPLIT, random_state=42, stratify=y
)

# === Step 3: Initialize the neural network ===
# Input dimension = number of features
# Output = 1 (binary classification)
model = NeuralNetwork(input_dim=X.shape[1], hidden_layers=[16, 8], output_dim=1)

# === Step 4: Training loop ===
for epoch in range(1, EPOCHS + 1):
    # Forward pass
    logits = model.forward(X_train)
    predictions = sigmoid(logits)  # Apply sigmoid to get probabilities

    # Compute binary cross-entropy loss
    epsilon = 1e-8  # for numerical stability
    loss = -np.mean(
        y_train * np.log(predictions + epsilon) + (1 - y_train) * np.log(1 - predictions + epsilon)
    )

    # Backpropagation step
    model.backward(predictions, y_train)

    # === Logging every 100 epochs ===
    if epoch % 100 == 0 or epoch == 1:
        print(f"Epoch {epoch}/{EPOCHS} | Loss: {loss:.4f}")

# === Step 5: Validation ===
val_preds = sigmoid(model.forward(X_val))
val_binary = (val_preds > 0.5).astype(int)

# === Step 6: Evaluate model performance ===
results = evaluate_classification(y_val, val_binary)

print("\n📊 Validation Results:")
for key, value in results.items():
    print(f"{key}: {value}")

