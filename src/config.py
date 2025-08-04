# config.py

# Reproducibility seed
SEED = 42

# Target column in the Titanic dataset
TARGET = 'Survived'

# File paths
TRAIN_PATH = 'data/train.csv'
TEST_PATH = 'data/test.csv'
SUBMISSION_PATH = 'data/gender_submission.csv'

# Selected features based on EDA
NUMERIC_FEATURES = ['Age', 'Fare', 'SibSp', 'Parch']
CATEGORICAL_FEATURES = ['Sex', 'Embarked', 'Pclass']  # Pclass treated as categorical

# Model hyperparameters
EPOCHS = 1000
LEARNING_RATE = 0.01
BATCH_SIZE = 32
L2_LAMBDA = 0.001  # Regularization strength
ACTIVATION = 'relu'  # or 'sigmoid'
LOSS = 'binary_crossentropy'
OPTIMIZER = 'adam'

# Split percentage for validation
VALIDATION_SPLIT = 0.2


