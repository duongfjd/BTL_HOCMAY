import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
import pickle

# Load data
data = pd.read_csv('BostonHousing.csv')

print("Checking for missing values...")
print(data.isnull().sum())
data['rm'].fillna(data['rm'].median(), inplace=True)
print(data.isnull().sum())
X = data.drop('medv', axis=1).values
y = data['medv'].values

# Split data into train, validation, and test sets (70:15:15)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define models
lr = LinearRegression()
ridge = Ridge()
mlp = MLPRegressor( 
    hidden_layer_sizes=(50, 50, 50),
    activation='relu',
    solver='lbfgs',
    alpha=10,
    max_iter=500,
    random_state=42
)

stacking = StackingRegressor(estimators=[('lr', lr), ('ridge', ridge), ('mlp', mlp)])

# Train models
print("Training Linear Regression...")
lr.fit(X_train, y_train)

print("Training Ridge Regression...")
ridge.fit(X_train, y_train)

print("Training MLP Neural Network (This may take some time)...")
mlp.fit(X_train, y_train)

print("Training Stacking Regressor...")
stacking.fit(X_train, y_train)

# Save models if necessary (optional)
model_names = ['lr_model.pkl', 'ridge_model.pkl', 'mlp_model.pkl', 'stacking_model.pkl']
models = [lr, ridge, mlp, stacking]

for name, model in zip(model_names, models):
    with open(name, 'wb') as f:
        pickle.dump(model, f)

# Ensure 'static' folder exists for saving plots
if not os.path.exists('static'):
    os.makedirs('static')

# Function to plot prediction errors
def plot_prediction_errors(model, X, y, dataset_type, title, filename):
    y_pred = model.predict(X)

    # Plot errors
    plt.scatter(y, y_pred)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    plt.xlabel('Giá trị thực tế')
    plt.ylabel('Giá trị dự đoán')
    plt.title(f"{title} - {dataset_type}\n Biểu đồ phân tán giá trị thực tế và giá trị dự đoán ")
    plt.grid()

    # Save plot to static folder
    plt.savefig(f'static/{filename}')
    plt.close()

# Plot for each model and dataset
datasets = [('train', X_train, y_train), ('validation', X_val, y_val), ('test', X_test, y_test)]
titles = ['Linear Regression', 'Ridge Regression', 'MLP Neural Network', 'Stacking Model']
filenames = [
    'linear_regression_train.png', 'linear_regression_validation.png', 'linear_regression_test.png',
    'ridge_regression_train.png', 'ridge_regression_validation.png', 'ridge_regression_test.png',
    'mlp_train.png', 'mlp_validation.png', 'mlp_test.png',
    'stacking_train.png', 'stacking_validation.png', 'stacking_test.png'
]

# Ensure filenames match correctly
for i, model in enumerate(models):
    for j, (dataset_type, X, y) in enumerate(datasets):
        filename = filenames[i * 3 + j]  # Tính chỉ số filename dựa trên chỉ số mô hình và tập dữ liệu
        plot_prediction_errors(model, X, y, dataset_type, titles[i], filename)

print("All models trained and plots saved successfully!")
