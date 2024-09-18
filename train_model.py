import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.impute import SimpleImputer
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

# Load dataset
data = pd.read_csv('BTL_HOC_MAY/BostonHousing.csv')

# Handle NaN values
imputer = SimpleImputer(strategy='mean')
X = data.drop('medv', axis=1)
y = data['medv']
X = imputer.fit_transform(X)  # Replace NaN with mean value

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
lr = LinearRegression()
ridge = Ridge()
mlp = MLPRegressor(max_iter=1000)

# Train models
lr.fit(X_train, y_train)
ridge.fit(X_train, y_train)
mlp.fit(X_train, y_train)

# Stacking Model
estimators = [('lr', lr), ('ridge', ridge), ('mlp', mlp)]
stacking = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())
stacking.fit(X_train, y_train)

# Save individual models
with open('lr_model.pkl', 'wb') as f:
    pickle.dump(lr, f)
with open('ridge_model.pkl', 'wb') as f:
    pickle.dump(ridge, f)
with open('mlp_model.pkl', 'wb') as f:
    pickle.dump(mlp, f)
with open('stacking_model.pkl', 'wb') as f:
    pickle.dump(stacking, f)

# ---- Create static directory if it does not exist ---- #
if not os.path.exists('static'):
    os.makedirs('static')

# ---- Plot MLP Training Loss ---- #
plt.plot(mlp.loss_curve_)
plt.title('MLP Training Loss Curve')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.grid()
plt.savefig('static/mlp_training_loss.png')  # Save the plot in the static directory
plt.close()  # Close the plot to avoid overlap with the next plot

# ---- Plot Linear Regression Loss Curve ---- #
# Calculate loss manually by evaluating training predictions
lr_losses = []
for i in range(5, len(X_train), 50):  # Bắt đầu từ i = 5 với bước nhảy 50 để giảm số lượng tính toán
    lr_partial_model = LinearRegression()
    lr_partial_model.fit(X_train[:i], y_train[:i])
    predictions = lr_partial_model.predict(X_train[:i])
    loss = np.mean((predictions - y_train[:i]) ** 2)
    lr_losses.append(loss)

plt.plot(range(5, len(X_train), 50), lr_losses)
plt.title('Linear Regression Training Loss Curve')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.grid()
plt.savefig('static/linear_regression_loss.png')
plt.close()

# ---- Plot Ridge Regression Loss Curve ---- #
ridge_losses = []
for i in range(5, len(X_train), 50):  # Bắt đầu từ i = 5 với bước nhảy 50
    ridge_partial_model = Ridge()
    ridge_partial_model.fit(X_train[:i], y_train[:i])
    predictions = ridge_partial_model.predict(X_train[:i])
    loss = np.mean((predictions - y_train[:i]) ** 2)
    ridge_losses.append(loss)

plt.plot(range(5, len(X_train), 50), ridge_losses)
plt.title('Ridge Regression Training Loss Curve')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.grid()
plt.savefig('static/ridge_regression_loss.png')
plt.close()

# ---- Plot Stacking Model Loss Curve ---- #
stacking_losses = []
for i in range(5, len(X_train), 50):  # Bắt đầu từ i = 5 với bước nhảy 50
    stacking_partial_model = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())
    stacking_partial_model.fit(X_train[:i], y_train[:i])
    predictions = stacking_partial_model.predict(X_train[:i])
    loss = np.mean((predictions - y_train[:i]) ** 2)
    stacking_losses.append(loss)

plt.plot(range(5, len(X_train), 50), stacking_losses)
plt.title('Stacking Model Training Loss Curve')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.grid()
plt.savefig('static/stacking_loss.png')
plt.close()

print("ĐÃ CHẠY THÀNH CÔNG!")
