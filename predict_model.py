import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Hàm tính NSE (Nash-Sutcliffe Efficiency)
def nse(observed, predicted):
    return 1 - (np.sum((observed - predicted) ** 2) / np.sum((observed - np.mean(observed)) ** 2))

# Load models and test data
def load_models_and_data():
    with open('BTL_HOC_MAY/lr_model_with_stderr.pkl', 'rb') as f:
        lr, lr_stderr = pickle.load(f)  # Load cả model và stderr
    with open('BTL_HOC_MAY/ridge_model.pkl', 'rb') as f:
        ridge = pickle.load(f)
    with open('BTL_HOC_MAY/mlp_model.pkl', 'rb') as f:
        mlp = pickle.load(f)
    with open('BTL_HOC_MAY/stacking_model.pkl', 'rb') as f:
        stacking = pickle.load(f)
        
    # Load the dataset to compute R² score
    data = pd.read_csv('BTL_HOC_MAY/BostonHousing.csv')
    imputer = SimpleImputer(strategy='mean')
    X = data.drop('medv', axis=1)
    y = data['medv']
    X = imputer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return lr, lr_stderr, ridge, mlp, stacking, X_test, y_test

# Predict function
def predict(model_name, features):
    # Load all models and data
    lr, lr_stderr, ridge, mlp, stacking, X_test, y_test = load_models_and_data()
    
    # Prepare data for evaluation
    X_test_single = np.array([features])
    
    # Xử lý dự đoán cho từng mô hình
    if model_name == 'Linear Regression':
        prediction = lr.predict(X_test_single)[0]
        y_pred = lr.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        nse_value = nse(y_test, y_pred)
        loss_image = 'linear_regression_loss.png'

    elif model_name == 'Ridge Regression':
        prediction = ridge.predict(X_test_single)[0]
        y_pred = ridge.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        nse_value = nse(y_test, y_pred)
        loss_image = 'ridge_regression_loss.png'

    elif model_name == 'Neural Network':
        prediction = mlp.predict(X_test_single)[0]
        y_pred = mlp.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        nse_value = nse(y_test, y_pred)
        loss_image = 'mlp_training_loss.png'

    elif model_name == 'Stacking':
        prediction = stacking.predict(X_test_single)[0]
        y_pred = stacking.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        nse_value = nse(y_test, y_pred)
        loss_image = 'stacking_loss.png'
    
    return prediction, r2, mae, rmse, nse_value, loss_image
