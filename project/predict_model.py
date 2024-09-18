import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# NSE calculation
def nse(observed, predicted):
    return 1 - (np.sum((observed - predicted) ** 2) / np.sum((observed - np.mean(observed)) ** 2))

# Load models and data
def load_models_and_data():
    with open('project/lr_model_with_stderr.pkl', 'rb') as f:
        lr, lr_stderr = pickle.load(f)
    with open('project/ridge_model.pkl', 'rb') as f:
        ridge = pickle.load(f)
    with open('project/mlp_model.pkl', 'rb') as f:
        mlp = pickle.load(f)
    with open('project/stacking_model.pkl', 'rb') as f:
        stacking = pickle.load(f)

    data = pd.read_csv('project/BostonHousing.csv')
    imputer = SimpleImputer(strategy='mean')
    X = data.drop('medv', axis=1)
    y = data['medv']
    X = imputer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return lr, ridge, mlp, stacking, X_test, y_test

def predict(model_name, features):
    lr, ridge, mlp, stacking, X_test, y_test = load_models_and_data()
    X_test_single = np.array([features])

    if model_name == 'Linear Regression':
        model = lr
        loss_image = 'linear_regression_loss.png'
    elif model_name == 'Ridge Regression':
        model = ridge
        loss_image = 'ridge_regression_loss.png'
    elif model_name == 'Neural Network':
        model = mlp
        loss_image = 'mlp_training_loss.png'
    elif model_name == 'Stacking':
        model = stacking
        loss_image = 'stacking_loss.png'
    else:
        raise ValueError("Invalid model name provided")

    prediction = model.predict(X_test_single)[0]
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    nse_value = nse(y_test, y_pred)

    return prediction, r2, mae, rmse, nse_value, loss_image
