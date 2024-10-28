import os
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Hàm tính NSE (Nash-Sutcliffe Efficiency)
def nse(observed, predicted):
    return 1 - (np.sum((observed - predicted) ** 2) / np.sum((observed - np.mean(observed)) ** 2))

# Hàm load các mô hình và dữ liệu kiểm tra
def load_models_and_data():
    with open('lr_model.pkl', 'rb') as f:
        lr, lr_stderr = pickle.load(f)
    with open('ridge_model.pkl', 'rb') as f:
        ridge = pickle.load(f)
    with open('mlp_model.pkl', 'rb') as f:
        mlp = pickle.load(f)
    with open('stacking_model.pkl', 'rb') as f:
        stacking = pickle.load(f)
    
    data = pd.read_csv('BostonHousing.csv')
    imputer = SimpleImputer(strategy='mean')
    X = data.drop('medv', axis=1)
    y = data['medv']
    X = imputer.fit_transform(X)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    return lr, lr_stderr, ridge, mlp, stacking,X_train, y_train, X_val, y_val, X_test, y_test

# Hàm dự đoán giá trị và đánh giá
def predict(model_name, features):
    # Load all models and data
    lr, lr_stderr, ridge, mlp, stacking, X_train, y_train, X_val, y_val, X_test, y_test = load_models_and_data()
    
    # Prepare data for evaluation
    X_test_single = np.array(features).reshape(1, -1)
    
    # Khởi tạo giá trị mặc định cho các biến
    prediction = None
    r2 = mae = rmse = nse_value = None
    loss_image = ''

    # Xử lý dự đoán cho từng mô hình
    if model_name == 'Linear Regression':
        prediction = lr.predict(X_test_single)[0]
        y_pred = lr.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        nse_value = nse(y_test, y_pred)
        loss_image = 'linear_regression_test.png'

    elif model_name == 'Ridge Regression':
        prediction = ridge.predict(X_test_single)[0]
        y_pred = ridge.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        nse_value = nse(y_test, y_pred)
        loss_image = 'ridge_regression_test.png'

    elif model_name == 'Neural Network':
        prediction = mlp.predict(X_test_single)[0]
        y_pred = mlp.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        nse_value = nse(y_test, y_pred)
        loss_image = 'mlp_test.png'

    elif model_name == 'Stacking':
        prediction = stacking.predict(X_test_single)[0]
        y_pred = stacking.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        nse_value = nse(y_test, y_pred)
        loss_image = 'stacking_test.png'
        
    # Kiểm tra xem prediction có giá trị không
    if prediction is None:
        raise ValueError("Model name is not recognized.")

    return prediction, r2, mae, rmse, nse_value, loss_image

# Hàm dự đoán giá trị và đánh giá CHO CÁC TẬP TRAIN,VALI,TEST
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    nse_value = nse(y, y_pred)
    return y_pred, r2, mae, rmse, nse_value

# Đường dẫn thư mục static
if not os.path.exists('static'):
    os.makedirs('static')

# Hàm tạo và lưu bảng dưới dạng ảnh PNG
def create_and_save_table(model_name, y_train_pred, y_val_pred, y_test_pred, y_train, y_val, y_test):
    results = {
        'Dataset': ['Train', 'Validation', 'Test'],
        'Prediction': [
            round(y_train_pred.mean(), 3),
            round(y_val_pred.mean(), 3),
            round(y_test_pred.mean(), 3)
        ],
        'R²': [
            round(r2_score(y_train, y_train_pred), 3),
            round(r2_score(y_val, y_val_pred), 3),
            round(r2_score(y_test, y_test_pred), 3)
        ],
        'MAE': [
            round(mean_absolute_error(y_train, y_train_pred), 3),
            round(mean_absolute_error(y_val, y_val_pred), 3),
            round(mean_absolute_error(y_test, y_test_pred), 3)
        ],
        'RMSE': [
            round(np.sqrt(mean_squared_error(y_train, y_train_pred)), 3),
            round(np.sqrt(mean_squared_error(y_val, y_val_pred)), 3),
            round(np.sqrt(mean_squared_error(y_test, y_test_pred)), 3)
        ],
        'NSE': [
            round(nse(y_train, y_train_pred), 3),
            round(nse(y_val, y_val_pred), 3),
            round(nse(y_test, y_test_pred), 3)
        ]
    }

    results_df = pd.DataFrame(results)
    
    # Vẽ bảng và lưu dưới dạng ảnh PNG
    fig, ax = plt.subplots(figsize=(8, 4))  # Kích thước hình ảnh
    ax.axis('tight')
    ax.axis('off')
    table_data = ax.table(cellText=results_df.values, colLabels=results_df.columns, cellLoc='center', loc='center')
    
    # Đặt tiêu đề
    plt.title(f'{model_name} Results', fontsize=16)
    
    # Lưu bảng thành file PNG
    plt.savefig(f'static/{model_name}_results.png', bbox_inches='tight', dpi=300)
    plt.close()

# Load mô hình và dữ liệu
lr, lr_stderr, ridge, mlp, stacking, X_train, y_train, X_val, y_val, X_test, y_test = load_models_and_data()

# Đánh giá mô hình Hồi quy tuyến tính
y_train_pred_lr, r2_train_lr, mae_train_lr, rmse_train_lr, nse_train_lr = evaluate_model(lr, X_train, y_train)
y_val_pred_lr, r2_val_lr, mae_val_lr, rmse_val_lr, nse_val_lr = evaluate_model(lr, X_val, y_val)
y_test_pred_lr, r2_test_lr, mae_test_lr, rmse_test_lr, nse_test_lr = evaluate_model(lr, X_test, y_test)
create_and_save_table("Linear_Regression", y_train_pred_lr, y_val_pred_lr, y_test_pred_lr, y_train, y_val, y_test)

# Đánh giá mô hình Hồi quy Ridge
y_train_pred_ridge, r2_train_ridge, mae_train_ridge, rmse_train_ridge, nse_train_ridge = evaluate_model(ridge, X_train, y_train)
y_val_pred_ridge, r2_val_ridge, mae_val_ridge, rmse_val_ridge, nse_val_ridge = evaluate_model(ridge, X_val, y_val)
y_test_pred_ridge, r2_test_ridge, mae_test_ridge, rmse_test_ridge, nse_test_ridge = evaluate_model(ridge, X_test, y_test)
create_and_save_table("Ridge_Regression", y_train_pred_ridge, y_val_pred_ridge, y_test_pred_ridge, y_train, y_val, y_test)

# Đánh giá mô hình MLP
y_train_pred_mlp, r2_train_mlp, mae_train_mlp, rmse_train_mlp, nse_train_mlp = evaluate_model(mlp, X_train, y_train)
y_val_pred_mlp, r2_val_mlp, mae_val_mlp, rmse_val_mlp, nse_val_mlp = evaluate_model(mlp, X_val, y_val)
y_test_pred_mlp, r2_test_mlp, mae_test_mlp, rmse_test_mlp, nse_test_mlp = evaluate_model(mlp, X_test, y_test)
create_and_save_table("MLP", y_train_pred_mlp, y_val_pred_mlp, y_test_pred_mlp, y_train, y_val, y_test)

# Đánh giá mô hình Stacking
y_train_pred_stacking, r2_train_stacking, mae_train_stacking, rmse_train_stacking, nse_train_stacking = evaluate_model(stacking, X_train, y_train)
y_val_pred_stacking, r2_val_stacking, mae_val_stacking, rmse_val_stacking, nse_val_stacking = evaluate_model(stacking, X_val, y_val)
y_test_pred_stacking, r2_test_stacking, mae_test_stacking, rmse_test_stacking, nse_test_stacking = evaluate_model(stacking, X_test, y_test)
create_and_save_table("Stacking", y_train_pred_stacking, y_val_pred_stacking, y_test_pred_stacking, y_train, y_val, y_test)

print("Tất cả các bảng đã lưu vào thư mục static !")
