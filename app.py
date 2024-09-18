from flask import Flask, render_template, request
import numpy as np
from predict_model import predict  # Import hàm predict từ predict.py

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_price():
    if request.method == 'POST':
        # Lấy dữ liệu từ form
        crim = float(request.form['crim'])
        zn = float(request.form['zn'])
        indus = float(request.form['indus'])
        chas = int(request.form['chas'])
        nox = float(request.form['nox'])
        rm = float(request.form['rm'])
        age = float(request.form['age'])
        dis = float(request.form['dis'])
        rad = int(request.form['rad'])
        tax = float(request.form['tax'])
        ptratio = float(request.form['ptratio'])
        b = float(request.form['b'])
        lstat = float(request.form['lstat'])
        model_name = request.form['model']
        
        # Tạo danh sách các đặc trưng để dự đoán
        features = [crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]

        # Gọi hàm predict từ predict.py
        prediction, r2, mae, rmse, nse_value, loss_image = predict(model_name, features)

        # Trả về trang index với kết quả dự đoán và các chỉ số khác
        return render_template('index.html', 
                               prediction_text = f'Predicted MEDV: {prediction:.2f}',
                               r2_text = f'R² =  {r2:.2f}',
                               mae_text = f'MAE = {mae:.2f}',
                               rmse_text = f'RMSE = {rmse:.2f}',
                               nse_text = f'NSE = {nse_value:.2f}',
                               loss_image = loss_image,
                               crim=crim, zn=zn, indus=indus, chas=chas, nox=nox,
                               rm=rm, age=age, dis=dis, rad=rad, tax=tax,
                               ptratio=ptratio, b=b, lstat=lstat, model_name=model_name)

if __name__ == "__main__":
    app.run(debug=True)
