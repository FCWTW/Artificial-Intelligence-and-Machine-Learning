import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# dataset path
path = "dataset/regression_data.csv"

def generate_csv(X, y):
    directory = os.path.dirname(path)
    y_series = pd.Series(y)

    # reset index
    y_series.reset_index(drop=True, inplace=True)
    X.reset_index(drop=True, inplace=True)

    # combine test data and class label
    X.insert(X.shape[1], column="class", value=y_series)
    X.to_csv(os.path.join(directory, "output.csv"), index=False)

# filter extreme values
def remove_outliers_iqr(X, y):
    Q1 = np.percentile(y, 25)
    Q3 = np.percentile(y, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    mask = (y >= lower_bound) & (y <= upper_bound)
    return X[mask], y[mask]

class PolynomialRegression:
    def __init__(self, degree, alpha):
        self.degree = degree
        self.alpha = alpha

    def polynomial_features(self, X):
        X_poly = X
        for d in range(2, self.degree + 1):
            X_poly = np.hstack((X_poly, X ** d))
        return X_poly

    def fit(self, X, y):
        X_poly = self.polynomial_features(X)
        X_poly = np.c_[np.ones((X_poly.shape[0], 1)), X_poly]

        I = np.eye(X_poly.shape[1])
        I[0, 0] = 0  # 不正則化偏置項

        # Ridge Regression: b = (X^T X + alpha * I)^(-1) X^T y
        self.b = np.linalg.inv(X_poly.T @ X_poly + self.alpha * I) @ X_poly.T @ y

    def predict(self, X):
        X_poly = self.polynomial_features(X)
        X_poly = np.c_[np.ones((X_poly.shape[0], 1)), X_poly]  # 加入偏置
        return X_poly @ self.b
    
    def get_coefficients(self):
        if self.b is None:
            raise ValueError("please use fit() first")
        
        intercept = self.b[0]
        coefficients = self.b[1:]
        return intercept, coefficients

def manual_mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def r2_score(y_true, y_pred):
        total_variance = np.sum((y_true - np.mean(y_true)) ** 2)
        residual_variance = np.sum((y_true - y_pred) ** 2)
        return 1 - (residual_variance / total_variance)

if __name__ == "__main__":
    # 讀取資料
    path = "dataset/regression_data.csv"
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X, y = remove_outliers_iqr(X, y)

    # 分割資料集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

    # 初始化並訓練多項式回歸模型
    model = PolynomialRegression(degree=3, alpha=0.5)
    model.fit(X_train, y_train)

    # 預測
    y_pred = model.predict(X_test)

    # 計算 MSE
    mse = manual_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    print(f'R² Score: {r2}')

    intercept, coefficients = model.get_coefficients()
    string = "f(x) = "
    string += f"{np.round(intercept, 4)} "

    for i in range(coefficients.shape[0]):
        if coefficients[i] >= 0:
            string += f"+ {np.round(coefficients[i], 4)}*x^{i + 1} "
        else:
            string += f"- {np.abs(np.round(coefficients[i], 4))}*x^{i + 1} "
    print(string)