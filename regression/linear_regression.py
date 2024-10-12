import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# dataset path
path = "dataset/classification_data.csv"

def generate_csv(X, y):
    directory = os.path.dirname(path)
    y_series = pd.Series(y)

    # reset index
    y_series.reset_index(drop=True, inplace=True)
    X.reset_index(drop=True, inplace=True)

    # combine test data and class label
    X.insert(X.shape[1], column="class", value=y_series)
    X.to_csv(os.path.join(directory, "output.csv"), index=False)

class Linear_regression():
    def __init__(self, X, y, alpha=0.1):
        self.X = X
        self.y = y
        self.alpha = alpha  # regularization strength

    def fit(self):
        self.X, self.mean, self.std = self.standardize(self.X)      # standardize
        self.b = self.estimate_coef()
    
    def estimate_coef(self): 
        X = np.c_[np.ones((self.X.shape[0], 1)), self.X]
        I = np.eye(X.shape[1])                     # identity matrix
        I[0, 0] = 0                                # bias term is not regularized

        # b = (X^T X + alpha * I)^(-1) X^T y
        b = np.linalg.inv(X.T @ X + self.alpha * I) @ X.T @ self.y
        return b
    
    def predict(self, testX):
        std_testX = (testX - self.mean) / self.std
        X = np.c_[np.ones((std_testX.shape[0], 1)), std_testX]
        return X @ self.b
    
    def standardize(self, X):
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean)/std, mean, std
    
    def mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def r2_score(self, y_true, y_pred):
        total_variance = np.sum((y_true - np.mean(y_true)) ** 2)
        residual_variance = np.sum((y_true - y_pred) ** 2)
        return 1 - (residual_variance / total_variance)

if __name__ == "__main__":
    df = pd.read_csv(path)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # split dataset and reshape
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
    
    Lr = Linear_regression(X_train, y_train)
    Lr.fit()

    y_pred = Lr.predict(X_test)
    mse = Lr.mean_squared_error(y_test, y_pred)
    r2 = Lr.r2_score(y_test, y_pred)
    
    print(f'Mean Squared Error: {mse}')
    print(f'R² Score: {r2}')

    generate_csv(X_test, pd.Series(y_pred, name="class"))