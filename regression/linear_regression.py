import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# dataset path
path = "dataset/regress1_trn.csv"
path2 = "dataset/regress1_tst.csv"

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
        b = b.reshape(-1, 1)
        return b
    
    def predict(self, testX):
        std_testX = (testX.to_numpy() - self.mean) / self.std
        X = np.c_[np.ones((std_testX.shape[0], 1)), std_testX]
        return np.dot(X, self.b).flatten()
    
    def standardize(self, X):
        mean = np.mean(X, axis=0).to_numpy()
        std = np.std(X, axis=0).to_numpy()
        return (X - mean)/std, mean, std
    
    def mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def r2_score(self, y_true, y_pred):
        total_variance = np.sum((y_true - np.mean(y_true)) ** 2)
        residual_variance = np.sum((y_true - y_pred) ** 2)
        return 1 - (residual_variance / total_variance)
    
    def get_coefficients(self):
        if self.b is None:
            raise ValueError("please use fit() first")
        
        intercept = self.b[0]
        coefficients = self.b[1:]
        return intercept, coefficients

if __name__ == "__main__":
    df = pd.read_csv(path)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, y_train = remove_outliers_iqr(X, y)

    Lr = Linear_regression(X_train, y_train)
    Lr.fit()

    # df = pd.read_csv(path2)
    # X_test = df.iloc[:, :]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

    y_pred = Lr.predict(X_test)
    mse = Lr.mean_squared_error(y_test, y_pred)
    r2 = Lr.r2_score(y_test, y_pred)
    
    print(f'Mean Squared Error: {mse}')
    print(f'R² Score: {r2}')

    # print(f"y_test range: {y_test.min()} - {y_test.max()}")
    # print(f"y_pred range: {y_pred.min()} - {y_pred.max()}")

    intercept, coefficients = Lr.get_coefficients()
    
    string = "f(x) = "
    string += f"{np.round(intercept, 4)} "

    for i in range(coefficients.shape[0]):
        if coefficients[i] >= 0:
            string += f"+ {np.round(coefficients[i], 4)}*x^{i + 1} "
        else:
            string += f"- {np.abs(np.round(coefficients[i], 4))}*x^{i + 1} "
    print(string)

    # generate_csv(X_test, pd.Series(y_pred, name="class"))