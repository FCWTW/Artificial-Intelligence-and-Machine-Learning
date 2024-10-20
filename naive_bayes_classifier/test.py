import numpy as np
from Gaussian_NB import Gaussian_NB
from LDA_Gauss import LDA_Gaussian_NB
from Diagonal_LDA_Gauss import Diagonal_LDA_Gaussian_NB
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# dataset path
path = "dataset/clsn_trn.csv"
path2 = "dataset/clsn_tst.csv"

def generate_csv(X, g, y):
    directory = os.path.dirname(path)
    g_list = g.tolist()
    print(f"Length of X: {len(X)}, Length of g: {len(g_list)}")

    g_series = pd.Series(g_list)
    y_series = pd.Series(y)
    print(f"Length of X: {len(X)}, Length of g: {len(g_series)}")
    # reset index
    g_series.reset_index(drop=True, inplace=True)
    y_series.reset_index(drop=True, inplace=True)
    X.reset_index(drop=True, inplace=True)

    # combine test data and class label
    result_df = pd.concat([X, g_series, y_series], axis=1)
    result_df.to_csv(os.path.join(directory, "output.csv"), index=False)

def NB_main():
    df = pd.read_csv(path, header=None)
    X_train = df.iloc[:, :-1]
    y_train = df.iloc[:, -1]
    data = np.concatenate([X_train, y_train.to_numpy().reshape(-1,1)],axis = 1)

    NB = Gaussian_NB(data)
    NB.fit()

    df = pd.read_csv(path2, header=None)
    X_test = df.iloc[:, :]
    y_result = NB.predict(X_test)

    h = NB.determinant_covariance()
    print(h)
    g = NB.return_discriminant(X_test)

    # # print prior probabilities
    # NB.print_prior()
    # print()

    # # print posterior probabilities
    # posterior = NB.posterior(X_test)
    # print("Posterior probabilities:")
    # for i in range(10):
    #     print(f"Sample {i + 1}:")
    #     for class_idx, prob in enumerate(posterior[i]):
    #         print(f"Class {class_idx + 1} = {prob}")
    #     print()
    
    # generate output csv
    generate_csv(X_test, g, y_result)

def D_LDA():
    df = pd.read_csv(path, header=None)
    X_train = df.iloc[:, :-1]
    y_train = df.iloc[:, -1]
    data = np.concatenate([X_train, y_train.to_numpy().reshape(-1,1)],axis = 1)

    NBD = Diagonal_LDA_Gaussian_NB(data)
    NBD.fit()

    df = pd.read_csv(path2, header=None)
    X_test = df.iloc[:, :]

    y_result = NBD.predict(X_test)
    g = NBD.return_discriminant(X_test)
    print(X_test.shape)
    print(g.shape)
    generate_csv(X_test, g, y_result)

def LDA():
    df = pd.read_csv(path, header=None)
    X_train = df.iloc[:, :-1]
    y_train = df.iloc[:, -1]
    data = np.concatenate([X_train, y_train.to_numpy().reshape(-1,1)],axis = 1)
    
    NBL = LDA_Gaussian_NB(data)
    NBL.fit()

    df = pd.read_csv(path2, header=None)
    X_test = df.iloc[:, :]

    y_result = NBL.predict(X_test)
    g = NBL.return_discriminant(X_test)
    print(X_test.shape)
    print(g.shape)
    generate_csv(X_test, g, y_result)


if __name__ == "__main__":

    # NB_main()

    # D_LDA()
    
    LDA()