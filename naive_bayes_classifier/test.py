import numpy as np
from Gaussian_NB import Gaussian_NB
from Bernoulli_NB import Bernoulli_NB
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd

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
    X.to_csv(os.path.join(directory+"/output.csv"), index=False)

if __name__ == "__main__":
    df = pd.read_csv(path)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # # split dataset and reshape for Gaussian_NB
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
    data = np.concatenate([X_train, y_train.to_numpy().reshape(-1,1)],axis = 1)

    NB = Gaussian_NB(data)
    NB.fit()
    y_result = NB.predict(X_test)
    print(f"Accuracy of Gaussian NB is {sum(y_result==y_test)/len(y_test):.6f}")
    print()

    # print prior probabilities
    NB.print_prior()
    print()

    # print posterior probabilities
    posterior = NB.posterior(X_test)
    print("Posterior probabilities:")
    for i in range(10):
        print(f"Sample {i + 1}:")
        for class_idx, prob in enumerate(posterior[i]):
            print(f"Class {class_idx + 1} = {prob}")
        print()
    
    # generate output csv
    generate_csv(X_test, y_result)

    '''
    # example of iris dataset
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
    data = np.concatenate([X_train,y_train.reshape(-1,1)],axis = 1)

    NB = Gaussian_NB(data)
    NB.fit()
    print(f"Accuracy of Gaussian NB is {sum(NB.predict(X_test)==y_test)/len(y_test)}")

    posterior = NB.posterior(X_test)
    print("posterior probabilities:")
    for i in range(len(X_test)):
        print(f"Sample{i}")
        print(f"class0 = {posterior[i][0]}, class1 = {posterior[i][1]}, class2 = {posterior[i][2]}")
    '''
    '''
    # example of milk quality dataset
    df = pd.read_csv(path)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # convert class labels from str to int
    label_map = {"high": 2, "medium": 1, "low": 0}
    y = y.map(label_map)
    if y.isnull().any():
        raise ValueError("class labels convert error.")

    # split dataset and reshape for Gaussian_NB
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
    data = np.concatenate([X_train, y_train.to_numpy().reshape(-1,1)],axis = 1)

    NB = Gaussian_NB(data)
    NB.fit()
    y_result = NB.predict(X_test)
    print(f"Accuracy of Gaussian NB is {sum(y_result==y_test)/len(y_test):.6f}")

    # print posterior probabilities
    posterior = NB.posterior(X_test)
    print("posterior probabilities for milk:")
    for i in range(10):
        print(f"Sample{i}")
        print(f"Low = {posterior[i][0]}, Medium = {posterior[i][1]}, High = {posterior[i][2]}")

    # convert class labels from int to str
    label_map = {2:"high", 1:"medium", 0:"low"}
    y_result = pd.Series(y_result).map(label_map)
    if y_result.isnull().any():
        raise ValueError("class labels convert error.")
    
    # generate output csv
    generate_csv(X_test, y_result)
    '''