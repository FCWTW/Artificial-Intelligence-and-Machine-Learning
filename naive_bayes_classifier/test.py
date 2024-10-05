import numpy as np
from Gaussian_NB import Gaussian_NB
from Bernoulli_NB import Bernoulli_NB
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd

# file path
# train_path = ''
# test_path = ''

if __name__ == "__main__":
    # read excel
    # df = pd.read_excel(train_path)

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
        print(f"class0={posterior[i][0]}, class1={posterior[i][1]}, class2={posterior[i][2]}")

    # directory = os.path.dirname(train_path)
    # generate the submission file
    # submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
    # submission.head()
    # submission["target"] = np.argmax(classifier.predict(X_test), axis=1)
    # submission.describe()
    # submission.to_csv("submission.csv", index=False)