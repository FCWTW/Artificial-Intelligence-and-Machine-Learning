import numpy as np
from math import pi
import matplotlib.pyplot as plt

class Gaussian_NB():
    def __init__(self, data):
        self.y = data[:, -1].astype(int)        # class labels
        self.data = data
        self.prior = dict()                     # prior probabilities
        self.class_dict = dict()
        self.ms = dict()                        # means and standard deviation
        
    def fit(self):
        for i in np.unique(self.y):
            self.class_dict[i] = self.data[self.data[:, -1]==i, :-1]        # segment corresponding features matrix according to different classes
            self.prior[i] = np.mean(self.data[:, -1]==i)                    # calculate prior probabilities for each class
        for class_value, rows in self.class_dict.items():
            m = np.mean(rows, axis = 0)
            s = np.std(rows, axis=0) + 1e-9             # Prevent standard deviation from being 0
            self.ms[class_value] = np.vstack([m,s])     # Stack the mean and standard deviation of the features for each class into a two-dimensional array
        
    def likelihood(self, test):
        likelihood = []
        for i in np.unique(self.y):
            log_prior = np.log(self.prior[i])
            log_likelihood = -0.5*np.log(2*pi*np.square(self.ms[i][1]))
            log_likelihood = log_likelihood - 0.5*(test - self.ms[i][0])**2/np.square(self.ms[i][1])
            likelihood.append(log_prior+np.sum(log_likelihood, axis = 1))
        return np.array(likelihood).T
    
    def predict(self,test):
        log_likelihood = self.likelihood(test)
        pre = np.argmax(log_likelihood, axis = 1)
        return pre