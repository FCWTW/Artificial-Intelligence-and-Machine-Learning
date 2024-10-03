import numpy as np
from math import pi
import matplotlib.pyplot as plt

class Gaussian_NB():
    def __init__(self, data):
        self.y = data[:,-1]
        self.data = data
        self.summaries = dict()
        self.class_dict = dict()
        self.class_prob = dict()
        
    def fit(self):
        #分割資料集，因為我們需要計算出個特徵對各類別的平均數和標準差
        self.class_dict = {i:self.data[self.data[:,-1]==i,:-1] for i in np.unique(self.y)}
        #計算每一個類別的prior
        self.class_prob = {i:np.mean(self.data[:,-1]==i) for i in np.unique(self.y)}
        summaries = dict()
        for class_value, rows in self.class_dict.items():
            m = np.mean(rows,axis = 0)
            s = np.std(rows,axis = 0)
            summaries[class_value] = np.vstack([m,s])
        self.summaries = summaries
        
    def likelihood(self, test):
        log_likelihood = []
        for i in np.unique(self.y):
            log_class_prob = np.log(self.class_prob[i])
            log_class_likelihood = -0.5*np.log(2*pi*np.square(self.summaries[i][1]))
            log_class_likelihood = log_class_likelihood-0.5*(test-self.summaries[i][0])**2/np.square(self.summaries[i][1])
            log_likelihood.append(log_class_prob+np.sum(log_class_likelihood, axis = 1))
        return np.array(log_likelihood).T
    
    def predict(self,test):
        log_likelihood = self.likelihood(test)
        pre = np.argmax(log_likelihood, axis = 1)
        return pre