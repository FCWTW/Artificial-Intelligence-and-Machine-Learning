import numpy as np
from math import pi
from numpy.linalg import inv, det

class LDA_Gaussian_NB():
    def __init__(self, data):
        self.y = data[:, -1].astype(int)        # class labels
        self.data = data
        self.prior = dict()                     # prior probabilities
        self.class_dict = dict()
        self.ms = dict()                        # means and standard deviation
        self.shared_cov = None                  # shared covariance matrix
        
    def fit(self):
        # Compute means and prior probabilities for each class
        for i in np.unique(self.y):
            self.class_dict[i] = self.data[self.data[:, -1]==i, :-1]
            self.prior[i] = len(self.class_dict[i]) / self.data.shape[0] 
        
        # Calculate the shared covariance matrix (pooled covariance)
        cov_matrices = []
        for class_value, rows in self.class_dict.items():
            m = np.mean(rows, axis = 0)
            s = np.cov(rows, rowvar=False)       # Calculate the covariance matrix for each class
            cov_matrices.append(s * (len(rows) - 1))  # Weighted by number of samples in the class
            self.ms[class_value] = m             # Store mean for each class
        
        # Shared covariance matrix: sum of weighted class covariances divided by total number of samples
        self.shared_cov = np.sum(cov_matrices, axis=0) / (self.data.shape[0] - len(np.unique(self.y)))
    
    # log likelihood function: log[p(x|C)]
    def likelihood(self, test):
        likelihood = []
        inv_cov = np.linalg.inv(self.shared_cov)  # Inverse of the common covariance matrix
        det_cov = np.linalg.det(self.shared_cov)  # Determinant of the common covariance matrix

        for i in np.unique(self.y):
            log_prior = np.log(self.prior[i])
            diff = test - self.ms[i]
            # Log likelihood using the common covariance matrix
            log_likelihood = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
            log_likelihood += -0.5 * np.log(det_cov) - 0.5 * test.shape[1] * np.log(2 * pi)
            likelihood.append(log_prior + log_likelihood)
        
        return np.array(likelihood).T
    
    # posterior probabilities: p(C|x)
    def posterior(self, test):
        log_likelihood = self.likelihood(test)
        max_log_likelihood = np.max(log_likelihood, axis=1, keepdims=True)
        log_sum = max_log_likelihood + np.log1p(np.sum(np.exp(log_likelihood - max_log_likelihood) - 1, axis=1, keepdims=True))
        log_posterior = log_likelihood - log_sum
        return np.round(np.exp(log_posterior), 10)

    # return the most likely class for each sample in test
    def predict(self, test):
        log_likelihood = self.likelihood(test)
        print("Log likelihood:\n", log_likelihood)
        pre = np.argmax(log_likelihood, axis = 1) + 1
        return pre

    # discriminant function g_i(x) for each class using common covariance matrix
    def discriminant(self, test):
        discriminant_values = {}
        
        cov_inv = inv(self.shared_cov)  # 求协方差矩阵的逆
        log_det_cov = np.log(det(self.shared_cov))  # 协方差矩阵行列式的对数

        for i in np.unique(self.y):
            # 计算 g_i(x)
            g = -0.5 * (np.dot((test - self.ms[i]), cov_inv) * (test - self.ms[i])).sum(axis=1)  # quadratic term
            # g += -0.5 * log_det_cov  # log determinant of covariance matrix
            g += np.log(self.prior[i])  # log prior
            discriminant_values[i] = g

        return discriminant_values
    
    def return_discriminant(self, test):
        num_classes = len(np.unique(self.y))
        result = np.zeros((test.shape[0], num_classes))
        discriminant_values = self.discriminant(test)

        # print(f"Discriminant values keys: {discriminant_values.keys()}")
        for i in range(test.shape[0]):
            num = test.index[i]
            for class_value in discriminant_values.keys():
                result[i][class_value-1] = discriminant_values[class_value][num]
        return result

    # Print discriminant function values
    def print_discriminant(self, test):
        discriminant_values = self.discriminant(test)
        for class_value, values in discriminant_values.items():
            print(f"Class {class_value} discriminant values:")
            print(values)
