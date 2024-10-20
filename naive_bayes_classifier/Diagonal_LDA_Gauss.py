import numpy as np
from math import pi

class Diagonal_LDA_Gaussian_NB():
    def __init__(self, data):
        self.y = data[:, -1].astype(int)        # class labels
        self.data = data
        self.prior = dict()                     # prior probabilities
        self.class_dict = dict()
        self.ms = dict()                        # means and standard deviation
        self.shared_var = None                  # shared diagonal covariance (variance) matrix
        
    def fit(self):
        # Compute means and prior probabilities for each class
        for i in np.unique(self.y):
            self.class_dict[i] = self.data[self.data[:, -1]==i, :-1]
            self.prior[i] = np.mean(self.data[:, -1]==i) 

        first_key = list(self.class_dict.keys())[0]
        # Calculate the shared diagonal covariance matrix (variances only)
        total_var = np.zeros(self.class_dict[first_key].shape[1])
        for class_value, rows in self.class_dict.items():
            m = np.mean(rows, axis = 0)
            s = np.var(rows, axis=0)  # Calculate the variance for each feature (diagonal elements only)
            total_var += s * (len(rows) - 1)  # Weighted by number of samples in the class
            self.ms[class_value] = m  # Store mean for each class
        
        # Average variance (shared diagonal matrix): sum of weighted variances divided by total number of samples
        self.shared_var = total_var / (self.data.shape[0] - len(np.unique(self.y)))
    
    # log likelihood function: log[p(x|C)] using diagonal covariance matrix
    def likelihood(self, test):
        likelihood = []
        for i in np.unique(self.y):
            log_prior = np.log(self.prior[i])
            diff = test - self.ms[i]
            # Log likelihood using the common diagonal variance matrix
            log_likelihood = -0.5 * np.sum((diff**2) / self.shared_var, axis=1)
            log_likelihood += -0.5 * np.log(2 * pi * self.shared_var).sum()  # Log determinant of diagonal covariance
            likelihood.append(log_prior + log_likelihood)
        
        return np.array(likelihood).T
    
    # posterior probabilities: p(C|x)
    def posterior(self, test):
        log_likelihood = self.likelihood(test)
        max_log_likelihood = np.max(log_likelihood, axis=1, keepdims=True)
        log_sum = max_log_likelihood + np.log(np.sum(np.exp(log_likelihood - max_log_likelihood), axis=1, keepdims=True))
        log_posterior = log_likelihood - log_sum
        return np.round(np.exp(log_posterior), 10)

    # return the most likely class for each sample in test
    def predict(self, test):
        log_likelihood = self.likelihood(test)
        pre = np.argmax(log_likelihood, axis = 1) + 1
        return pre

    # discriminant function g_i(x) for each class
    def discriminant(self, test):
        discriminant_values = {}
        for i in np.unique(self.y):
            # Calculate g_i(x) using the shared diagonal covariance matrix
            g = -0.5 * np.sum(np.log(2 * np.pi * self.shared_var))  # log determinant of diagonal covariance matrix
            g += -0.5 * np.sum(((test - self.ms[i]) ** 2) / self.shared_var, axis=1)  # quadratic term
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
