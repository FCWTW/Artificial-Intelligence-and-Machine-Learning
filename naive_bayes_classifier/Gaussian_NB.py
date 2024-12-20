import numpy as np
from math import pi

'''
程式碼來源:
https://roger010620.medium.com/%E8%B2%9D%E6%B0%8F%E5%88%86%E9%A1%9E%E5%99%A8-naive-bayes-classifier-%E5%90%ABpython%E5%AF%A6%E4%BD%9C-66701688db02
'''

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
            s = np.std(rows, axis=0) + 1e-9             # prevent standard deviation from being 0
            self.ms[class_value] = np.vstack([m,s])     # stack the mean and standard deviation of the features for each class into a two-dimensional array
    
    # log likelihood function: log[p(x|C)]
    def likelihood(self, test):
        likelihood = []
        for i in np.unique(self.y):
            log_prior = np.log(self.prior[i])           # use logarithmic space calculations to avoid numerical overflows
            log_likelihood = -0.5*np.log(2*pi*np.square(self.ms[i][1]))
            log_likelihood = log_likelihood - 0.5*(test - self.ms[i][0])**2/np.square(self.ms[i][1])
            likelihood.append(log_prior+np.sum(log_likelihood, axis = 1))
        return np.array(likelihood).T
    
    # print likelihood value
    def print_likelihood(self, test):
        log_likelihood = self.likelihood(test)
        # Convert log likelihood to likelihood and round to 10 decimal places
        likelihood = np.round(np.exp(log_likelihood), 10)
        
        for i, row in enumerate(likelihood):
            row_6 = ", ".join([f"{value:.6f}" for value in row])
            print(f"Test Sample {i+1}: Likelihoods = {row_6}")

    # prior probabilities: p(C)
    def print_prior(self):
        for class_value, prior_prob in self.prior.items():
            print(f"Class {class_value}: Prior Probability = {prior_prob:.6f}")

    # posterior probabilities: p(C|x)
    def posterior(self, test):
        log_likelihood = self.likelihood(test)

        # calculate (prior probabilities * likelihood) in logarithmic space
        max_log_likelihood = np.max(log_likelihood, axis=1, keepdims=True)
        log_sum = max_log_likelihood + np.log(np.sum(np.exp(log_likelihood - max_log_likelihood), axis=1, keepdims=True))

        # log(posterior probabilities) = log(likelihood) + log(prior) - log(sum)
        log_posterior = log_likelihood - log_sum
        return np.round(np.exp(log_posterior), 10)

    # return the most likely class for each sample in test
    def predict(self, test):
        log_likelihood = self.likelihood(test)
        # Maximum Likelihood Estimator (MLE)
        class_indices = np.argmax(log_likelihood, axis=1)
        
        # Map the index back to the original class label
        unique_classes = np.unique(self.y)
        pre = unique_classes[class_indices]
        return pre

    # print means and standard deviation
    def print_mean(self):
        for class_value, values in self.ms.items():
            m, s = values[0], values[1]
            m_str = ", ".join([f"{mean_value:.6f}" for mean_value in m])
            s_str = ", ".join([f"{std_value:.6f}" for std_value in s])
            print(f"Class {class_value}:")
            print(f"Mean: {m_str}")
            print(f"Standard Deviation: {s_str}")

    # return the determinant of covariance matrix
    def determinant_covariance(self):
        determinants = {}
        for class_value, rows in self.class_dict.items():
            cov_matrix = np.cov(rows, rowvar=False)
            determinants[class_value] = np.linalg.det(cov_matrix)
        return determinants

    # discriminant function g_i(x) for each class
    def discriminant(self, test):
        discriminant_values = {}
        for i in np.unique(self.y):
            # Calculate g_i(x)
            g = -0.5 * np.sum(np.log(2 * np.pi * self.ms[i][1]))  # log determinant of diagonal covariance matrix
            g += -0.5 * np.sum(((test - self.ms[i][0]) ** 2) / np.square(self.ms[i][1]), axis=1)  # quadratic term
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