# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 17:34:30 2019

@author: flztiii
"""

import numpy as np

def PCA(data, k=1):
    n_samples, n_features = data.shape
    #print(data.shape)
    mean_data = np.mean(data, axis=0)
    data = data - mean_data
    cov_data = np.cov(data, rowvar=0)
    values,egiens = np.linalg.eig(cov_data)
    arg_sorts = np.argsort(values)
    arg_sorts = arg_sorts[:-(k+1):-1]
    egiens = egiens[:,arg_sorts]
    low_mat = np.dot(data, egiens)
    #print(low_mat)
    return low_mat, egiens

if __name__ == "__main__":
    test_data = np.array([[-1, 1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    print(test_data.shape)
    PCA(test_data)