import numpy as np
from numpy import random


def divide_dataset_into_train_test(dataset, ratio=0.9):
    '''
    Divide a data set into two subsets randomly, where the first part is
    considered as a training set containing the ratio part of the data set and
    the second part is considered as a test set	containing the remaining part
    of the data set.
    :param dataset: the original data set
    :param ratio: the ratio of training set size w.r.t. the data set size
    :return: a tuple containing two arrays, (training set, test set)
    '''
    shuffled = random.permutation(dataset)
    boundary = int(len(dataset) * ratio)
    X_train = shuffled[:boundary]
    X_test = shuffled[boundary:]

    return (X_train, X_test)


def mean(X_train):
    return np.mean(X_train, axis=0)


def zero_mean(X_train):
    return np.subtract(X_train, mean(X_train))


def compute_cov(X_train):
    return np.cov(X_train)


def compute_eigenvalues(cov):
    '''
    :return: eigenvalues and eigenvectors as a tuple
    '''
    return np.linalg.eigh(cov)


def sort_eigenpairs(eigenvalues, eigenvectors):
    indices = np.argsort(eigenvalues)[::-1]
    return eigenvalues[indices], eigenvectors.T[indices].T
    
    


