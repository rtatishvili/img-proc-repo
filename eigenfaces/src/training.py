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
    X_train = np.array(shuffled[:boundary])
    X_test = np.array(shuffled[boundary:])

    return (X_train, X_test)


def mean(X_train):
    return np.mean(X_train, axis=0)


def zero_mean(target, source):
    return np.subtract(target, mean(source))


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


def extract_sample(array, size):
    sample = random.random_integers(low=0, high=array.shape[0], size=size)

    return array[sample]


def compute_distances_all(sample, dataset):
    distances = []

    for i in range(dataset.shape[0]):
        dist = np.square(np.subtract(sample, dataset[i,])).sum(axis=1)
        distances.append(dist)

    distances = np.array(distances).reshape(len(distances), sample.shape[0])
    distances = np.sort(distances, axis=0)[::-1]

    return distances


def project(X_train, v, cut_index):
    subspace = v[:, :cut_index]

    # training data contains image per row
    # so we need to transpose to match
    # eigenvector subspace dimensions
    # and transpose back to match the original
    # dataset transposition
    return np.dot(subspace.T, X_train.T).T
