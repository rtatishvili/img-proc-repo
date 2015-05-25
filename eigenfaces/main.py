import src.image_loader as iml
import src.training as tr
import src.plot as plot
import numpy as np


ALL_IMAGES = 2429
SIGNIFICANCE = 0.9
TEST_SAMPLE_SIZE = 10
dataset = []
global X_train
global X_test


def compare_datasets_of_different_sizes():
    global X_train
    global X_test
    
    e1, v1 = get_eigenvalues_of_images(X_train[:100])
    e2, v2 = get_eigenvalues_of_images(X_train[:500])
    e3, v3 = get_eigenvalues_of_images(X_train[:1000])
    e4, v4 = get_eigenvalues_of_images(X_train)
    
    plot.eigenvalues((e1, e2, e3, e4))


def get_eigenvalues_of_images(X_train):
    C = tr.compute_cov(X_train.T)
    eigenvalues, eigenvectors = tr.compute_eigenvalues(C)
    eigenvalues, eigenvectors = tr.sort_eigenpairs(eigenvalues, eigenvectors)

    return eigenvalues, eigenvectors


def find_eigenvalue_cutting_point():
    e, v = get_eigenvalues_of_images(dataset)
    eigensum = np.cumsum(e)
    threshold = SIGNIFICANCE * eigensum[-1]
    i = 0

    while eigensum[i] < threshold:
        i += 1

    plot.sum_of_eigenvalues(eigensum, threshold)
    plot.sum_of_eigenvalues_range(eigensum, threshold, i)    

    return i


if __name__ == '__main__':
    global X_train
    global X_test
    
    print 'Loading image data set'
    dataset = iml.load_images()
    X_train, X_test = tr.divide_dataset_into_train_test(dataset)
    X_train = tr.zero_mean(X_train, X_train)
    X_test = tr.zero_mean(X_test, X_train)    
    
#     print 'Comparing data sets of different sizes. See eigenvalues.png'
#     compare_datasets_of_different_sizes()    
     
    print 'Finding first N eigenvalues and eigenvectors of \'high significance\'. See cutting_point.png and cutting_point_zoomed.png'
    cut_index = find_eigenvalue_cutting_point()
     
    print 'Visualizing first ' + str(cut_index) + ' eigenvectors a.k.a. eigenfaces. See eigenfaces.png'
    e, v = get_eigenvalues_of_images(dataset)
    plot.images(v.T, cut_index)
    
    print 'Plotting Euclidean distance between some of the test set instances and training set'
    
    # TODO refactor to the function
    from numpy import random
        
    sample = random.random_integers(low=0, high=X_test.shape[0], size=TEST_SAMPLE_SIZE)
    
    X_test_sample = X_test[sample]    
    
    distances = []
    
    for i in range(X_train.shape[0]):
        dist = np.square(np.subtract(X_test_sample, X_train[i,])).sum(axis=1)
        distances.append(dist)
    
    distances = np.array(distances).reshape(len(distances), TEST_SAMPLE_SIZE)
    distances = np.sort(distances, axis=0)[::-1]
        
    plot.dist_values(distances)
    
    