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

    print 'Loading image data set...'
    dataset = iml.load_images()
    X_train, X_test = tr.divide_dataset_into_train_test(dataset)
    X_train_m = tr.zero_mean(X_train, X_train)
    X_test_m = tr.zero_mean(X_test, X_train)
    print 'Done.'
    print ''

    print 'Comparing data sets of different sizes...'
    compare_datasets_of_different_sizes()
    print 'Done. See eigenvalues.png'
    print ''

    print 'Finding first N eigenvalues and eigenvectors of \'high significance\'...'
    cut_index = find_eigenvalue_cutting_point()
    print 'Done. See cutting_point.png and cutting_point_zoomed.png'
    print ''

    print 'Visualizing first ' + str(cut_index) + ' eigenvectors a.k.a. eigenfaces...'
    e, v = get_eigenvalues_of_images(dataset)
    plot.images(v.T, cut_index)
    print 'Done. See eigenfaces.png'
    print ''

    print 'Plotting Euclidean distance between some of the test set instances and training set...'
    X_test_sample_m = tr.extract_sample(X_test, TEST_SAMPLE_SIZE)
    distances, nearest = tr.compute_distances_all(X_test_sample_m, X_train)
    plot.dist_values(distances, filename='distances_highdim.png')
    print 'Done. See distances_highdim.png'
    print ''

    print 'Projecting the dataset and the samples into the subspace...'
    X_train_p = tr.project(X_train_m, v, cut_index)
    X_test_sample_p = tr.project(X_test_sample_m, v, cut_index)

    distances_p, nearest_p = tr.compute_distances_all(X_test_sample_p, X_train_p)
    distances = np.vstack((distances.T, distances_p.T)).T
    distances = np.sqrt(distances)

    plot.dist_comparison(distances, filename='dist_comparison.png')

    print 'Done. See X_dist_comparison.png'
    print ''

    print 'Comparing original and subspace nearest neighbors for the samples...'
    print nearest
    print nearest_p
