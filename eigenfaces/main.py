import src.image_loader as iml
import src.training as tr
import src.plot as plot
import numpy as np

ALL_IMAGES = 2429

def compare_datasets_of_different_sizes():
    e1, v1 = get_eigenvalues_of_images(100)
    e2, v2 = get_eigenvalues_of_images(500)
    e3, v3 = get_eigenvalues_of_images(1000)
    e4, v4 = get_eigenvalues_of_images(ALL_IMAGES)
    
    plot.eigenvalues((e1, e2, e3, e4))

def get_eigenvalues_of_images(count):
    images = iml.load_images(0, count)
    (X_train, X_test) = tr.divide_dataset_into_train_test(images)   
    C = tr.compute_cov(X_train.T)
    eigenvalues, eigenvectors = tr.compute_eigenvalues(C)
    eigenvalues, eigenvectors = tr.sort_eigenpairs(eigenvalues, eigenvectors)

    return eigenvalues, eigenvectors


def find_eigenvalue_cutting_point():
    e, v = get_eigenvalues_of_images(ALL_IMAGES)
    eigensum = np.cumsum(e)
    threshold = 0.9 * eigensum[-1]
    i = 0

    while eigensum[i] < threshold:
        i += 1

    plot.sum_of_eigenvalues(eigensum, threshold)
    plot.sum_of_eigenvalues_range(eigensum, threshold, i)    

    return i


if __name__ == '__main__':
    print 'Comparing data sets of different sizes. See eigenvalues.png'
    compare_datasets_of_different_sizes()    
    print 'Finding first N eigenvalues and eigenvectors of \'high significance\'. See cutting_point.png and cutting_point_zoomed.png'
    cut_index = find_eigenvalue_cutting_point()
    print 'Visualizing first 25 eigenvectors a.k.a. eigenfaces. See eigenfaces.png'
    e, v = get_eigenvalues_of_images(ALL_IMAGES)
    plot.images(v.T, 25)

