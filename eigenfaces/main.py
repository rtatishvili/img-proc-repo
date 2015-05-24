import src.image_loader as iml
import src.training as tr
import numpy as np
import matplotlib.pyplot as plt

def get_eigenvalues_of_images(count):
    images = iml.load_images(0, count)
    (X_train, X_test) = tr.divide_dataset_into_train_test(images)
    C = tr.compute_cov(X_train.T)
    (eigenvalues, eigenvectors) = tr.compute_eigenvalues(C)

    sorted_eigenvalues = np.sort(eigenvalues)[::-1]

    return sorted_eigenvalues


def plot_eigenvalues():
    a = get_eigenvalues_of_images(100)
    b = get_eigenvalues_of_images(500)
    c = get_eigenvalues_of_images(1000)
    d = get_eigenvalues_of_images(2429)

    e = np.vstack((a, b, c, d)).T

    plt.plot(e)
    plt.xlabel("eigenvalues")
    plt.ylim(1, 100000)
    plt.yscale('log')
    plt.savefig("results/eigenvalues.png", format='png')


def find_eigenvalue_cutting_point():
    e = get_eigenvalues_of_images(2429)
    sum = np.cumsum(e)
    i = 0

    while sum[i] < 0.9 * sum[-1]:
        print i, sum[i], sum[-1]
        i += 1

    print
    print i, sum[i], sum[-1]

    return i

if __name__ == '__main__':
    cut_index = find_eigenvalue_cutting_point()
    # TODO check whether eigenvectors should be sorted as well
