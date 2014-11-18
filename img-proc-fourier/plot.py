import numpy as np
import matplotlib.pyplot as plt


def plot(im, sh, picture, inverse_p, title):
    fig = plt.figure()
    a = fig.add_subplot(2, 2, 1)
    a.imshow(np.abs(im), cmap=plt.get_cmap('gray'))
    a.set_title(title[0])
    a = fig.add_subplot(2, 2, 2)
    a.imshow(np.log(np.abs(sh)), cmap=plt.get_cmap('gray'))
    a.set_title(title[1])
    a = fig.add_subplot(2, 2, 3)
    a.imshow(np.abs(picture), cmap=plt.get_cmap('gray'))
    a.set_title(title[2])
    a = fig.add_subplot(2, 2, 4)
    a.imshow(np.abs(inverse_p), cmap=plt.get_cmap('gray'))
    a.set_title(title[3])
    plt.show()

def plot_clean(im, sh, picture, inverse_p, title):
    fig = plt.figure()
    a = fig.add_subplot(2, 2, 1)
    a.imshow(np.abs(im), cmap=plt.get_cmap('gray'))
    a.set_title(title[0])
    a = fig.add_subplot(2, 2, 2)
    a.imshow(np.abs(sh), cmap=plt.get_cmap('gray'))
    a.set_title(title[1])
    a = fig.add_subplot(2, 2, 3)
    a.imshow(np.abs(picture), cmap=plt.get_cmap('gray'))
    a.set_title(title[2])
    a = fig.add_subplot(2, 2, 4)
    a.imshow(np.abs(inverse_p), cmap=plt.get_cmap('gray'))
    a.set_title(title[3])
    plt.show()


def plot_2d_gray_multi(im_list):
    """
    :param im_list:
    """
    plt.figure()
    max_row_size = max([len(x) for x in im_list])
    max_col_size = len(im_list)  # TODO: Gigiti

    subplot = 100 * max_row_size + 10 * max_col_size + 1

    for each_row in im_list:
        for each in each_row:
            plt.subplot(subplot)
            plt.imshow(each, cmap=plt.get_cmap('gray'))
            subplot += 1

    plt.show()