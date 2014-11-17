'''
Created on Nov 12, 2014

@author: kostadin
'''
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