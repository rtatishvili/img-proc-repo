'''
Created on Nov 6, 2014

@author: revaz
'''
from fourier.Helper import Helper


def task_1_1():
    image = Helper.read_image('test/bauckhage.jpg', False)
    image.show()

if __name__ == '__main__':
    task_1_1()