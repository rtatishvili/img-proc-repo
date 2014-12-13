import numpy as np

def calculate_sigma(M):
    """
    """
    return (M - 1.0) / (2.0 * 2.575)

def calculate_gauss_dimension(size):
    if type(size) is tuple:
        m,n = [(ss-1.)/2. for ss in size]
        y,x = np.ogrid[-m:m+1,-n:n+1]
        return  (x*x + y*y)
    else:
        m = (size-1.)/2.
        x = np.arange(-m, m+1)
        return x*x

def generate_gauss(size=(3, 3), sigma=0.5):
    """
    """
    p = calculate_gauss_dimension(size)
    mask = np.exp( -p / (2.*sigma*sigma) )
    sum_ = mask.sum()
    if sum_ != 0:
        mask /= sum_
    return mask

def generate_gauss_1D(size=3, sigma = None):
    if sigma == None:
        sigma = calculate_sigma(size)
    return generate_gauss(size, sigma)

def generate_gauss_2D(size=(3,3), sigma = None):
    """
    2D gaussian mask 
    """
    if sigma == None:
        sigma = calculate_sigma(size[0])
    return generate_gauss(size, sigma)