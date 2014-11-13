'''
Created on Nov 10, 2014

@author: kostadin
'''
import numpy as np
from fourier.Assert import Assert

def phase(val):
    """
     This method extracts the phase of a complex number. Client must provide complex number.
     @param val: complex number from which the phase is extracted
     @return: phase from a complex number   
    """
    Assert.isTrue(val.dtype.type == np.complex128, "Parameter is not of type complex!")    
    return np.arctan2(np.imag(val), np.real(val))

def magnitude(val):
    """
    This method extracts the magnitude of a complex number. Client must provide complex number.
    @param val: complex number from which the phase is extracted
    @return: magnitude of a complex number
    """
    Assert.isTrue(val.dtype.type == np.complex128, "Parameter is not of type complex!")    
    return np.sqrt(np.imag(val) ** 2 + np.real(val) ** 2)


def create_complex_number(magnitude, phase):
    """
    This method creates complex number. Client must provide magnitude and phase.
    @param magnitude: of the number (distance of the point P from the origin O)
    @param phase: angle between the positive real axis and the line segment OP
    @return: complex number
    """
    return magnitude * (np.cos(phase) + np.sin(phase)*1j)