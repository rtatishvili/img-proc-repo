import numpy as np


def phase(val):
    """
     This method extracts the phase of a complex number. Client must provide complex number.
     @param val: complex number from which the phase is extracted
     @return: phase from a complex number   
    """
    assert val.dtype.type == np.complex128, "Parameter is not of type complex!"
    return np.arctan2(np.imag(val), np.real(val))


def magnitude(val):
    """
    This method extracts the magnitude of a complex number. Client must provide complex number.
    @param val: complex number from which the phase is extracted
    @return: magnitude of a complex number
    """
    assert val.dtype.type == np.complex128, "Parameter is not of type complex!"
    return np.sqrt(np.imag(val) ** 2 + np.real(val) ** 2)


def create_complex_array(mag, angle):
    """
    This method creates complex array. Client must provide mag and angle.
    @param mag: of the number (distance of the point P from the origin O)
    @param angle: angle between the positive real axis and the line segment OP
    @return: complex number
    """
    return mag * (np.cos(angle) + np.sin(angle) * 1j)