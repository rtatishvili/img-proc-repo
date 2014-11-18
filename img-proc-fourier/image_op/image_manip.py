import numpy as np
from Assert import Assert
from calc import fourier_calc as fourier_calc


def combine_magnitude_and_phase(ft_magnitude_image, ft_phase_image):
    """
    This method creates new image by combining the magnitude form **ft_magnitude_image** and phase from **ft_phase_image**. Client must provide
    two images with same resolution.
    @param ft_magnitude_image: Fourier transform of the first image
    @param ft_phase_image: Fourier transform of the second image
    @return: new image
    """
    Assert.isTrue(ft_magnitude_image.size == ft_phase_image.size, "image_op have different resolution!")
    
    magnitude = fourier_calc.magnitude(ft_magnitude_image)
    phase = fourier_calc.phase(ft_phase_image)
    combined_ft_image = fourier_calc.create_complex_array(magnitude, phase)

    return np.fft.ifft2(combined_ft_image)

