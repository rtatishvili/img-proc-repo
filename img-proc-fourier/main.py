# imports from project
from Helper.image_io import read_image, save_image, create_image
from Helper.image_manip import combine_magnitude_and_phase
from Helper.ring_mask import RingMask
from Helper.plot import plot, plot_clean

# imports from libraries
import numpy as np

# GLOBAL VARIABLES
DEFAULT_IMAGES = ['Resources/bauckhage.jpg',
                  'Resources/clock.jpg',
                  'Resources/cat.png',
                  'Resources/asterixGrey.jpg']


def task_1(file_path=DEFAULT_IMAGES[0], radius_min=25., radius_max=55.):
    # Import image as a numpy.array
    input_image = read_image(file_path, as_array=True)

    # create and apply ring mask, where a ring of True lies in a sea of False
    
    # set element to 0 inside the ring
    func_inside_ring = lambda x: 0
    # no change to element outside the ring
    func_outside_ring = lambda x: x  
    # center of ring
    image_center = (input_image.shape[0] / 2, input_image.shape[1] / 2) 

    # Create a mask object containing the area of the ring and 
    # the functions to apply inside and outside the ring
    ring_mask = RingMask(input_image.shape,
                         func_inside_ring, func_outside_ring,
                         radius_min, radius_max,
                         image_center)

    # The image containing the ring (filled with black inside) 
    output_image_array = ring_mask.apply_mask(input_image)

    # create an image object from array to show on a screen
    create_image(output_image_array).show()


def task_2(image_path=DEFAULT_IMAGES[0], radius_min=25., radius_max=55.):
    """
    :param image_path: The file path to the image on which the Fourier Transformation is applied
    :param radius_min: Minimal radius defining the inner edge of the ring
    :param radius_max: Maximal radius defining the outer edge of the ring
    """
    
    # Load an image as Image object
    input_image_array = read_image(image_path, as_array=False)

    # Apply fast Fourier transformation to an image
    fft_image_array = np.fft.fft2(input_image_array)
    
    # Shift the Fourier image to the center of the image  
    ft_image = np.fft.fftshift(fft_image_array)

    # save the shifted Fourier transform image
    save_image(np.log(np.abs(ft_image)), 'Generated/FourierTransformation.jpg')

    # create and apply a ring shaped frequency mask on a Fourier transform of an image
    # where the mask suppresses the frequencies outside the ring
    
    # no change to the element, i.e. allow the frequency
    func_inside_ring = lambda x: x  
    # set element to zero, i.e. suppress the frequency
    func_outside_ring = lambda x: 0
    # center of ring
    center = (ft_image.shape[0] / 2, ft_image.shape[1] / 2)  

    # Create a mask object containing the area of the ring and 
    # the functions to apply inside and outside the ring
    ring_freq_mask = RingMask(ft_image.shape,
                              func_inside_ring, func_outside_ring,
                              radius_min, radius_max,
                              center)

    suppressed_ft_image = ring_freq_mask.apply_mask(ft_image)

    # save the suppressed version shifted Fourier transform image
    save_image(np.log(np.abs(suppressed_ft_image)), 'Generated/FrequencySuppression.jpg')

    # out_fft_image_array = ifftshift.numpy_ifftshift(suppressed_ft_image) # ifftshit has no effect on final image
    # ift_image = ifft2.numpy_ifft2(out_fft_image_array)

    # Apply inverse Fourier transformation on a suppressed image
    ift_image = np.fft.ifft2(suppressed_ft_image)

    # plot images
    # TODO: improve plotting
    labels = ('Original Image', 'Fourier Transformation', 'Frequency Suppression', 'Inverse Fourier Transformation')
    plot(input_image_array, ft_image, suppressed_ft_image, ift_image, labels)


def task_3(image_path_list=DEFAULT_IMAGES[:2]):
    """ 
    :param image_path_a: The file path to the first image
    :param image_path_b: The file path to the second image
    """
    # import intensity images as arrays from list of file paths into a list of input images
    input_image_array_list = []
    for path in image_path_list:
        input_image_array_list.append(read_image(path, as_array=True))
 
    # assert that all imported image arrays are of the same shape
    assert np.all([input_image_array_list[0].shape == shape for shape in
                   [image.shape for image in input_image_array_list]]), "All images need to be of the same size"
 
    # calculate the fft of all images
    ft_list = []
    for image_array in input_image_array_list:
        ft_list.append(np.fft.fft2(image_array))

    image_list = []

    for i in range(len(ft_list)):
        for j in range(len(ft_list)):
            # Construct new image from the magnitude part of the first image and phase part of the second image
            image_list.append(combine_magnitude_and_phase(ft_list[i], ft_list[j]))
                
    # display the images for comparison
    plot_clean(image_list[0], image_list[1], image_list[2], image_list[3],
                ('Image A', 'Magnitude A, Phase B', 'Magnitude B, Phase A', 'Image B'))



if __name__ == '__main__':
    task_1()
    task_2()
    task_2(DEFAULT_IMAGES[1])
    task_2(DEFAULT_IMAGES[3])
    task_3()