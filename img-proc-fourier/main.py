# imports from project
import Helper.image_io as image_io
import Helper.image_manip as image_manip
from Helper.ring_mask import RingMask
from Helper.plot import plot, plot_clean

# imports from libraries
import numpy as np

# GLOBAL VARIABLES
DEFAULT_IMAGES = ['Resources/bauckhage.jpg',
                  'Resources/clock.jpg',
                  'Resources/cat.png',
                  'Resources/asterixGrey.jpg']


def task_1_1(image_path=DEFAULT_IMAGES[0], r_min=20, r_max=40):
    image = image_io.read_image(image_path, as_array=True)
    new_image = image_manip.draw_circle(image, r_min, r_max, inside=True)
    image_io.combine_magnitude_and_phase(new_image).show()


def task_1(file_path=DEFAULT_IMAGES[0], radius_min=25., radius_max=55.):
    # Import image as a numpy.array
    input_image = image_io.read_image(file_path, as_array=True)

    # create and apply ring mask, where a ring of True lies in a sea of False
    effect_true = lambda x: x * 0  # set element to 0 where mask is True (inside the ring)
    effect_false = lambda x: x  # no change to element when mask is False (outside the ring)
    image_center = (input_image.shape[0] / 2, input_image.shape[1] / 2)  # center of ring

    ring_mask = RingMask(input_image.shape,
                         effect_true, effect_false,
                         radius_min, radius_max,
                         image_center)

    output_image_array = ring_mask.apply_mask(input_image)

    image_io.combine_magnitude_and_phase(output_image_array).show()


def task_1_2(image_path=DEFAULT_IMAGES[0], r_min=20, r_max=40):
    labels = ('Original Image', 'Fourier Transformation', 'Frequency Suppression', 'Inverse Fourier Transformation')
    im = image_io.read_image(image_path, as_array=False)
    ft = np.fft.fft2(im)
    sh = np.fft.fftshift(ft)
    image_io.save_image(np.log(np.abs(sh)), 'Generated/FourierTransformation.jpg')
    picture = image_manip.draw_circle(sh.copy(), r_min, r_max, False)
    image_io.save_image(np.log(np.abs(picture)), 'Generated/FrequencySuppression.jpg')
    inverse_p = np.fft.ifft2(picture)
    image_io.save_image(np.abs(inverse_p), 'Generated/InverseFourierTransformation.jpg')

    plot(im, sh, picture, inverse_p, labels)


def task_1_3(image_path_a=DEFAULT_IMAGES[0], image_path_b=DEFAULT_IMAGES[1]):
    image_a = image_io.read_image(image_path_a, as_array=False)
    image_b = image_io.read_image(image_path_b, as_array=False)
    image_c = image_manip.combine_magnitude_and_phase(image_a, image_b)
    image_d = image_manip.combine_magnitude_and_phase(image_b, image_a)
    plot_clean(image_a, image_b, image_c, image_d,
               ('Image A', 'Image B', 'Magnitude from A, phase from B', 'Magnitude from B, phase from A'))


if __name__ == '__main__':
    # task_1_1()
    # task_1_2()
    # task_1_2(DEFAULT_IMAGES[1])
    task_1_2(DEFAULT_IMAGES[3])
    task_1_3()
    # task_1()