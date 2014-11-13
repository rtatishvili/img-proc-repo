import fourier.image_io as image_io
import fourier.image_manip as image_manip
import numpy as np

from fourier.helper import plot, plot_clean

bauckhage = 'Resources/bauckhage.jpg'
clock = 'Resources/clock.jpg'
cat = 'Resources/cat.png'
asterix = 'Resources/asterixGrey'

def task_1_1(image_path = bauckhage, r_min = 20, r_max = 40):
    image = image_io.read_image(image_path, as_array=True)
    new_image = image_manip.draw_circle(image, r_min, r_max, inside = True)
    image_io.combine_magnitude_and_phase(new_image).show() 


def task_1_2(image_path = bauckhage, r_min = 20, r_max = 40):
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


def task_1_3(image_path_a = bauckhage, image_path_b = clock):
    imA = image_io.read_image(image_path_a, as_array=False)
    imB = image_io.read_image(image_path_b, as_array=False)
    imC = image_manip.combine_magnitude_and_phase(imA, imB)
    imD = image_manip.combine_magnitude_and_phase(imB, imA)
    plot_clean(imA, imB, imC, imD, ('Image A', 'Image B', 'Magnitude from A, phase from B', 'Magnitude from B, phase from A'))


if __name__ == '__main__':
    task_1_1()
    task_1_2()
    task_1_2(clock)
    task_1_2(cat)
    task_1_3()