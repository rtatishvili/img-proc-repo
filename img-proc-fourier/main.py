import fourier.image_io as image_io
import fourier.image_manip as image_manip
import numpy as np

from fourier.helper import plot, plot_clean


def task_1_1():
    image = image_io.read_image('Resources/bauckhage.jpg', as_array=True)
    new_image = image_manip.draw_circle(image, 20, 40, inside = True)
    image_io.create_image(new_image).show() 


def task_1_2():
    labels = ('Original Image', 'Fourier Transformation', 'Frequency Suppression', 'Inverse Fourier Transformation')
    im = image_io.read_image('Resources/bauckhage.jpg', as_array=False)   
    ft = np.fft.fft2(im)
    sh = np.fft.fftshift(ft)
    image_io.save_image(np.log(np.abs(sh)), 'Generated/FourierTransformation.jpg')
    picture = image_manip.draw_circle(sh.copy(), 20, 50, False)
    image_io.save_image(np.log(np.abs(picture)), 'Generated/FrequencySuppression.jpg')
    inverse_p = np.fft.ifft2(picture)
    image_io.save_image(np.abs(inverse_p), 'Generated/InverseFourierTransformation.jpg')
    
    plot(im, sh, picture, inverse_p, labels)

    
    im = image_io.read_image('Resources/clock.jpg', as_array=False)   
    ft = np.fft.fft2(im)
    sh = np.fft.fftshift(ft)
    image_io.save_image(np.log(np.abs(sh)), 'Generated/ClockFourierTransformation.jpg')
    picture = image_manip.draw_circle(sh.copy(), 20, 50, False)
    image_io.save_image(np.log(np.abs(picture)), 'Generated/ClockFrequencySuppression.jpg')
    inverse_p = np.fft.ifft2(picture)
    image_io.save_image(np.abs(inverse_p), 'Generated/ClockInverseFourierTransformation.jpg')
    
    plot(im, sh, picture, inverse_p, labels)

    
    im = image_io.read_image('Resources/cat.png', as_array=False)   
    ft = np.fft.fft2(im)
    sh = np.fft.fftshift(ft)
    image_io.save_image(np.log(np.abs(sh)), 'Generated/CatFourierTransformation.jpg')
    picture = image_manip.draw_circle(sh.copy(), 20, 50, False)
    image_io.save_image(np.log(np.abs(picture)), 'Generated/CatFrequencySuppression.jpg')
    inverse_p = np.fft.ifft2(picture)
    image_io.save_image(np.abs(inverse_p), 'Generated/CatInverseFourierTransformation.jpg')
    
    plot(im, sh, picture, inverse_p, labels)
    
    im = image_io.read_image('Resources/asterixGrey.jpg', as_array=False)   
    ft = np.fft.fft2(im)
    sh = np.fft.fftshift(ft)
    image_io.save_image(np.log(np.abs(sh)), 'Generated/asterixGreyFourierTransformation.jpg')
    picture = image_manip.draw_circle(sh.copy(), 20, 50, False)
    image_io.save_image(np.log(np.abs(picture)), 'Generated/asterixGreyFrequencySuppression.jpg')
    inverse_p = np.fft.ifft2(picture)
    image_io.save_image(np.abs(inverse_p), 'Generated/CasterixGreyInverseFourierTransformation.jpg')
    
    plot(im, sh, picture, inverse_p, labels)


def task_1_3():
    imA = image_io.read_image('Resources/bauckhage.jpg', as_array=False)
    imB = image_io.read_image('Resources/clock.jpg', as_array=False)
    imC = image_manip.create_image(imA, imB)
    imD = image_manip.create_image(imB, imA)
    plot_clean(imA, imB, imC, imD, ('Image A', 'Image B', 'Magnitude from A, phase from B', 'Magnitude from B, phase from A'))


if __name__ == '__main__':
    task_1_1()
    task_1_2()
    task_1_3()