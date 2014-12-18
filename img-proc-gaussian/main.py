import sys
import numpy as np
import matplotlib.pyplot as plt

from image_op.image_io import read_image
import calc.gaussian_mask as gm
import image_op.image_io as io
import image_op.image_manip as im
from plot import plot_multiple_arrays
import time


# GLOBAL VARIABLES
DEFAULT_IMAGES = ['Resources/bauckhage.jpg',
                  'Resources/clock.jpg',
                  'Resources/bauckhage.jpg',
                  'Resources/cat.png',
                  'Resources/asterixGrey.jpg']

times = 10


def task_1_1(size=(3, 3)):
    print "####################################"
    print
    print "Running Task 1.1"

    original_image = read_image(DEFAULT_IMAGES[0], as_array=True)
    mask = gm.generate_gauss_2d(size)
    new_image_2d = im.apply_matrix_mask(original_image, mask)

    # Save the output image
    io.save_array_as_gray_image(new_image_2d, "Generated/task_1_1.jpg")

    # plot the images for visual comparison
    plot_multiple_arrays([[original_image, new_image_2d]],
                         "Task 1.1", ["Input Image: original image", "Output Image: filtered image"])


def task_1_2(size=(3, 3)):
    print "####################################"
    print
    print "Running Task 1.2"

    original_image = read_image(DEFAULT_IMAGES[0], as_array=True)
    mask_x = gm.generate_gauss_1d(size=size[0])
    mask_y = gm.generate_gauss_1d(size=size[1])
    new_image_1d = im.apply_array_mask(original_image, mask_x, mask_y)

    # Save the output image
    io.save_array_as_gray_image(new_image_1d, "Generated/task_1_2.jpg")

    # plot the images for visual comparison
    plot_multiple_arrays([[original_image, new_image_1d]],
                         "Task 1.2", ["Input Image: original image", "Output Image: filtered image"])


def task_1_3(size=(3, 3)):
    print "####################################"
    print
    print "Running Task 1.3"

    original_image = read_image(DEFAULT_IMAGES[0], as_array=True)
    mask = gm.generate_gauss_2d(size)
    mask_zero = gm.zero_pad_mask(mask, original_image.shape)
    new_image_freq = im.apply_fourier_mask(original_image, mask_zero)

    # Save the output image
    io.save_array_as_gray_image(new_image_freq, "Generated/task_1_3.jpg")

    # plot the images for visual comparison
    plot_multiple_arrays([[original_image, abs(new_image_freq)]],
                         "Task 1.3", ["Input Image: original image", "Output Image: filtered image"])


def task_1_4():
    print "####################################"
    print
    print "Running Task 1.4"

    original_image = read_image(DEFAULT_IMAGES[0], as_array=True)
    mask_2d = [gm.generate_gauss_2d((size, size)) for size in range(3, 26) if size % 2 == 1]
    mask_1d = [gm.generate_gauss_1d(size) for size in range(3, 26) if size % 2 == 1]
    mask_padded = [gm.zero_pad_mask(mask_2d[i], original_image.shape) for i in range(len(mask_2d))]

    time_results = np.zeros(4*len(mask_2d)).reshape((len(mask_2d), 4))

    # Average performance measure of Gaussian Filter using 2D matrix
    print "----- 2d convolution -----"
    count = 0
    for mask in mask_2d:
        time_sum = 0.
        for i in range(times):
            temp_time = time.clock()
            im.apply_matrix_mask(original_image, mask)
            time_sum += time.clock() - temp_time
        time_results[count, 0] = mask.shape[0]
        time_results[count, 2] = time_sum / float(times)
        count += 1
        print 'Mask size ' + str(mask.shape[0]) + ', in time:' + str(time_sum / float(times))

    # Average performance measure of Gaussian Filter using two 1D vectors
    print "----- 1d convolution -----"
    count = 0
    for mask in mask_1d:
        time_sum = 0.
        for i in range(times):
            temp_time = time.clock()
            im.apply_array_mask(original_image, mask, mask)
            time_sum += time.clock() - temp_time
        time_results[count, 0] = mask.shape[0]
        time_results[count, 1] = time_sum / float(times)
        count += 1
        print 'Mask size ' + str(mask.shape[0]) + ', in time:' + str(time_sum / float(times))

    # Average performance measure of Gaussian Filter in frequency domain
    print "----- fourier filtering -----"
    output_images = []
    count = 0
    for mask in mask_padded:
        time_sum = 0.
        for i in range(times):
            temp_time = time.clock()
            im.apply_fourier_mask(original_image, mask)
            time_sum += time.clock() - temp_time
        time_results[count, 3] = time_sum / float(times)
        count += 1
        print 'Mask size ' + str(mask.shape[0]) + ', in time:' + str(time_sum / float(times))

    labels = ['Fourier filter', '2D - Convolution', '1D - Convolution']
    plt.title('Performance of different implementation of Gaussian Filter').set_fontsize(28)
    plt.xlabel('Mask dimension', fontsize=24)
    plt.ylabel('time (s)', fontsize=24)

    plt.xticks(time_results.T[0])
    plt.plot(time_results.T[0], time_results.T[3])
    plt.plot(time_results.T[0], time_results.T[2])
    plt.plot(time_results.T[0], time_results.T[1])
    plt.legend(labels, loc='upper left',
               labelspacing=0.0, handletextpad=0.0,
               handlelength=1.5,
               fancybox=True, shadow=True, prop={'size': 18})
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.show()

    for mask in mask_padded:
        output_images.append(abs(im.apply_fourier_mask(original_image, mask)))

    plot_multiple_arrays(np.array(output_images).reshape(3, 4, 256, 256),
                         "Task 1.4", [str(i.shape[0]) + "x" + str(i.shape[0]) for i in mask_2d])


def task_2_1(size=(3, 3)):
    print "####################################"
    print
    print "Running Task 2.1"
    original_image = read_image(DEFAULT_IMAGES[0], as_array=True)
    img_dx, img_dy = gm.gauss_derivatives(original_image, size)

    plot_multiple_arrays([[original_image,
                           io.normalize_image(img_dx),
                           io.normalize_image(img_dy),
                           np.sqrt(np.power(img_dx, 2) + np.power(img_dy, 2))]],
             "Task 2.1", ["original image",
                              "gradient x",
                              "gradient y",
                              "magnitude of gradient"])

    io.save_array_as_gray_image(img_dx, "Generated/task_2_1_dx.jpg", normalize=True)
    io.save_array_as_gray_image(img_dy, "Generated/task_2_1_dy.jpg", normalize=True)
    io.save_array_as_gray_image(np.sqrt(np.power(img_dx, 2) + np.power(img_dy, 2)),
                                "Generated/task_2_1_magnitude.jpg", normalize=True)


def main(argv):
    task_1_1((15, 15))
    task_1_2((15, 15))
    task_1_3((15, 15))
    task_1_4()
    task_2_1()


if __name__ == '__main__':
    main(sys.argv)