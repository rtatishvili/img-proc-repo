from image_op.image_io import read_image
import numpy as np

f = read_image("Resources/bauckhage.jpg", as_array=True)

x_in = np.arange(f.shape[1])
y_in = [x for x in xrange(f.shape[0])]

a = 128
ph = 0.
tp = f.shape[0] * 2.
freq = 1./tp

x_out = a * np.sin(2 * np.math.pi * freq * x_in + ph)

# import matplotlib.pyplot as plt
#
# plt.plot(x_in, x_out)
# plt.show()

print x_out.min(), x_out.max()

g = np.zeros((f.shape[0] + abs(x_out.min()) + abs(x_out.max()), f.shape[1]))

for y_out in xrange(f.shape[0]):
    for x in xrange(f.shape[1]):
        g[y_out + a - x_out[x], x] = f[y_out, x]

import plot

plot.plot_multiple_arrays([[g]], '', [''])