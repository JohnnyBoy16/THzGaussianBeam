import pdb
import sys

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

from util import gaussian_2d, gaussian_1d

sys.path.insert(0, 'D:\\PycharmProjects\\THzProcClass')

from THzData import THzData

basedir = 'D:\\Work\\Pin Reflector'
filename = 'Pin_mp_0.tvl'

gate = [[0, 4096], [-175, 200]]

radius = 0.75  # mm
sigma = 1.25

# load in the data file that was taken at the focus height
data = THzData(filename, basedir, gate=gate, follow_gate_on=True, signal_type=1)

yy, xx = np.meshgrid(data.y, data.x, indexing='ij')

xdata = np.vstack((yy.flatten(), xx.flatten()))

# sol = curve_fit(gaussian_2d, xdata, data.c_scan)

# need to determine what the approximate shape of C-Scan gaussian
# initial guess for each of the arguments
A = data.c_scan.max()
x0 = 0
y0 = 0
sigma_x = 0.7
sigma_y = 0.7
p0 = (A, x0, y0, sigma_x, sigma_y)

popt, pcov = curve_fit(gaussian_2d, xdata, data.c_scan.flatten(), p0)

# change p0 to see what 2D gaussian looks like
p0 = (1, 0, 0, sigma, sigma)
made_c_scan = gaussian_2d(xdata, *p0).reshape(data.y_step, data.x_step)

x = np.linspace(-3, 3, 100)
y = gaussian_1d(x, 1, 0, sigma)

beam_spot = Circle((0, 0), radius)
patches = list()
patches.append(beam_spot)
p = PatchCollection(patches, edgecolors='red', facecolors='none')

plt.figure('True C-Scan')
plt.imshow(data.c_scan, cmap='gray', extent=data.c_scan_extent)
plt.xlabel('X Scan Location (mm)')
plt.ylabel('Y Scan Location (mm)')
plt.grid()
plt.colorbar()

fig = plt.figure('Manufactured C-Scan')
axis = fig.add_subplot(111)
image = axis.imshow(made_c_scan, cmap='gray', extent=data.c_scan_extent,
                    interpolation='bicubic')
axis.add_collection(p)
plt.xlabel('X scan Location (mm)')
plt.ylabel('Y Scan Location (mm)')
plt.grid()
plt.colorbar(image)

plt.figure('Gaussian Profile')
plt.plot(x, y, 'b')
plt.axhline(gaussian_1d(radius, 1, 0, sigma), color='r', linestyle='--')
plt.title('Gaussian Profile')
plt.xlabel('Radius (mm)')
plt.ylabel('Amplitude')
plt.grid()

plt.show()
