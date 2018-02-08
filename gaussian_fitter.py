import pdb
import sys
import glob

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

from util import gaussian_2d, gaussian_1d

sys.path.insert(0, 'C:\\PycharmProjects\\THzProcClass')

from THzData import THzData

basedir = 'C:\\Work\\Pin Reflector\\Ordered Data'
basefile = 'Pin_mp_'

gate = [[0, 4096], [-175, 200]]

radius = 0.75  # mm
sigma = 1.25

z = np.linspace(-2.5, 2.5, 11)

data_array = list()

n_files = len(glob.glob(basedir + '\\*.tvl'))

for i in range(n_files):
    if i < 10:
        filename = basefile + '0' + str(i) + '.tvl'
    else:
        filename = basefile + str(i) + '.tvl'

    data_array.append(THzData(filename, basedir, gate=gate, signal_type=1))

data = data_array[5]  # this is focus

c_scans = np.zeros((n_files, data.y_step, data.x_step))

for i, scan in enumerate(data_array):
    c_scans[i] = scan.c_scan

# find the maximum amplitude of each C-Scan, resulting array is n_files long
max_amps = c_scans.max(axis=1).max(axis=1)

normalized_c_scan = data.c_scan / data.c_scan.max()

yy, xx = np.meshgrid(data.y, data.x, indexing='ij')

xdata = np.vstack((yy.flatten(), xx.flatten()))

# sol = curve_fit(gaussian_2d, xdata, data.c_scan)

# need to determine what the approximate shape of C-Scan gaussian
# initial guess for each of the arguments
A = 1
x0 = 0
y0 = 0
sigma_x = 0.7
sigma_y = 0.7
p0 = (A, x0, y0, sigma_x, sigma_y)

popt, pcov = curve_fit(gaussian_2d, xdata, normalized_c_scan.flatten(), p0)

# change p0 to see what 2D gaussian looks like
# p0 = (1, 0, 0, sigma, sigma)
made_c_scan = gaussian_2d(xdata, *popt).reshape(data.y_step, data.x_step)

x = np.linspace(-3, 3, 100)
y = gaussian_1d(x, 1, 0, sigma)

beam_spot = Circle((0, 0), radius)
patches = list()
patches.append(beam_spot)
p = PatchCollection(patches, edgecolors='red', facecolors='none')

plt.figure('True C-Scan')
plt.imshow(data.c_scan, cmap='gray', extent=data.c_scan_extent,
           interpolation='none')
plt.xlabel('X Scan Location (mm)')
plt.ylabel('Y Scan Location (mm)')
plt.grid()
plt.colorbar()

fig = plt.figure('Manufactured C-Scan')
axis = fig.add_subplot(111)
image = axis.imshow(made_c_scan, cmap='gray', extent=data.c_scan_extent,
                    interpolation='none')
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

# now we want to determine the vertical spread of the beam
A = max_amps.max()
mu = 0
sigma = 0.7
p0 = (A, mu, sigma)
popt, pcov = curve_fit(gaussian_1d, z, max_amps, p0)

z_dense = np.linspace(-2.5, 2.5, 1001)
best_fit = gaussian_1d(z_dense, a=max_amps.max(), mu=0, sigma=1)

plt.figure('Vertical Beam Spread')
plt.plot(z, max_amps, 'bo', label='Data')
plt.plot(z_dense, best_fit, 'r', label='Best Fit')
plt.title('Vertical Beam Spread')
plt.xlabel('Height Relative to Focus (mm)')
plt.ylabel('C-Scan Max Amplitude')
plt.grid()
plt.legend()

plt.show()
