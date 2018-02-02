import pdb
import sys

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from util import gaussian_2d

sys.path.insert(0, 'D:\\PycharmProjects\\THzProcClass')

from THzData import THzData

basedir = 'D:\\Work\\Pin Reflector'
filename = 'Pin_mp_0.tvl'

gate = [[0, 4096], [-175, 200]]

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

made_c_scan = gaussian_2d(xdata, *popt).reshape(data.y_step, data.x_step)

plt.figure('True C-Scan')
plt.imshow(data.c_scan, cmap='gray', extent=data.c_scan_extent)
plt.xlabel('X Scan Location (mm)')
plt.ylabel('Y Scan Location (mm)')
plt.grid()
plt.colorbar()

plt.figure('Manufactured C-Scan')
plt.imshow(made_c_scan, cmap='gray', extent=data.c_scan_extent,
           vmin=data.c_scan.min(), vmax=data.c_scan.max())
plt.xlabel('X scan Location (mm)')
plt.ylabel('Y Scan Location (mm)')
plt.grid()
plt.colorbar()

plt.show()
