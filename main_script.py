import pdb
import sys

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from util import gaussian_2d

sys.path.insert(0, 'C:\\PycharmProjects\\THzProcClass')

from THzData import THzData

basedir = 'C:\\Work\\Pin Reflector'
filename = 'Pin_mp_0.tvl'

gate = [[0, 4096], [-175, 200]]

data = THzData(filename, basedir, gate=gate, follow_gate_on=True, signal_type=1)

yy, xx = np.meshgrid(data.y, data.x, indexing='ij')



sol = curve_fit(gaussian_2d, xdata, data.c_scan)

plt.show()
