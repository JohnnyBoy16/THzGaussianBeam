import pdb

import numpy as np


def gaussian_2d(data, A, x0, y0, std_x, std_y):

    y, x = data

    f = A * np.exp(-((x-x0)**2/(2*std_x**2) + (y-y0)**2/(2*std_y**2)))

    return f
