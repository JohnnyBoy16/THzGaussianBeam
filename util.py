import pdb

import numpy as np


def gaussian_2d(data, a, x0, y0, std_x, std_y):
    """
    2D Gaussian function that can be used to curve fit or whenever a Gaussian
    function is needed for something.
    :param data: A 2xn matrix of location where the data is to be calculated at.
        Y locations are in the first row and X locations are the 2nd
    :param a: The amplitude of the Gaussian
    :param x0: X center location
    :param y0: Y center location
    :param std_x: Standard Deviation of x
    :param std_y: Standard Deviation of y
    :return: The function value at the given (x, y) location
    """

    y, x = data

    f = a * np.exp(-((x-x0)**2/(2*std_x**2) + (y-y0)**2/(2*std_y**2)))

    return f
