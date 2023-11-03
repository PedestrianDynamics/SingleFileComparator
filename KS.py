"""
Functions to calculate KS and percentiles
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# from scipy import stats


def CalcBiVarCDF(x, y, xGrid, yGrid):
    """
    Calculate the bivariate CDF of two given input signals on predefined grids.
    input:
      - x: array of size n1
      - y: array of size n2
      - xGrid: array of size m1
      - yGrid: array of size m2
    output:
      - CDF2D: matrix
    """
    nPoints = np.size(x)
    xGridLen = np.size(xGrid)
    yGridLen = np.size(yGrid)
    CDF2D = np.zeros([xGridLen, yGridLen])

    for i in range(xGridLen):
        for j in range(yGridLen):
            CDF2D[i, j] = np.sum((x <= xGrid[i]) & (y <= yGrid[j]))

    CDF2D = CDF2D / nPoints
    return CDF2D


def CDFDistance(x1, y1, x2, y2):
    """
    For two input 2D signals calculate the distance between their CDFs.
    input:
      - x1: array of size n
      - y2: array of size n
      - x2: array of size m
      - y2: array of size m
    output:
      - KSD: not negative number
    """
    xPoints = 10
    yPoints = 10
    x = np.hstack((x1, x2))
    xCommonGrid = np.linspace(np.min(x), np.max(x), xPoints)
    y = np.hstack((y1, y2))
    yCommonGrid = np.linspace(np.min(y), np.max(y), yPoints)
    CDF1 = CalcBiVarCDF(x1, y1, xCommonGrid, yCommonGrid)
    CDF2 = CalcBiVarCDF(x2, y2, xCommonGrid, yCommonGrid)
    #    KSD = np.linalg.norm(CDF1-CDF2); # Frobenius norm (p=2)
    KSD = np.max(np.abs(CDF1 - CDF2))
    # Kolmogorov-Smirnov distance (p=inf)
    return KSD


def percentiles(data: pd.DataFrame, dx: float, N: int):
    """
    Calculate the 10th, 50th, and 90th percentiles of 'velocity' for subsets of data within intervals of 'rho'.

    Parameters:
    data (pd.DataFrame): The input data containing 'rho' and 'velocity' columns.
    dx (float): The width of each 'rho' interval.
    N (int): The minimum number of data points required within a 'rho' interval to calculate percentiles.

    Returns:
    tuple: A tuple containing lists of the 10th, 50th, and 90th percentile values of 'velocity', and the corresponding 'rho' interval values.

    Note: The function will return percentile values only for the 'rho' intervals with at least N data points.
    """
    if "rho" not in data.columns or "velocity" not in data.columns:
        raise ValueError("Input DataFrame must contain 'rho' and 'velocity' columns.")

    x_values = np.arange(data["rho"].min(), data["rho"].max() + dx, dx)
    v10_values = []
    v50_values = []
    v90_values = []

    for x in x_values:
        data_points = data[(data["rho"] >= x) & (data["rho"] < x + dx)]

        if len(data_points) < N:
            break

        v10 = np.percentile(data_points["velocity"], 10)
        v50 = np.percentile(data_points["velocity"], 50)
        v90 = np.percentile(data_points["velocity"], 90)

        v10_values.append(v10)
        v50_values.append(v50)
        v90_values.append(v90)

    return v10_values, v50_values, v90_values, x_values


# Example usage:
# df = pd.DataFrame({
#     'rho': np.random.rand(1000),
#     'velocity': np.random.rand(1000)
# })
# print(percentiles(df, 0.1, 50))
