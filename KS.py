"""
Functions to calculate KS and percentiles
"""
import numpy as np
import pandas as pd

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
    """return 10, 50 and 90 percentiles"""

    x_values = np.arange(0, data["rho"].max() + dx, dx)
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


# def confidence_intervall(_data):
#     density = []
#     velocity = []
#     for _, df in _data.items():
#         density.append(df["rho"])
#         velocity.append(df["velocity"])

#     data = np.column_stack((density, velocity))
#     num_samples = 1000
#     bootstrapped_stats = np.empty((num_samples, 2))

#     # Perform bootstrapping
#     for i in range(num_samples):
#         # Resample the data with replacement
#         resampled_data = data[np.random.choice(len(data), size=len(data), replace=True)]
#         bootstrapped_stats[i, 0] = np.mean(resampled_data[:, 0])  # Mean of density
#         bootstrapped_stats[i, 1] = np.mean(resampled_data[:, 1])  # Mean of velocity
#         # Calculate the statistics of interest on the resampled data

#     # Calculate the confidence interval using percentiles
#     confidence_interval = np.percentile(bootstrapped_stats, [2.5, 97.5], axis=0)

#     # Print the confidence interval for density and velocity
#     print("Confidence Interval for Density:", confidence_interval[0])
#     print("Confidence Interval for Velocity:", confidence_interval[1])
#     return confidence_interval
