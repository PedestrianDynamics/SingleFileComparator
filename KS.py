import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import numpy as np
import plotly.graph_objs as go

from scipy import stats

def CalcBiVarCDF(x,y,xGrid,yGrid):
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
    nPoints = np.size(x);
    xGridLen = np.size(xGrid);
    yGridLen = np.size(yGrid);
    CDF2D = np.zeros([xGridLen,yGridLen]);

    for i in range(xGridLen):
        for j in range(yGridLen):
            CDF2D[i,j] = np.sum((x <= xGrid[i]) & (y <= yGrid[j]));
                     

    CDF2D = CDF2D / nPoints;
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
    xPoints = 10;
    yPoints = 10;
    x = np.hstack((x1, x2))
    xCommonGrid = np.linspace(np.min(x), np.max(x), xPoints);
    y = np.hstack((y1, y2))
    yCommonGrid = np.linspace(np.min(y), np.max(y), yPoints);
    CDF1 = CalcBiVarCDF(x1,y1,xCommonGrid,yCommonGrid);
    CDF2 = CalcBiVarCDF(x2,y2,xCommonGrid,yCommonGrid);
#    KSD = np.linalg.norm(CDF1-CDF2); # Frobenius norm (p=2)
    KSD = np.max(np.abs(CDF1-CDF2)); # Kolmogorov-Smirnov distance (p=inf)
    return KSD

def CDFDistance2(rho1, v1, rho2, v2, rho_min, rho_max):
    """
    For two input 2D signals calculate the "distance" between their CDFs - 
    averaged (over density bins) distance between two 1D CDFs of speed 
    calculated for specific density bin. 
    input: 
      - rho1: density array of size n
      - v1: speed array of size n
      - rho2: density array of size m
      - v2: speed array of size m
      - rho_min: lower boundary of density value considered (used for bins creation)
      - rho_max: upper boundary of density value considered (used for bins creation)
    output: 
      - KSD: not negative number from 0 to 1
    """
    EMPTY = -1; 
    nBins = 20; # Now it is not obvious which value to get
    bins = np.linspace(rho_min, rho_max, nBins+1);
#    dist1D = EMPTY*np.ones((1,nBins));
    dist1D = [] ;
    
    for iBin in range(nBins):
        v1_b = v1[(rho1 >= bins[iBin]) * (rho1 <= bins[iBin+1])];        
        v2_b = v2[(rho2 >= bins[iBin]) * (rho2 <= bins[iBin+1])];
        st.info(iBin)
        st.info(f"{v1_b}")
        st.info(f"{v2_b}")
        #if ((len(v1_b) > 0) and (len(v2_b) > 0)):
        [ks2stat, _] = stats.ks_2samp(v1_b, v2_b)
        print(iBin, ks2stat)
        dist1D.append(ks2stat)
    
    #KSD = np.sum(dist1D[dist1D != EMPTY])/len(dist1D[dist1D != EMPTY]);
    KSD = np.sum(dist1D)/len(dist1D)
    return KSD



