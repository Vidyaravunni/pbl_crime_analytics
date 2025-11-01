# src/dist_fit.py
from scipy import stats
import numpy as np

def fit_poisson(counts):
    lam = np.mean(counts)
    # For goodness-of-fit, compare observed vs expected frequencies
    # user can plot histogram vs Poisson pmf
    return lam

def fit_normal(series):
    mu, sigma = np.mean(series), np.std(series, ddof=1)
    return mu, sigma
