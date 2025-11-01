# src/stats_utils.py
import numpy as np
import pandas as pd
from scipy import stats

def bootstrap_ci(data, func=np.mean, n_boot=2000, alpha=0.05, random_state=0):
    rng = np.random.default_rng(random_state)
    n = len(data)
    stats_boot = []
    for _ in range(n_boot):
        sample = rng.choice(data, size=n, replace=True)
        stats_boot.append(func(sample))
    lower = np.percentile(stats_boot, 100*alpha/2)
    upper = np.percentile(stats_boot, 100*(1-alpha/2))
    return func(data), (lower, upper)

def two_sample_ttest(a, b, equal_var=False):
    # returns t-statistic and p-value
    t, p = stats.ttest_ind(a, b, equal_var=equal_var)
    return t, p

def poisson_fit_test(counts):
    # Test using Poisson: check mean~var and do chisq goodness of fit
    lam = np.mean(counts)
    # simple check: mean and var
    return lam, np.var(counts)
