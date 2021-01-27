#========================================#
# ESSC 4510 Tutorial 06 Python Script
# Name:
# Student ID:
# Sample Script made by Benjamin Loi
#========================================#

# Import Necessary Packages
#========================================#
import pandas as pd # Reading Table
import numpy as np # Processing Array
import scipy.stats # Computing Statistic
import xarray as xr # Reading .nc File
import matplotlib.pyplot as plt # Drawing Graph
# import sklearn.utils
# Scikit-Learn is for machine learning,
# but it provides some utilities that can be useful.


#========================================#

# Ex 6.1
# June Temperature of Non-El Nino and El Nino years.
#========================================#
# Non-El Nino
pres_El = np.array([1009.5, 1010.7, 1009.3, 1009.9, 1009.3])
n_years_El = len(pres_El)
# El Nino
pres_nEl = np.array([1010.9, 1011.2, 1011.9, 1011.2, 1011.1,
                    1012.0, 1011.4, 1010.9, 1011.5, 1011.0,
                    1011.2, 1012.5, 1011.1, 1011.8, 1010.6])
n_years_nEl = len(pres_nEl)

#========================================#

# Carry out the Wilcoxon-Mann-Whitney test,
# by scipy.stats.mannwhitneyu(<data1>, <data2>).
# The first output is the U-statistic, smaller of the two,
# and the second one is the p-value.
# Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html.
#========================================#
U, p = scipy.stats.mannwhitneyu(pres_El, pres_nEl)
mu_U = n_years_El * n_years_nEl / 2
sigma_U = (mu_U / 6 * (n_years_El + n_years_nEl + 1)) ** (1/2)
z = (U - mu_U)/sigma_U


#========================================#

# Ex 6.2
# June Rainfall for the region.
#========================================#
rainfall = np.array([43, 10, 4, 0, 2,
                    np.nan, 31, 0, 0, 0,
                    2, 3, 0, 4, 15,
                    2, 0, 1, 127, 2])

#========================================#
rainfall = rainfall[~np.isnan(rainfall)] # drop the nan because it will cause problem in the following calculations

# Resampling to produce the bootstraps,
# either by np.random.choice(),
# or sklearn.utils.resample() (needed to import sklearn),
# both of them requires the argument "replace" set to True,
# and "size" specified.

# Produce at least 1000 bootstraps by for loop.
# At each iteration, evaluate the skewness,
# by scipy.stats.skew(), storing it into an array.
#========================================#
bootstrap_size = 1000 # Change as you like.

# one_line for loop: first make a sample, then calculate skewness, repeat.
skewness_bootstrap = np.array([
    scipy.stats.skew(np.random.choice(rainfall, len(rainfall), replace=True))
    for i in range(bootstrap_size)
])

#========================================#

# Sort the skewness array (can be skipped),
# and find the skewness corresponding to p = 0.025, 0.975,
# by np.percentile(), or otherwise.
# The range is then the desired 95% confidence interval.
#========================================#
output_range = [np.percentile(skewness_bootstrap, 2.5), np.percentile(skewness_bootstrap, 97.5)]

#========================================#