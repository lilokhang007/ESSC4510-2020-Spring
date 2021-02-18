#========================================#
# ESSC 4510 Tutorial 08 Python Script
# Name: Li Lok Hang
# Student ID: 1155092891
# Sample Script made by Benjamin Loi
#========================================#

# Import Necessary Packages
#========================================#
import pandas as pd # Reading Table
import numpy as np # Processing Array
import scipy.stats # Computing Statistic
import matplotlib.pyplot as plt # Drawing Graph
import statsmodels.api as sm # Statistical Models



#========================================#

# Ex 8.2
# Read the .csv of HKO July Climate data you made in Ex 2.
#========================================#
HKO_data = pd.read_csv("HKO_data.csv",  # Replace the file name appropriately
                       index_col = 0) # Set the index column as the 0th column


#========================================#

# Do Multiple Linear Regression, using statsmodels.api.OLS().
#========================================#
# extract the variable x and y
y = HKO_data['Mean Temp.']
x = sm.add_constant(HKO_data[['Total rainfall', 'Mean pressure']])

# Fit the ols model
model = sm.OLS(y, x).fit()
print(model.summary2())
temp = model.predict([1, 324.1, 1004.2]) # answer to d
#========================================#

# Ex 8.3
# Initialize an Pandas table to store the time and mass.
#========================================#
exp_Sodium = pd.DataFrame({"Hours passed": [6, 8, 12, 24, 36, 48],
                    "Mass (g)": [75.8, 69.1, 57.3, 33.0, 18.8, 10.8]})



#========================================#

# Transform the data through linearization by np.log()
#========================================#
# extract the variable n and t
n = np.log(exp_Sodium['Mass (g)'])
t = sm.add_constant(exp_Sodium['Hours passed'])

#========================================#

# Carry out Linear Regression, again using statsmodels.api.OLS().
#========================================#
model = sm.OLS(n, t).fit()
n0, k = model.params # obtain the parameters fitted
#========================================#

# Recover the initial mass and half-life.
#========================================#
N0 = np.exp(n0) # recover N0
t_half = np.log(2) / (-k) # note that the negative sign is omitted


#========================================#