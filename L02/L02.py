#========================================#
# ESSC 4510 Tutorial 02 Python Script
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



#========================================#

# Ex 2.1
# Read HKO climate data using the same way as in Ex 1.1.
# You should have concatenate the data into an n x 3 array.
#========================================#
HKO_data = pd.read_csv("HKO_data.csv",  # Replace the file name appropriately
                       index_col = 0) # Set the index column as the 0th column
print(HKO_data)


#========================================#

# Evaluate lag-k auto-correlation,
# by extracting the earliest and latest n-k data as two arrays,
# and use scipy.stats.pearsonr(<>, <>)[0] with the arrays as arguments,
# since lag-k auto-correlation is essentially the correlation
# between the earliest and latest n-k data.
# Use for loop to evaluate from k = 1 to 5
#========================================#
n = len(HKO_data)
lag_k_auto_correlation = [] # Initialize an empty list to put the values down
for k in np.arange(1, 5+1):
    lag_k_auto_correlation.append(
        round(scipy.stats.pearsonr(
            HKO_data['Mean Temp.'].iloc[:(n-k)].values, # the earlist n-k data in array format
            HKO_data['Mean Temp.'].iloc[k:].values # the latest n-k data in array format
        )[0],3) # Use [0] because scipy returns an array as default
    )


#========================================#


# Ex 2.2
# Prepare a figure with 2x2 panels.
# With 3 variables, there will be 3C2 = 3 scatter plots.
#========================================#
n_var = 3
# Initialize figsize large enough so that things do not squash together
fig, axes = plt.subplots(n_var-1, n_var-1, figsize=(12,12))
plt.show()
#========================================#

# Use <ax>.scatter(x,y) to produce a scatter plot on a particular subplot.
# Add legends, titles, axis labels for each subplot.
# Double for loops can be used to go through all the panels as shown below.
#========================================#
for i in np.arange(n_var-1):
    for j in np.arange(i+1):
        x = HKO_data.columns[i] # extract variable 'x'
        y = HKO_data.columns[j] # extract variable 'y'
        axes[i,j].scatter(HKO_data[x], HKO_data[y]) # make a scatter plot
        axes[i,j].legend(['Data']) # add legend
        axes[i,j].set_xlabel(x) # add xlabel
        axes[i,j].set_ylabel(y) # add ylabel
        axes[i,j].set_title('Scatter plot of monthly mean {} and {} \n'
                            'of the station HKO between 2001 and 2020'.format(x, y)) # add title
#========================================#

# (optional) Remove the unused axes by <ax>.set_visible(False).
#========================================#
axes[0,1].set_visible(False)
plt.tight_layout() # Prevent things overlap together
#========================================#

# Output the figure as .png format.
#========================================#
fig.savefig("Ex_2_2.png")



#========================================#

# Ex 2.3
# Reference: https://realpython.com/numpy-scipy-pandas-correlation-python/,
# and https://benalexkeen.com/correlation-in-python/.
# For Pearson correlation, np.corrcoef() is sufficient.
#========================================#

Pearson_corr = np.corrcoef(HKO_data, rowvar=False) # calculate Pearson correlation
Pearson_corr = np.round(Pearson_corr, 3) # round to 3 dp

#========================================#

# For Spearman correlation, create a 3x3 array,
# and use double for loop to run all the combinations.
# At each iteration, fill in the array element,
# with corresponding Spearman correlation computed by scipy.stats.spearmanr(x,y)[0].
#========================================#
Spearman_corr = np.zeros([3,3]) # Create an empty array with all entries 0
for i in np.arange(n_var):
    for j in np.arange(n_var):
        x = HKO_data[HKO_data.columns[i]] # extract variable 'x'
        y = HKO_data[HKO_data.columns[j]] # extract variable 'y'
        Spearman_corr[i, j] = scipy.stats.spearmanr(x,y)[0] # calculate Spearman correlation

Spearman_corr = np.round(Spearman_corr, 3)

#========================================#