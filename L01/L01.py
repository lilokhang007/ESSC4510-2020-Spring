#========================================#
# ESSC 4510 Tutorial 01 Python Script
# Name: Li Lok Hang
# Student ID: 1155092891
# Sample Script made by Benjamin Loi
#========================================#

# Import Necessary Packages
#========================================#
import pandas as pd # Reading Table
import numpy as np # Processing Array
import matplotlib.pyplot as plt # Drawing Graph



#========================================#

# Copy the whole rows of HKO rainfall data from 2001 to 2020,
# and paste into an Excel spreadsheet, (important) saved as a .csv file.

# Ex 1.1
# Read the .csv file into a table using pandas.
#========================================#
# Read csv to variable HKO_data_rainfall
HKO_data_rainfall = pd.read_csv("HKO_data_rainfall.csv",  # Replace the file name appropriately
                                names=["Year","Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]) # Define the header names
print(HKO_data_rainfall)

#========================================#

# Select July rainfall data for the required years.
#========================================#
years = HKO_data_rainfall["Year"]
# from 2001 - 2020
required_years = np.logical_and(2001 <= years, years <= 2020) # Replace the numbers with suitable years
rainfall_Jul = HKO_data_rainfall["Jul"]
rainfall_Jul_period = rainfall_Jul[required_years]
print(rainfall_Jul_period)



#========================================#

# Compute the median and mean,
# by using the functions np.mean(<>), np.median(<>), or
# the methods <>.mean(), <>.median() on Pandas table.
#========================================#
# roundings to prevent precision errors by floats
rainfall_Jul_mean = round(rainfall_Jul_period.mean(), 2) # calculate rounded mean
rainfall_Jul_median = rainfall_Jul_period.median() # calculate median
print(rainfall_Jul_mean, rainfall_Jul_median)

#========================================#

# Ex 1.2
# Copy and modify the first part of Ex 1.1 accordingly to process pressure data.
HKO_data_pressure = pd.read_csv("HKO_data_pressure.csv",  # Another csv filename
                                names=["Year","Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]) # Define the header names
years = HKO_data_pressure["Year"]
required_years = np.logical_and(2001 <= years, years <= 2020)
pressure_Jul = HKO_data_pressure["Jul"]
pressure_Jul_period = pressure_Jul[required_years]
print(pressure_Jul_period)

# Calculate the median, upper and lower quantiles first,
# by using np.median(<>), np.percentile(<>, <25/75>) (or np.quantile(<>, <0.25/0.75>)),
# or <>.median(), <>.quantile(<0.25/0.75>) on Pandas table.
#========================================#
percentiles = [0.25, 0.5, 0.75] # Put the requested percentiles in list
pressure_Jul_lq, pressure_Jul_median, pressure_Jul_uq = [
    round(pressure_Jul_period.quantile(percentile), 2) for percentile in percentiles
] # Get the required median, upper and lower quantiles using list comprehension
print(pressure_Jul_lq, pressure_Jul_median, pressure_Jul_uq)


#========================================#

# Subsequently calculate the IQR - Inter-quantile Range,
# and MAD - Median Absolute Deviation.
# np.abs() is useful for computing MAD.
#========================================#
pressure_Jul_IQR = pressure_Jul_uq - pressure_Jul_lq # Calculate IQR = UQ - LQ
# Calculate rounded median = median(abs(x_i - median(x))), where x_i is individual elements of x
pressure_Jul_MAD = round(np.median(np.abs(pressure_Jul_period - pressure_Jul_median)), 2)
print(pressure_Jul_IQR, pressure_Jul_MAD)


#========================================#

# Standard deviation is simply obtained by
# np.std(<>, ddof=1), where ddof=1 indicates the sample standard deviation,
# by default ddof=0 which refers to the population standard deviation.
# Alternatively, apply <>.std() on Pandas table, note that however
# by default ddof=1 so the output is already the sample standard deviation.
#========================================#
pressure_Jul_std = round(pressure_Jul_period.std(),2) # Calculate standard deviation
print(pressure_Jul_std)

#========================================#

# Ex 1.3
# Create an empty figure.
#========================================#
fig = plt.figure()
ax = fig.add_subplot()



#========================================#

# Sort the data by using np.sort(<>) and store the result in another variable.
#========================================#
pressure_Jul_period_sorted = np.sort(pressure_Jul_period) # Sort the variable


#========================================#

# Prepare the cumulative probabilities using one of the formula provided in the slide.
# Example: Tukey, Weibull.
#========================================#
n_years = 20 # No. of years, please modify accordingly.
cumul_prob = ((np.arange(n_years)+1)-0.5)/n_years # Here Hazen formula is used.
# (important) Please use other formula by changing the expression.
cumul_prob = ((np.arange(n_years)+1)-1/3)/(n_years+1/3) #Tukey formula


#========================================#

# Plot the CFD - Cumulative Frequency Distribution by using plt.step(<sorted values>, <cumulative probabilities>).
# Usage: https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.pyplot.step.html
#========================================#
plt.step(pressure_Jul_period_sorted, cumul_prob) # Make a step plot out of cumulative probability


#========================================#

# Plot the histogram on the same figure by plt.hist().
# Usage: https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.pyplot.hist.html
#========================================#
# Normalize the histogram to total area = 1, so that the y-axis make sense,
# as it is requested to put ECDF and histogram together
plt.hist(pressure_Jul_period, density=True)


#========================================#

# (important) You need to add legends, titles, axis labels,
# as well as tweak some parameters in the plotting function to refine the plots.
#========================================#
# Add title, labels and legends
plt.title("Empirical cumulative distribution function and \n"
          "histogram of monthly mean pressure in July between 2001 and 2020 of HKO")
plt.xlabel("Monthly Mean Pressure in July (hpa)")
plt.ylabel("Frequency")
plt.legend(["Cumulative Frequency", "Normalized Histogram"])
plt.show()

#========================================#

# Output the figure as .png format.
#========================================#
fig.savefig("Ex_1_3.png")



#========================================#
