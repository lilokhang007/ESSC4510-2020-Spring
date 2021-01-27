#========================================#
# ESSC 4510 Tutorial 05 Python Script
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



#========================================#

# Ex 5.1
# June Temperature of Non-El Nino and El Nino years.
#========================================#
# Non-El Nino
temp_El = np.array([26.1, 24.8, 26.4, 26.6, 26.8])
n_years_El = len(temp_El)
# El Nino
temp_nEl = np.array([24.5, 24.5, 24.1, 24.3, 24.9, 23.7, 23.5, 
                    24.0, 24.1, 23.7, 24.3, 24.6, 24.8, 24.4, 25.2])
n_years_nEl = len(temp_nEl)



#========================================#

# Evaluate the combined variance for the sample means,
# which is (s_1^2/n_1 + s_2^2/n_2).
#========================================#
s_El = np.std(temp_El)
s_nEl = np.std(temp_nEl)
std_combined = (s_El ** 2 / n_years_El + s_nEl ** 2 / n_years_nEl) ** (1/2)

#========================================#

# Compute the p-value by scipy.stats.ttest_ind(<data1>, <data2>).
# Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html.
# The argument "equal_var" should be set to a suitable boolean value.
# The second output is the required p-value.
#========================================#
t_ex5_1, p_value_ex5_1 = scipy.stats.ttest_ind(temp_El, temp_nEl)


#========================================#

# Calculate the 95% confidence interval for the t-test.
# The null hypothesis has a mean of zero for Delta x = x_1 - x_2.
# Plugging the combined std computed above,
# and ddof = min(n_1, n_2) - 1 into,
# the function scipy.stats.t.interval(0.95, <ddof>, 0, <pooled std>).
# Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html.
#========================================#
ddof = min(n_years_El, n_years_nEl) - 1
interval = scipy.stats.t.interval(0.95, ddof, 0, std_combined)


#========================================#

# Ex 5.2
# (Important) Please state clearly in your pdf, 
# which test, (a) or (b), is a paired test.
# This is important in deciding the form of t-test used.

# Get the temperature data from the given .nc file.
# The data are the same as Tutorial 3 and 4.
#========================================#
data_5 = xr.open_dataset("temp_precip_city_China_July_1998-2015.nc")
#========================================#

# Select the appropriate data.
# For extracting data from specific time period,
# please take a look at the sample script for tutorial 3.
#========================================#

def query_by_month(dataset, month, year):
    dates = dataset.date # get the dates
    mod = (dates % 1000).values  # taking modulus
    div = dates // 10000 # division
    bool_array = (month * 100 < mod) & (mod < (month + 1) * 100) & (div == year) # save as boolean array
    return dataset[bool_array] # return selected items

temp_BJ_7_2015 = query_by_month(data_5['temp_BJ'], 7, 2015)
temp_HK_7_1998 = query_by_month(data_5['temp_HK'], 7, 1998)
temp_HK_7_2015 = query_by_month(data_5['temp_HK'], 7, 2015)

#========================================#

# Use scipy.stats.ttest_ind() for two independent samples,
# and scipy.stats.ttest_rel() for the paired test,
# to retrieve the p-value, in a way similar to the last part.
#========================================#

t_ex5_2a, p_value_ex5_2a = scipy.stats.ttest_rel(temp_BJ_7_2015, temp_HK_7_2015)
t_ex5_2b, p_value_ex5_2b = scipy.stats.ttest_ind(temp_HK_7_1998, temp_HK_7_2015)

#========================================#

# Ex 5.3
# Download and read the HKO rainfall data.
# Remember to upload the related dataset in submission.
#========================================#
df = pd.read_csv('HKO_dataset.csv', index_col=0)
HKO_rainfall = df['Total annual rainfall']

#========================================#

# Evaluate the mean and standard deviation of the data,
# by np.mean(), np.std(), or otherwise.
#========================================#
mean_rainfall = np.mean(HKO_rainfall)
std_rainfall = np.std(HKO_rainfall)


#========================================#

# Fitting Gaussian and Gamma distribution into the data,
# by comparing the parameters like in Tutorial 4.
#========================================#
# I directly call ".fit()", see below

#========================================#
fig = plt.figure() # create figure
ax = fig.add_subplot()

#========================================#
# Prepare everything that needs to be in the plot
x = np.linspace(min(HKO_rainfall), max(HKO_rainfall), 100)[1:] # predefine x values
legend_ls = ['Gaussian Distribution with mu = {}, sigma = {}',
             'Gamma Distribution with alpha = {}, location = {} and beta = {}']  # Initialize a list with legend strings
args_ls = [] # Initialize a list to put in parameters of distributions

#========================================#

# Plot the histogram, by plt/ax.hist() 
# and the two fitted distribution,
# by scipy.stats.norm.pdf() and scipy.stats.gamma.pdf(),
# then plt/ax.plot()
# just like in Tutorial 4.
# To facilitate the Chi-square fitting,
# it is better to save the output returned from calling hist(),
# which is the frequency and range of each bin.
# This can be done like hist_freq = plt.hist()
# Documentation: https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.pyplot.hist.html.
#========================================#
n_bins = 10 # set number of bins
hist_freq = plt.hist(HKO_rainfall, bins=n_bins) # observed frequency
exp_freq = [] # initialize a list to hold expected frequency
bin_diff = (max(HKO_rainfall) - min(HKO_rainfall)) / n_bins # calculate bin width
unnormalizing_factor = bin_diff * len(HKO_rainfall) # 'unnormalize' from 'probability density' to actual quantities

for i, dist in enumerate((scipy.stats.norm, scipy.stats.gamma)):  # loop required distributions
    args = dist.fit(HKO_rainfall)  # get parameter values by directing ".fitting" the values
    ax.plot(x, dist.pdf(x, *args) * unnormalizing_factor)  # pass args to pdf and plot
    legend_ls[i] = legend_ls[i].format(*(round(arg,1) for arg in args))  # Put args into legends
    args_ls.append(args)

#========================================#

# We have the observed bin frequency from plt.hist().
# Now we want to compute the expected frequency
# of the two fitted distributions.
# This can be done by extracting the probability within
# the range of each bin, by
# scipy.stats.norm.cdf(<upper limit>) - scipy.stats.norm.cdf(<lower limit>),
# with appropriate parameters for the mean and std.
# Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html.
# Similar for Gamma distribution.

# Notice that they are vectorized functions, which
# take an array and output another array.
# So we can utilize whole array operations, and
# need not to use for loops, which are acceptable anyways.
#========================================#
    exp_freq.append(
        (dist.cdf(hist_freq[1][1:], *args) - dist.cdf(hist_freq[1][:-1], *args)) * len(HKO_rainfall)
    )
#========================================#

# Carry out the Chi-square goodness of fit test,
# by scipy.stats.chisquare(<f_obs>, <f_exp>, <ddof>).
# Degree of freedom is no. of bins − no. of parameters fit − 1.
# no. of parameters fit are two in both cases.
# Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html.
#========================================#
    print(scipy.stats.chisquare(hist_freq[0], exp_freq[i], n_bins - 3))


#========================================#

# Finally, decorate the figure
ax.legend(legend_ls) # set legend
ax.set_title('Fitted distributions for HKO annual rainfall from 1961 - 2020')
ax.set_ylabel('Number of years') # set ylabel
ax.set_xlabel('HKO annual rainfall (mm)') # set xlabel with units
