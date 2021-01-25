#========================================#
# ESSC 4510 Tutorial 04 Python Script
# Name: Li Lok Hang
# Student ID: 1155092891
# Sample Script made by Benjamin Loi
#========================================#

# Import Necessary Packages
#========================================#
import pandas as pd # Reading Table (not needed)
import numpy as np # Processing Array
import scipy.stats # Computing Statistic
import xarray as xr # Reading .nc File
import matplotlib.pyplot as plt # Drawing Graph


###
# Sorry that I have re-written some of the parts to reduce code duplication,
# please check my comments below specifically
# but otherwise the code will produce the required answers and pics
###

#========================================#

# Ex 4.2
# Get the temperature and rainfall data from the given .nc file.
# The data are the same as Tutorial 3.
#========================================#
data_4 = xr.open_dataset("temp_precip_city_China_July_1998-2015.nc")
prec_BJ, prec_WH, prec_HK = data_4["prec_BJ"], data_4["prec_WH"], data_4["prec_HK"]
temp_BJ, temp_WH, temp_HK = data_4["temp_BJ"], data_4["temp_WH"], data_4["temp_HK"]



# I moved the pre-processing mean and std part, it can be absorbed in .fit()
#========================================#

# Part (a)
#================================================================================#
# START OF FUNCTION
#================================================================================#
# I removed the variable 'time_series' as it is not necessary and can reduce code duplication
# Basically if one variable is dependent on another variable, it is sufficient to put only one in the function args
# Therefore, you are left with only var_name, other can all be absorbed

def hist_and_fit(var_name):
    """
        A plotting function to produce the required graph.
        Apply this function for all the six sets of data.
        var_name: the name of the variable.
    """


    # Create an empty figure.
    #========================================#
    fig = plt.figure()
    ax = fig.add_subplot()

    #========================================#

    # Plot the histogram by plt.hist().
    # You may want to scale y-axis to fraction,
    # by setting density = True.
    # Since we read the data into an xarray Dataset,
    # we need to convert it into an numpy array,
    # by np.array() to avoid bugs.
    #========================================#
    vals = data_4[var_name].values # get the time series in np.array
    plt.hist(vals, density=True) # plot the histogram
    x = np.linspace(min(vals), max(vals), 100)[1:] # discard the first point to prevent "exploding gamma"

    legend_ls = ['Gaussian Distribution with mu = {}, sigma = {}',
                 'Gamma Distribution with alpha = {}, beta = {}'] # Initialize a list with legend strings

    args_ls = []
    # I rewritten your code, to do what can be done into a few lines of code
    for i, dist in enumerate((scipy.stats.norm, scipy.stats.gamma)): # loop required distributions
        args = dist.fit(vals) # get parameter values by directing ".fitting" the values
        ax.plot(x, dist.pdf(x, *args)) # pass args to pdf and plot

        # Note: I follow beta's definition in the docs, which is slightly different from the slides
        legend_ls[i] = legend_ls[i].format(*args) # Put args into legends
        args_ls.append(args)

    #========================================#

    # Add legends, titles, axis labels, as well as 
    # the values of the fitted parameters, either
    # in a textbox, using ax.text(), or simply in the legends.
    # https://matplotlib.org/3.1.1/gallery/recipes/placing_text_boxes.html
    #========================================#
    ax.legend(legend_ls) # set legend
    ax.set_title('Fitted distributions for variable {}'.format(var_name)) # set title
    ax.set_ylabel('Probability density') # set ylabel

    # Define units first
    if 'prec' in var_name:
        units = 'mm'
    elif 'temp' in var_name:
        units = '°C'
    ax.set_xlabel('Value of variable {} ({})'.format(var_name, units)) # set xlabel with units

    #========================================#

    # Output the figure as .png format.
    #========================================#
    fig.savefig("Ex_4_2a_" + var_name + ".png")

    #========================================#
    return args_ls # return for next function use

#================================================================================#
# END OF FUNCTION
#================================================================================#

# Call the function hist_and_fit() for all the six cases.
#========================================#
args_ls_ls = [] # initialize an empty list, this saves args for next function use
for item in ['prec_BJ', 'prec_WH', 'prec_HK', 'temp_BJ', 'temp_WH', 'temp_HK']:
    args_ls_ls.append(hist_and_fit(item))

#========================================#

# Part (b)
#================================================================================#
# START OF FUNCTION
#================================================================================#
def QQ_plot(var_name, args):
    """
        I rewritten the input of this function, to avoid code duplication,
        *args: parameter list of the fitted distribution
    """

    # Define fit inside the function, as it is dependent to var_name
    if 'prec' in var_name:
        fit = "Gamma"
    elif 'temp' in var_name:
        fit = "Gaussian"

    # Create an empty figure.
    #========================================#
    fig = plt.figure()
    ax = fig.add_subplot()

    #========================================#

    # Extract the 1st-99th percentiles from the observation,
    # by np.percentile(<data>, <percentiles>).
    #========================================#
    vals = data_4[var_name].values  # get the time series in np.array
    per_obs = np.percentile(vals, np.arange(1,100))

    #========================================#

    # Compute the 1st-99th percentiles from the fitted distribuition,
    # either for Gaussian or Gamma.
    #========================================#
    # Convert the percentiles, or ranks,
    # to cumulative probabilities by formula like Tukey.
    rank = np.arange(1,100)
    prob_fit = (rank - 1/3) / (100 + 1/3)
    if fit == "Gaussian":
        # Compute the inverse of CDF to obtain the theoretical percentiles.
        per_fit = scipy.stats.norm.ppf(prob_fit, *(args[0]))
    elif fit == "Gamma":
        # Similar applies for Gamma distribution.
        per_fit = scipy.stats.gamma.ppf(prob_fit, *(args[1]))

    #========================================#

    # Draw the diagonal.
    #========================================#
    diag = np.linspace(min(vals), max(vals), 100) # Adjust the numbers yourself.
    ax.plot(diag, diag)



    #========================================#

    # Draw the scatter plot from both observed and fitted percentiles,
    # by ax.scatter(<fit>, <obs>).
    #========================================#
    ax.scatter(per_fit, per_obs)


    #========================================#

    # Add legends, titles, axis labels.
    #========================================#
    ax.legend(['Diagonal','{}'.format(fit)]) # set legend
    ax.set_title('QQ-plot for variable {} with {} distribution'.format(var_name, fit)) # set title

    # Define units first
    if 'prec' in var_name:
        units = 'mm'
    elif 'temp' in var_name:
        units = '°C'

    ax.set_ylabel('Observed values ({}) of variable {}'.format(units,var_name)) # set ylabel
    ax.set_xlabel('Inverse of CDF ({})'.format(units))  # set xlabel

    #========================================#

    # Output the figure as .png format.
    #========================================#
    fig.savefig("Ex_4_2b_" + var_name + ".png")



    #========================================#

#================================================================================#
# END OF FUNCTION
#================================================================================#

# Call the function QQ_plot() for all the six cases.
#========================================#
for i, item in enumerate(['prec_BJ', 'prec_WH', 'prec_HK', 'temp_BJ', 'temp_WH', 'temp_HK']):
    QQ_plot(item, args_ls_ls[i]) # pass the args back to the function



#========================================#

# Ex 4.3
#================================================================================#
# START OF FUNCTION
# Alternatively, you can use some functions that
# compute auto-corrlation directly, and
# no need to create a function as below.
#================================================================================#

def get_x_y(city_name, k):
    # I took this function out of the function you designed below, to reduce code duplication, as this part will be used later on as well
    temp = globals()['temp_' + city_name]
    prec = globals()['prec_' + city_name]

    # function to extract the earliest n-k data and latest n-k data respectively
    n = len(temp)
    if k >= 0:
        x = temp[:(n-k)]
        y = prec[k:]
    else: #handles negative k case
        x = temp[(-k):]
        y = prec[:k]
    return x, y

def lag_corr(city_name, k):
    """
        Calculate the lagged correlation between two sets of data.
        city_name: used to extract temp and prec
        k: time lag.
    """

    # Evaluate lag-k correlation, 
    # by extracting the earliest n-k data from the first time series,
    # and the latest n-k data from the second time series.
    # Use scipy.stats.pearsonr()[0] to obtain the number.
    #========================================#
    x, y = get_x_y(city_name, k) # get the earliest n-k data and latest n-k data respectively
    #========================================#

    # Output the correlation by return(<value>).
    #========================================#
    return scipy.stats.pearsonr(x, y)[0]


    #========================================#

#================================================================================#
# END OF FUNCTION
#================================================================================#

# Use the function lag_corr() to calculate from k = -4 to 4.
#========================================#
cities = ['BJ', 'HK', 'WH'] # define the cities
k_generator = range(-4, 4+1) # define the values of k
for city in cities:
    fig = plt.figure() # make a new plot
    ax = fig.add_subplot()
    lag_corr_ls = [] # initialize a list to store the correlations
    for k in k_generator: # loop over different value of ks
        lag_corr_ls.append(lag_corr(city, k)) # store the correlation in list
    ax.stem(k_generator, lag_corr_ls) # make the stem plot
    ax.legend(['City {}'.format(city)]) # make the legends
    ax.set_xlabel('k') # make the xlabels
    ax.set_ylabel('Lagged Correlation rk') # make the y labels
    ax.set_title('Temperature and Rainfall lagged correlation for city {}'.format(city)) # make the titles
    plt.show()


#========================================#

# Plot the scatter plot for the case with largest r_k.
#========================================#
# For largest, I suspect you mean absolutely largest (|r_k|), because strongly negatively correlated is also a strong correlation
    fig = plt.figure() # make a new plot
    ax = fig.add_subplot()
    k_with_largest_r_k = k_generator[np.argmax(np.abs(lag_corr_ls))] # k where |r_k| is largest
    ax.scatter(*get_x_y(city, k_with_largest_r_k)) # python syntax '*' (unbracketing) to get x and y as two parameters
    ax.legend(['City {}'.format(city)]) # make the legends
    ax.set_xlabel('Temperature (°C)') # make the xlabels
    ax.set_ylabel('Rainfall (mm)') # make the y labels
    ax.set_title('Temperature and Rainfall , with a lagged period k = {}'.format(k_with_largest_r_k)) # make the titles
    plt.show()
#========================================#