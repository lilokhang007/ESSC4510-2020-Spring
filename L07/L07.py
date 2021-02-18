#========================================#
# ESSC 4510 Tutorial 07 Python Script
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

# Ex 7.1
# Store the June temperature and pressure as a Pandas table.
#========================================#
data_7 = pd.DataFrame({"Temp": [26.1, 24.5, 24.8, 24.5, 24.1, 24.3, 26.4, 24.9, 23.7, 23.5,
                    24.0, 24.1, 23.7, 24.3, 26.6, 24.6, 24.8, 24.4, 26.8, 25.2],
                    "Pres": [1009.5, 1010.9, 1010.7, 1011.2, 1011.9, 1011.2, 1009.3, 1011.1, 1012.0, 1011.4,
                    1010.9, 1011.5, 1011.0, 1011.2, 1009.9, 1012.5, 1011.1, 1011.8, 1009.3, 1010.6]})
print(data_7)



#========================================#

# Carry out Linear Regression on the data, and compute R^2.
# There are many packages providing functions for that, such as 
# scipy.stats.linregress() and sklearn.linear_model.LinearRegression().
# For statistical purpose, please use statsmodels.api.OLS(),
# as it provides a summary for many important quantities,
# answers for some parts can be easily read from.
# Reference: https://datatofish.com/statsmodels-linear-regression/.
# s_e^2 can be checked via <model>.mse_resid,
# or using .summary2() instead and check the scale entry.
# The whole documentation can be viewed at 
# https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.html#statsmodels.regression.linear_model.RegressionResults.
# Extract the required values from the instance, no hard coding is allowed!
#========================================#
Y = data_7["Temp"]
X = sm.add_constant(data_7["Pres"]) # Adding the intercept term.
ols = sm.OLS(Y, X).fit() # fit the model
t = ols.tvalues['Pres'] #7.1C
r2 = ols.rsquared #7.1D
s_e2 = ols.mse_resid # get the required variance
s_e = s_e2 ** (1/2) # taking square root

#========================================#

# Evaluate the prediction variance for x_0 = 1013 mb,
# as well as for a nearby range like 1005 - 1015 mb.
# and check the probability and prediction interval, by
# scipy.stats.norm.cdf() and scipy.stats.norm.ppf().
#========================================#
x_bar = np.average(data_7["Pres"]) # get mean of data
x_sd = np.std(data_7["Pres"]) # get std of data
n = len(data_7) # get length of data

def get_prob(p):
    s_y2s = s_e2 * (1 + 1 / n + (p - x_bar) ** 2 / (n * (x_sd ** 2)))  # calculate prediction variance
    prob = [np.diff(scipy.stats.norm(0, s_y2 ** (1 / 2)).cdf([-1, 1])).item() for s_y2 in s_y2s] # calculate probability
    return prob

x0 = 1013 # set x0
prob0 = get_prob([x0]) # get the probability when x0 = 1013 to answer (e)

x = np.arange(1005, 1015, 0.05)
prob = get_prob(x) # get the probability for the array of inputs above
u_ls = []; pred = []; l_ls = []; # initialize three lists
for predict in ols.predict([1, x])[0]: # get the prediction interval in the lists
    u_ls.append(predict + 1.96 * s_e)
    pred.append(predict)
    l_ls.append(predict - 1.96 * s_e)


#========================================#

# Plot the regression function and the 95% prediction interval.
# I think you have been familiar with plotting,
# so I won't write any outline from now.
#========================================#
fig = plt.figure() # create figure
ax = fig.add_subplot()
# plot the three lines
ax.plot(x, u_ls, label='upper bound')
ax.plot(x, pred, label='prediction')
ax.plot(x, l_ls, label='lower bound')

# plot the points
ax.scatter(data_7['Pres'], data_7['Temp'])

# aesthetics
ax.legend() # show legend
ax.set_title('95% prediction intervals for June Temperature and Pressure')  # set title
ax.set_ylabel('June Temperature (deg C)')  # set ylabel
ax.set_xlabel('June Pressure (mb)')  # set ylabel


# Durbin-Watson Test
from statsmodels.stats.stattools import durbin_watson
d = durbin_watson(ols.resid) # calculate d value