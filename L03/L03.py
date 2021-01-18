#========================================#
# ESSC 4510 Tutorial 03 Python Script
# Name: Li Lok Hang
# Student ID: 1155092891
# Sample Script made by Benjamin Loi
#========================================#

# Import Necessary Packages
#========================================#
import pandas as pd # Reading Table
import numpy as np # Processing Array
import scipy.stats # Computing Statistic
import xarray as xr # Reading .nc File
import matplotlib.pyplot as plt # Drawing Graph
import cartopy.crs as ccrs # Map Projection



#========================================#

# Ex 3.2
# Get the rainfall data from the given .nc file.
#========================================#
data_3_2 = xr.open_dataset("temp_precip_city_China_July_1998-2015.nc")
prec_BJ, prec_WH, prec_HK = data_3_2["prec_BJ"], data_3_2["prec_WH"], data_3_2["prec_HK"]
print(prec_HK)



#========================================#

# Select all July dates
#========================================#

def get_a_rainy_day_prob(prec):
    ''' This functions requires a 'prec_XX' precipitation data variable,
        that is in a form of xarray format. Outputs ONE rainy day probability. '''
    dates = prec["date"]
    # The dates are in YYYYMMDD.
    # Two possible approaches:
    # 1. Convert the dates into string and check if the MM substring is 07,
    # 2. Use modulus to get the last 4 digits and check if they are within the range of 700-799.
    # The July dates should be in the form of a boolean array, with respect to all dates.
    mod = (dates % 1000).values # taking modulus
    bool_array = (700 < mod) & (mod < 800) # save as boolean array

    #========================================#

    # Calculate the probabilities of having a rainy day in July,
    # which is the amounts of rainy days in July over the total numbers of days in July.
    # Use the boolean array obtained in above selection for filtering, and
    # apply array comparison to get the dates when the rainfall is > 1mm.
    # Something along the lines like <prec>[<July dates>] and ... > 1.
    # The numbers of True in an array is simply given by np.sum() or np.count_nonzero().
    #========================================#
    return np.sum(prec[bool_array] > 1) / len(prec[bool_array]) # calculate the probability

#========================================#

# Calculate the required probabilities.
# You may want to compute the binomial coefficients, or nCr.
# Some approaches: https://stackoverflow.com/questions/3025162/statistics-combinations-in-python
#========================================#
    # probability that a day is a rainy day in July
    p_HK = get_a_rainy_day_prob(prec_HK)
    p_BJ = get_a_rainy_day_prob(prec_BJ)
    p_WH = get_a_rainy_day_prob(prec_WH)

    # probability that every day is a rainy day in July
    print(p_HK ** 31) #3.2a
    print(p_BJ ** 31) #3.2b
    print(p_WH ** 31) #3.2c

    # probability that at least 25 days are rainy days in July
    # = 1 - probability that less than 25 days are rainy days
    # note that scipy.stats.binom.cdf(k, n, p), where k = 25, n = 31, p = p_HK respectively
    print( 1 - scipy.stats.binom.cdf(25, 31, p_HK) ) #3.2d

#========================================#


# Ex 3.3
# Extract the pressure data from the given .nc file.
#========================================#
data_3_3 = xr.open_dataset("slp_1948-2014.nc")
slp = data_3_3["slp"]
print(slp)
lat_array = data_3_3["lat"]
lon_array = data_3_3["lon"]



#========================================#

# Locate Darwin (Lat: -12.46, Lon: 130.84) from the data,
# by manipulating whole array operation, np.abs(), <>.argmin(),
# to search the nearest lat/lon indices.
#========================================#
lat_Darwin = -12.46
lon_Darwin = 130.84
nearest_y = int(np.abs(lat_Darwin - lat_array).argmin())
print("Nearest y-index:", nearest_y)
# Repeat the same for longitude.
nearest_x = int(np.abs(lon_Darwin - lon_array).argmin())
print("Nearest x-index:", nearest_x)



#========================================#

# Find the slp correlation between Darwin and every grid point,
# by using double for loop to fill in a NaN array,
# that has the same lat/lon extent as the slp array.
#========================================#
slp_corr = np.empty([len(lon_array), len(lat_array)])
# Notice that slp_corr has the shape (row = nlon) * (column = nlat).
# Make sure that all data are initially NaN to deal with potential problems during later calculation.
slp_corr[:] = np.nan
for i in np.arange(len(lon_array)):
    for j in np.arange(len(lat_array)):
        # Iterate over all grid points, slp[{"lon": i, "lat": j}]),
        # and compared against the Darwin slp at nearest_lat, nearest_lon.
        # Use scipy.stats.pearsonr(<>, <>)[0] to compute the correlation,
        # and assign the answer to the empty array slp_corr.
        Darwin_slp = slp[{"lon": nearest_x, "lat": nearest_y}]
        Grid_slp = slp[{"lon": i, "lat": j}]
        slp_corr[i][j] = scipy.stats.pearsonr(Darwin_slp, Grid_slp)[0]



#========================================#
# Remark: it is possible to utilize vectorized functions only and avoid for loop.

# Prepare the map object for plotting.
#========================================#
fig = plt.figure()
# Mercator Projection
projection = ccrs.PlateCarree(central_longitude=0.0)
ax_map = fig.add_subplot(1,1,1, projection=projection)
ax_map.coastlines()



#========================================#

# Draw the correlation contour plot by <>.contour(<lon>, <lat>, <values>)
#========================================#
ct = ax_map.contour(lon_array, lat_array, slp_corr.T, # T stands for transpose.
                    np.linspace(-1, 1, 11)) # the last argument indicates the contour intervals.
ax_map.clabel(ct)



#========================================#
# Remark: A discontinuity appears in the middle of the contour plot.
# This can be fixed by manually cycling the data,
# adding an extra longtitude of 360.0, copied from longtitude of 0.0.
# from cartopy.util import add_cyclic_point may help,
# Reference: https://scitools.org.uk/cartopy/docs/v0.15/cartopy/util/util.html

# Add legends, titles, axis labels.
#========================================#
import matplotlib.patches as mpatches # add legend manually
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter # to format longitude and latitudes

# Adding legend
patch = mpatches.Patch(label = 'Pearson correlation between point and Darwin')
ax_map.legend(handles=[patch])

# Adding title
plt.title('Correlation map for the city Darwin')

# Adding axis labels
ax_map.set_xticks(np.linspace(-180, 180, 5), crs=projection)
ax_map.set_yticks(np.linspace(-90, 90, 5), crs=projection)
ax_map.xaxis.set_major_formatter(LongitudeFormatter())
ax_map.yaxis.set_major_formatter(LatitudeFormatter())
plt.show()


#========================================#

# Output the figure as .png format.
#========================================#
fig.savefig("Ex_3_3.png")



#========================================#
