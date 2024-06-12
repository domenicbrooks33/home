import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter
from datetime import datetime, timedelta
import netCDF4 as nc
import xarray as xr
from metpy.units import masked_array, units
from urllib.request import urlopen
from pyproj import Proj, transform

'''
# fucntion to read in netcdf file

def netcdf_readin(filename):
    df_qpe = nc.Dataset('/homes/metogra/dbrooks6/wpc_research/stageIVqpe_data/2023/' + filename, mode='r')
    return df_qpe
    
df_qpe = netcdf_readin('nws_precip_1day_20231007_conus.nc') # runs file read in
'''
# data request for time period that you want (in this case you can only do days, 12z-12z)
year = 2023 
month = 12
day = 4
obs_end = datetime(year, month, day, 12) # set year, month, and day you want data for

################################################# gridded data ##############################################

dt = obs_end - timedelta(days=1)  

# reads in data file from url
url = 'http://water.weather.gov/precip/downloads/{obs_end:%Y/%m/%d}/nws_precip_1day_{obs_end:%Y%m%d}_conus.nc'.format(obs_end=obs_end)
data = urlopen(url).read()
df_qpe = nc.Dataset('data', memory=data)

print(url)

'''

# extract desired variables from file

qpe_precip = df_qpe.variables['observation'] #gridded rainfall amounts (in inches)

gridded_precip = masked_array(qpe_precip[:], units(qpe_precip.units.lower())).to('in') # makes sure data is in inches


x = df_qpe.variables['x'][:] # in meters (because it's in a projected coordinate system)
y = df_qpe.variables['y'][:]


proj_var = df_qpe.variables[qpe_precip.grid_mapping] # get the projection for the plot

#print(type(df_qpe.creation_time))
print(gridded_precip.shape)
#print(y)
# extract projection information from data file (it's in stereographic projection)

globe = ccrs.Globe(semimajor_axis=proj_var.earth_radius) # globe shape used (like ellipsoid shape)
proj = ccrs.Stereographic(central_latitude=90.0,
                          central_longitude=proj_var.straight_vertical_longitude_from_pole,
                          true_scale_latitude=proj_var.standard_parallel, globe=globe)
                       

#################################### rain gauge observations #####################################

gauge_data = pd.read_csv('https://www.wpc.ncep.noaa.gov/qpf/obsmaps/p24i_{obs_end:%Y%m%d}_sortbyarea.txt'.format(obs_end=obs_end),
 skiprows=6, sep= '    ', engine='python') # reads in data file for same time as above
gauge_data = gauge_data.to_numpy()
#print(gauge_data.shape)
gauge_lat = gauge_data[:,0]
gauge_lon = gauge_data[:,1]
gauge_precip = gauge_data[:,2]

# Define the projection parameters for North Pole Stereographic projection (same as HRAP grid)
projection_params = {
    'proj': 'stere',
    'lat_0': 90,
    'lon_0': -105,
    'lat_ts': 60,
    'a': 6371200,  # semi-major axis of the ellipsoid
    'b': 6371200,  # semi-minor axis of the ellipsoid
}


# Create the projection object
projection = Proj(projection_params)

i=0
while i < len(gauge_lon):
    gauge_lon[i], gauge_lat[i] = projection(gauge_lon[i], gauge_lat[i])
    i = i + 1
    
rain_gauges = np.column_stack((gauge_lon, gauge_lat, gauge_precip)) # makes rain gauge coords into single array
#print(rain_gauges)
    
      

######################################## plotting the figure ###############################################
fig = plt.figure(figsize=(12,8))
ax = plt.axes(projection=proj)

# draw coastlines, state and country boundaries, edge of map.
ax.coastlines()
ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.STATES)
ax.set_extent([-123.0,-70.0, 23.0, 60.0], crs=ccrs.PlateCarree()) # sets domain limits of map

############################# Creating the colormap ###################################
# draw filled contours.
clevs = [0.01, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5,
         3, 4, 5, 6, 8, 10, 12, 15, 18]

# makes first color in colorbar white
colormap = cm.get_cmap('gist_ncar')
#newcolors = colormap(np.linspace(0, 1, 18))
#white = np.array([0, 0, 0, 0]) #rgb for white
#newcolors[:0, :] = white
cmap = mcolors.ListedColormap(colormap(np.linspace(0.15, 0.95, 18))) # selects range on colormap I want to use for my colors

norm = mcolors.BoundaryNorm(clevs, cmap.N) #normalizes the bounds for the color mapping
########################################################################################

############################# turn these puppies on if you just want to show HRAP domain #########################################
#cmap = mcolors.ListedColormap(colormap(np.linspace(0.15, 0.15, 18))) # selects range on colormap I want to use for my colors


#z=gridded_precip.shape

#ax.pcolormesh(x, y, gridded_precip, norm=norm, edgecolor='k' transform=proj) #with black outlines of pixels
ax.pcolormesh(x, y, gridded_precip, norm=norm, cmap=cmap, transform=proj) # plots gridded data
cs = ax.scatter(gauge_lon, gauge_lat, c=gauge_precip, cmap=cmap, s=25, edgecolors='black', norm=norm, transform=proj, label='COOP Stations') # plots gauge points (line must be below pixel data in order to be visible)


# add colorbar.
cbar = plt.colorbar(cs, shrink=0.75)
cbar.cmap.set_under('white')
cbar.set_label(qpe_precip.units)
fig.tight_layout()

ax.legend(loc=0)

ax.set_title('Precipitation from ' + datetime.strftime(dt, "%Y-%m-%d %H:%M:%SZ") + ' to ' + datetime.strftime(obs_end, "%Y-%m-%d %H:%M:%SZ"))
plt.show()


# combining x and y into one array as coordinate pairs
lon_grid, lat_grid = np.meshgrid(x, y)
coordinates = np.column_stack((lon_grid.ravel(), lat_grid.ravel()))


# Convert precipitation values to dimensionless because for some reason the precip values are inherently in inches
import pint
ureg = pint.get_application_registry()
gridded_precip = gridded_precip / ureg.inch

# Flatten the gridded_precip array to 1-D
precip_values = gridded_precip.ravel()


# Combine the coordinates and precipitation values
grid_data_coords = np.column_stack((coordinates, precip_values))


min_x = np.min(x)
min_y = np.min(y)
max_x = np.max(x)
max_y = np.max(y)

#print(min_x, max_x)
#print(min_y, max_y)

# gets the resolution of each grid pixel
x_resolution = abs(min_x - max_x) / 1121 
y_resolution = abs(min_y - max_y) / 881

df = pd.DataFrame(rain_gauges, columns=['x_index', 'y_index', 'gauge_precip']) # dataframe for rain gauge obs

df = df[(df["x_index"] <= max_x ) & (df["x_index"] >= min_x)] # sorts out gauges only within domain of HRAP
rain_gauges = df[(df["y_index"] <= max_y ) & (df["y_index"] >= min_y)] # dataframe for rain gauge obs

point_id = np.arange(len(rain_gauges['y_index']))
rain_gauges['point_id'] = point_id # assigns each rain gauge its own value to identify itself by


pixel_data = pd.DataFrame(grid_data_coords, columns=['x', 'y', 'precip']) # dataframe containing the coordinates of each pixel center
point_id = np.arange(len(pixel_data['x']))
pixel_data['pixel_id'] = point_id # assigns each pixel its own value to identify itself by

pixel_data = pixel_data[(pixel_data['precip'] > 0.0)] # sorts out all pixels with precip estimate of zero

from shapely.geometry import Point, box
import geopandas as gpd

# creates seperate columns to store the geometry of the points and boxes
rain_gauges['geometry'] = [Point(xy) for xy in zip(rain_gauges['x_index'], rain_gauges['y_index'])]
pixel_data['geometry'] = [box(row['x'] - x_resolution/2, row['y'] - y_resolution/2,
                              row['x'] + x_resolution/2, row['y'] + y_resolution/2) for _, row in pixel_data.iterrows()]
                              
# Convert the DataFrame to a GeoDataFrame
gdf_rain_gauges = gpd.GeoDataFrame(rain_gauges, geometry='geometry')

# Create spatial index for rain_gauges
sindex_rain_gauges = gdf_rain_gauges.sindex

# algorithim to select pixels which contain points in them
pixels_with_points = []
gauges_in_pixels = []

for _, row in pixel_data.iterrows():
    pixel_box = row['geometry']
    possible_matches_index = list(sindex_rain_gauges.intersection(pixel_box.bounds)) # creates list of spatial indices based on pixel box bounds
    print(row['pixel_id'])
    
    for index in possible_matches_index:
        point_geom = rain_gauges['geometry'].iloc[index] #iterates through each spatial index for the rain gauge points
        
        if pixel_box.contains(point_geom):
            pixels_with_points.append(row['pixel_id']) # adds pixel id to a list if there is a gauge within that pixel
            gauges_in_pixels.append(rain_gauges['point_id'].iloc[index]) # adds point id to list 
            x1, y1 = pixel_box.exterior.xy # saves geometry of boxes as seperate variables to use to plot boxes
            print(row['pixel_id'], rain_gauges['point_id'].iloc[index])
            rain_gauges.at[index, 'pixel_id'] = row['pixel_id'] # adds row to rain gauge df to have pixel id it corresponds to
            
        
rain_gauges = rain_gauges.dropna() # eliminates any rows with NaN 
print(rain_gauges)   

# Calculate average rainfall for each unique "pixel_id"
average_precip_by_pixel = rain_gauges.groupby('pixel_id')['gauge_precip'].mean().reset_index()

# Display the result
print(average_precip_by_pixel)


# filter out only the pixels with gauges in them in the dataframe
pixels_w_gauges = pixel_data[pixel_data['pixel_id'].isin(pixels_with_points)]
print(pixels_w_gauges)

# Merge DataFrames on the 'pixel_id' column
merged_df = pd.merge(average_precip_by_pixel, pixels_w_gauges, on='pixel_id', how='inner')
#print(merged_df)
    
merged_df['bias'] = merged_df['precip'] / merged_df['gauge_precip'] # calculates bias for each pixel
#print(type(merged_df))

# Convert the DataFrame to a GeoDataFrame
gdf = gpd.GeoDataFrame(merged_df, geometry='geometry')
polygons = gdf['geometry']
bias = gdf['bias']

print(bias)

######################################## plotting the figure ###############################################
fig = plt.figure(figsize=(12,8))
ax = plt.axes(projection=proj)

# draw coastlines, state and country boundaries, edge of map.
ax.coastlines()
ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.STATES)
ax.set_extent([-123.0,-70.0, 23.0, 60.0], crs=ccrs.PlateCarree()) # sets domain limits of map

############################# Creating the colormap ###################################
# draw filled contours.
clevs = [0.01, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5,
         3, 4, 5, 6, 8, 10, 12, 15, 18]

# makes first color in colorbar white
colormap = cm.get_cmap('gist_ncar')
#newcolors = colormap(np.linspace(0, 1, 18))
#white = np.array([0, 0, 0, 0]) #rgb for white
#newcolors[:0, :] = white
cmap = mcolors.ListedColormap(colormap(np.linspace(0.15, 0.95, 18))) # selects range on colormap I want to use for my colors

norm = mcolors.BoundaryNorm(clevs, cmap.N) #normalizes the bounds for the color mapping
########################################################################################

#ax.pcolormesh(x, y, gridded_precip, norm=norm, edgecolor='k' transform=proj) #with black outlines of pixels
#cs = ax.pcolormesh(x, y, gridded_precip, norm=norm, cmap=cmap, transform=proj) # plots gridded data
#cs = ax.scatter(rain_gauges['x_index'], rain_gauges['y_index'], c=rain_gauges['gauge_precip'], cmap=cmap, s=25, marker = '^', alpha=0.7, edgecolors='black', norm=norm, label='COOP Stations', transform=proj) 
#ax.scatter(pixel_data['x'], pixel_data['y'], c=pixel_data['precip'], cmap=cmap, s=25, edgecolors='blue', norm=norm, transform=proj)


# Plot each polygon using the exterior coordinates
for polygon in polygons:
    x, y = polygon.exterior.xy
    ax.plot(x, y, color='red', transform=proj)

# Plot each polygon using the exterior coordinates
for polygon, bias in zip(polygons, bias):
    x, y = polygon.exterior.xy
    ax.fill(x, y, color=plt.cm.RdYlBu(bias), transform=proj)


# Add a colorbar
norm = Normalize(vmin=0, vmax=2)
sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu, norm=norm)
sm.set_array([])  # empty array for the data range
cbar_ax = fig.add_axes([0.9, 0.125, 0.03, 0.755])  # Adjust the position as needed
cbar = fig.colorbar(sm, cax=cbar_ax, label='Bias')


# add colorbar.
cbar = plt.colorbar(cs, shrink=0.75)
cbar.cmap.set_under('white')
cbar.set_label(qpe_precip.units)

fig.tight_layout()

ax.legend(loc=0)

ax.set_title('Bias between Stage IV QPE and COOP Rain Gauge Precipitation \n from ' + datetime.strftime(dt, "%Y-%m-%d %H:%M:%SZ") + ' to ' + datetime.strftime(obs_end, "%Y-%m-%d %H:%M:%SZ"))

plt.show()
'''
