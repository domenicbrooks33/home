import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
from scipy.stats import pearsonr
import geopandas as gpd
from sklearn.metrics import r2_score
import seaborn as sns


folders = glob.glob("/homes/metogra/dbrooks6/wpc_research/final_analysis_data/*")

#print(folders)

# Read file paths
csv_files=[]

for folder in folders:
  string = glob.glob("{0}/*.csv".format(folder))
  #print(string)
  csv_files.append(string)
  
csv_files.sort()

# Read RFC names
rfc = pd.read_csv('/homes/metogra/dbrooks6/wpc_research/analysis_scripts/all2022.csv', header='infer')
rfc_names = rfc["RFC_NAME"]
print(rfc_names)

'''
############## Bias and correlation #############
bias = []
correlation=[]

n_values=[]

################## All 2022, no conditions ####################
fig, axes = plt.subplots(3, 4, figsize=(15, 10), sharex=False, sharey=False)


for i, files in enumerate(csv_files):
    #print(files)
    data_table = pd.read_csv(files[0], header='infer')
    #print(data_table)
    stageIV_sum = data_table['stageIV_pr'].sum()
    gauge_sum = data_table['Gauge_precip'].sum()
    
    n_values.append(len(data_table['stageIV_pr']))
    
    
    # bias calculation
    bias_calc = stageIV_sum/gauge_sum
    bias.append(bias_calc)
    
    # correlation calculation
    cc = pearsonr(data_table['stageIV_pr'], data_table['Gauge_precip']).statistic
    #print(cc)
    correlation.append(cc)
    
    ############################## Scatter Plots ####################################
    
    row = i // 4
    col = i % 4

    ax = axes[row, col]

    
    # Extract the columns you want to plot
    x_values = data_table['Gauge_precip']
    y_values = data_table['stageIV_pr']
    
    # Plot the scatter plot
    ax.scatter(x_values, y_values, alpha=0.3, edgecolor='blue')
    
    # Plot the line with a slope of 1
    ax.plot([0,max(y_values)+30], [0,max(y_values)+30], color='black')
    
    # Calculate R-squared value
    r_squared = r2_score(y_values, x_values)
    
    # Add R-squared value to the plot
    ax.text(max(y_values)*0.3, (max(y_values)+30)*0.9, r'$R^2$' + f' = {r_squared:.2f}', ha='center', va='center', fontsize=10)

    plt.xlim(0,max(y_values)+30)
    plt.ylim(0,max(y_values)+30)
    
    # Add labels and title
    #ax.set_xlabel('Gauge Precip (mm/day)')
    #ax.set_ylabel('Stage IV Precip (mm/day)')
    ax.set_title('{0}'.format(rfc_names[i]))
    
    # Add legend
    #plt.legend()
    
    
plt.tight_layout()  # Adjust spacing between subplots
plt.show()
'''   


'''
# Define a function to map each month to its corresponding season
def get_season(month):
  if month in [12, 1, 2]:
      return 'Winter'
  elif month in [3, 4, 5]:
      return 'Spring'
  elif month in [6, 7, 8]:
      return 'Summer'
  elif month in [9, 10, 11]:
      return 'Fall'
  else:
      print('uhhhhhhhhhhhhhhhh')
          
          
winter_bias = []
spring_bias = []
summer_bias = []
fall_bias = []

winter_corr = []
spring_corr = []
summer_corr = []
fall_corr = []

################# By season #############################

for files in csv_files:
    #print(files)
    data_table = pd.read_csv(files[0], header='infer')

    #print(data_table['date'])
    # Convert 'Date' column to datetime
    data_table['date'] = pd.to_datetime(data_table['date'], format='%Y-%m-%d')
    
    #print(data_table['date'])
    
    # Create a new column 'Season' based on the 'Date' column
    data_table['Season'] = data_table['date'].dt.month.map(get_season)
    
    #data_table.to_csv('/homes/metogra/dbrooks6/wpc_research/analysis_scripts/test.csv', index=True)
    
    # Group by 'Season' and sum up the gauge values
    gauge_seasonal_precipitation = data_table.groupby('Season')['Gauge_precip'].sum()
    stageIV_seasonal_precipitation = data_table.groupby('Season')['stageIV_pr'].sum()
    
    # Bias calculation
    bias_by_season = stageIV_seasonal_precipitation / gauge_seasonal_precipitation
    
    winter_bias2 = bias_by_season.loc['Winter']
    spring_bias2 = bias_by_season.loc['Spring']
    summer_bias2 = bias_by_season.loc['Summer']
    fall_bias2 = bias_by_season.loc['Fall']
    
    winter_bias.append(winter_bias2)
    spring_bias.append(spring_bias2)
    summer_bias.append(summer_bias2)
    fall_bias.append(fall_bias2)

    #print(winter_bias)
    
    
    # correlation calculation
    
    # Group data_table by season and calculate the correlation coefficient for each season
    correlation_by_season = data_table.groupby('Season').apply(lambda x: pearsonr(x['Gauge_precip'], x['stageIV_pr'])[0])
    
    winter_corr2 = correlation_by_season.loc['Winter']
    spring_corr2 = correlation_by_season.loc['Spring']
    summer_corr2 = correlation_by_season.loc['Summer']
    fall_corr2 = correlation_by_season.loc['Fall']
    
    winter_corr.append(winter_corr2)
    spring_corr.append(spring_corr2)
    summer_corr.append(summer_corr2)
    fall_corr.append(fall_corr2)
    #print(correlation_by_season)
    
    
#print(winter_corr)
#print(spring_bias)

'''

##################### Rainfall Intensity Analysis ##################
# Define a function to calculate correlation coefficient for each group
def calculate_correlation(group):
    return pearsonr(group['Gauge_precip'], group['stageIV_pr'])[0]


light = []
light_med = []
med = []
heavy = []

light_corr = []
light_med_corr = []
med_corr = []
heavy_corr = []

for files in csv_files:
    #print(files)
    data_table = pd.read_csv(files[0], header='infer')
    
    # Drop all rows where 'Gauge_precip' is zero
    data_table = data_table[data_table['Gauge_precip'] != 0]
    
    # Calculate the percentiles dynamically
    percentiles = data_table['Gauge_precip'].quantile([0, 0.5, 0.7, 0.9, 1]).tolist()
    labels = ['0-50', '50-70', '70-90', '90+']
    
    print(percentiles)
    #print(percentiles)
    
    # Use pd.cut() to classify the rows based on 'Gauge_precip' percentiles
    data_table['Precipitation Class'] = pd.cut(data_table['Gauge_precip'], bins=percentiles, labels=labels, right=False)
    
    #print(data_table)

    # Group the dataframe by 'Precipitation Class' and calculate correlation coefficient for each group
    correlation_by_class = data_table.groupby('Precipitation Class').apply(calculate_correlation)
    
    #print(correlation_by_class)
    
    light_corr.append(correlation_by_class.loc['0-50'])
    light_med_corr.append(correlation_by_class.loc['50-70'])
    med_corr.append(correlation_by_class.loc['70-90'])
    heavy_corr.append(correlation_by_class.loc['90+'])
    
    # Group the dataframe by the classification and sum the values in 'Gauge_precip' and 'stageIV_pr'
    summary_by_classification = data_table.groupby('Precipitation Class')[['Gauge_precip', 'stageIV_pr']].sum()
    
    #print(summary_by_classification)
    
    light_bias = summary_by_classification.loc['0-50']['stageIV_pr'] / summary_by_classification.loc['0-50']['Gauge_precip']
    #print(light_bias)
    light_med_bias = summary_by_classification.loc['50-70']['stageIV_pr'] / summary_by_classification.loc['50-70']['Gauge_precip']
    med_bias = summary_by_classification.loc['70-90']['stageIV_pr'] / summary_by_classification.loc['70-90']['Gauge_precip']
    heavy_bias = summary_by_classification.loc['90+']['stageIV_pr'] / summary_by_classification.loc['90+']['Gauge_precip']
    
    light.append(light_bias)
    light_med.append(light_med_bias)
    med.append(med_bias)
    heavy.append(heavy_bias)
'''
    

rfc_boundaries = glob.glob("/homes/metogra/dbrooks6/wpc_research/rfc_boundaries/*.shp")
rfc_boundaries.sort()


rfc_gdf = []
p=0
# Iterate over each shapefile path
for shapefile_path in rfc_boundaries:
    # Load the shapefile
    gdf = gpd.read_file(shapefile_path)
    gdf['Light corr'] = light_corr[p]
    gdf['Light-Med corr'] = light_med_corr[p]
    gdf['Med corr'] = med_corr[p]
    gdf['Heavy corr'] = heavy_corr[p]
    rfc_gdf.append(gdf)
    #print(gdf)
    p=p+1
    
# Concatenate all GeoDataFrames into a single GeoDataFrame
combined_gdf = pd.concat(rfc_gdf, ignore_index=True)
print(combined_gdf.columns)
print(combined_gdf[['RFC_NAME','Light corr','Light-Med corr','Med corr', 'Heavy corr']])

# plot bias by RFC
#combined_gdf.plot(column='Light Bias', cmap='seismic',edgecolor='black', vmin=0.7, vmax=1.3, legend=True, legend_kwds={'orientation': 'horizontal'})
ax = combined_gdf.plot(column='Heavy corr', cmap='Greens',edgecolor='black', vmin=0.2, vmax=1,  legend=True, legend_kwds={'orientation': 'horizontal'})
#ax.set_axis_off()
# Add title and show the plot
#plt.title('Winter (DJF) Bias of Each RFC')
plt.show()

'''
rfc_boundaries = glob.glob("/homes/metogra/dbrooks6/wpc_research/rfc_boundaries/*.shp")
rfc_boundaries.sort()


rfc_gdf = []
p=0
# Iterate over each shapefile path
for shapefile_path in rfc_boundaries:
    # Load the shapefile
    gdf = gpd.read_file(shapefile_path)
    gdf['Light'] = light_corr[p]
    gdf['Light-Medium'] = light_med_corr[p]
    gdf['Medium'] = med_corr[p]
    gdf['Heavy'] = heavy_corr[p]
    #gdf['Spring Corr'] = fall_corr[p]
    rfc_gdf.append(gdf)
    #print(gdf)
    p=p+1
    
# Concatenate all GeoDataFrames into a single GeoDataFrame
combined_gdf = pd.concat(rfc_gdf, ignore_index=True)

no_geo_df = combined_gdf.drop(columns=['geometry','OBJECTID', 'SITE_ID', 'STATE','RFC_CITY','BASIN_ID'])

no_geo_df.to_csv('/homes/metogra/dbrooks6/wpc_research/analysis_scripts/intensity2022.csv', index=True)

print(no_geo_df)

# Set 'RFC_NAME' as index
no_geo_df.set_index('RFC_NAME', inplace=True)

# Plot the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(no_geo_df, annot=True, cmap='Greens',vmin=0.2, vmax=1, fmt=".2f", annot_kws={"size": 10})
#plt.title('Bias Heatmap')
plt.xlabel('Precipitation Intensity')
plt.ylabel('RFC')
plt.tight_layout()  # Ensures all elements fit into the figure
plt.show()

# plot bias by RFC
#combined_gdf.plot(column='Light Bias', cmap='seismic',edgecolor='black', vmin=0.7, vmax=1.3, legend=True, legend_kwds={'orientation': 'horizontal'})
#ax = combined_gdf.plot(column='Heavy Bias', cmap='seismic',edgecolor='black', vmin=0.7, vmax=1.3)
#ax.set_axis_off()
# Add title and show the plot
#plt.title('Winter (DJF) Bias of Each RFC')
#plt.show()


'''
rfc_boundaries = glob.glob("/homes/metogra/dbrooks6/wpc_research/rfc_boundaries/*.shp")
rfc_boundaries.sort()


rfc_gdf = []
p=0
# Iterate over each shapefile path
for shapefile_path in rfc_boundaries:
    # Load the shapefile
    gdf = gpd.read_file(shapefile_path)
    gdf['Winter'] = winter_corr[p]
    gdf['Spring'] = spring_corr[p]
    gdf['Summer'] = summer_corr[p]
    gdf['Fall'] = fall_corr[p]
    rfc_gdf.append(gdf)
    #print(gdf)
    p=p+1
    
# Concatenate all GeoDataFrames into a single GeoDataFrame
combined_gdf = pd.concat(rfc_gdf, ignore_index=True)
#print(combined_gdf)

no_geo_df = combined_gdf.drop(columns=['geometry','OBJECTID', 'SITE_ID', 'STATE','RFC_CITY','BASIN_ID'])

no_geo_df.to_csv('/homes/metogra/dbrooks6/wpc_research/analysis_scripts/season2022.csv', index=True)

print(no_geo_df)
# Set 'BASIN_ID' as index
no_geo_df.set_index('RFC_NAME', inplace=True)

# Plot the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(no_geo_df, annot=True, cmap='Greens',vmin=0.85, vmax=1, fmt=".2f", annot_kws={"size": 10})
#plt.title('Bias Heatmap')
plt.xlabel('Season')
plt.ylabel('RFC')
plt.tight_layout()  # Ensures all elements fit into the figure
plt.show()

#print(combined_gdf[['RFC_NAME','Spring Corr']])

# plot bias by RFC
#combined_gdf.plot(column='Winter Bias', cmap='seismic',edgecolor='black', vmin=0.7, vmax=1.3, legend=True, legend_kwds={'orientation': 'horizontal'})
#ax = combined_gdf.plot(column='Fall Bias', cmap='seismic',edgecolor='black', vmin=0.7, vmax=1.3)
#ax.set_axis_off()
# Add title and show the plot
#plt.title('Winter (DJF) Bias of Each RFC')
#plt.show()

# plot bias by RFC
#ax2 = combined_gdf.plot(column='Spring Corr', cmap='Greens',edgecolor='black', vmin=0.85, vmax=1)
#ax2.set_axis_off()
# Add title and show the plot
#plt.title('Winter (DJF) Correlation of Each RFC')
#plt.show()
'''

'''
########################## All 2022 ###############################
rfc_boundaries = glob.glob("/homes/metogra/dbrooks6/wpc_research/rfc_boundaries/*.shp")
rfc_boundaries.sort()

rfc_gdf = []
p=0
# Iterate over each shapefile path
for shapefile_path in rfc_boundaries:
    # Load the shapefile
    gdf = gpd.read_file(shapefile_path)
    gdf['Bias'] = bias[p]
    gdf['Corr'] = correlation[p]
    gdf['N'] = n_values[p]
    rfc_gdf.append(gdf)
    #print(gdf)
    p=p+1
    
# Concatenate all GeoDataFrames into a single GeoDataFrame
combined_gdf = pd.concat(rfc_gdf, ignore_index=True)
print(combined_gdf)

no_geo_df = combined_gdf.drop(columns='geometry')

no_geo_df.to_csv('/homes/metogra/dbrooks6/wpc_research/analysis_scripts/all2022.csv', index=True)
'''
'''
# plot bias by RFC
combined_gdf.plot(column='Bias', cmap='seismic',edgecolor='black', vmin=0.75, vmax=1.25, legend=True)

# Add title and show the plot
#plt.title('Bias of Each RFC')
plt.xlabel('Lon')
plt.ylabel('Lat')
plt.tight_layout()
plt.show()

# plot bias by RFC
combined_gdf.plot(column='Corr', cmap='Greens',edgecolor='black', vmin=0.97, vmax=1, legend=True)
plt.xlabel('Lon')
plt.ylabel('Lat')
plt.tight_layout()
# Add title and show the plot
#plt.title('Correlation of Each RFC')
plt.show()
'''

'''
###################### Plotting bar graphs #########################
# Plotting the bar graph
plt.bar(rfc_name, correlation)

# Adding labels and title
plt.xlabel('RFC Domains')
plt.ylabel('Correlation')
plt.title('Correlation of Stage IV vs. Rain Gauge Measurements for Each RFC (2022)')

plt.xticks(rotation=45)
plt.ylim(0.97, 1)

# Displaying the plot
plt.show()
'''


    