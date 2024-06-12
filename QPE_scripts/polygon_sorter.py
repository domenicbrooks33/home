import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from shapely.geometry import Point
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from tqdm import tqdm
import os
import matplotlib.cm as cm
import glob


# Function to read CSV file and convert coordinates to GeoDataFrame with precipitation values
def read_csv_and_create_geodataframe(coordinates, gauge_data):
    points = []
    
    # create dfs
    gauge_coords = pd.read_csv(coordinates, index_col=0)
    gauge_precip = pd.read_csv(gauge_data, index_col=0)

    gauge_coords['gauge_precip'] = np.array(gauge_precip.iloc[0]) # gets lat lon and precip values into one df
    
    # loop that creates points to put in df
    row=0
    while row < len(gauge_coords):
        precipitation = float(gauge_coords['gauge_precip'].iloc[row])
        latitude = float(gauge_coords['Latitude'].iloc[row])
        longitude = float(gauge_coords['Longitude'].iloc[row])

        point = Point(longitude, latitude)
        points.append((point, precipitation))
        
        row=row+1
    
    # Create a DataFrame from the list of tuples
    df = pd.DataFrame([(point.x, point.y, precipitation) for point, precipitation in points], columns=['Longitude', 'Latitude', 'Gauge_precip'])
    # Convert DataFrame to GeoDataFrame
    geometry = [Point(xy) for xy in zip(df.Longitude, df.Latitude)]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    return gdf


coordinates = "/homes/metogra/dbrooks6/wpc_research/uscrn_data/rain_gauge_coordinates.csv"
gauge_data = glob.glob("/homes/metogra/dbrooks6/wpc_research/uscrn_data/2022/raingauges_2022*.csv")
stageIV_shapefile_paths = glob.glob("/homes/metogra/dbrooks6/wpc_research/stageIVqpe_data_rfc/arkansas_red/2022/shapefiles/*.shp")
mrms_shapefile_paths = glob.glob('/homes/metogra/dbrooks6/wpc_research/mrms_data_rfc/arkansas_red/2022/shapefiles/*.shp', recursive=True)

gauge_data.sort()
stageIV_shapefile_paths.sort()
mrms_shapefile_paths.sort()

print(len(gauge_data))
print(len(stageIV_shapefile_paths))
print(len(mrms_shapefile_paths))

print(gauge_data[170])
print(stageIV_shapefile_paths[170])
print(mrms_shapefile_paths[170])

daily_measurements = []


'''
p=0
while p < len(gauge_data):

    #run function
    points_gdf = read_csv_and_create_geodataframe(coordinates, gauge_data[p])
    #print(points_gdf)
    
    
    # Load Stage IV shapefiles containing polygons into a GeoDataFrame
    polygons_gdf = gpd.read_file(stageIV_shapefile_paths[p])
    
    #print(polygons_gdf)
    
    # Filter out polygons with precip values less than zero
    polygons_gdf = polygons_gdf[polygons_gdf['stageIV_pr'] >= 0]
    
    #print(polygons_gdf)
    
    polygons_gdf = polygons_gdf.to_crs(points_gdf.crs) # makes them both in the same coordinate system (the one used by the points)
    
    # Spatial index for polygons
    polygons_sindex = polygons_gdf.sindex
    
    # Reset index of points GeoDataFrame
    points_gdf = points_gdf.reset_index(drop=True)
    
    # Spatial join between polygons and points
    sjoined = gpd.sjoin(polygons_gdf, points_gdf, op='contains', how='inner')
    
    # Convert latitude and longitude to Point objects
    #sjoined['point_geometry'] = [Point(lon, lat) for lon, lat in zip(sjoined['Longitude'], sjoined['Latitude'])]
                
    #print(sjoined) # contains the stage IV polygon geometry (under "geometry") and the rain gauge points (under "point geometry)
    
    
    # This little section calculates avergae gauge values for pixels with more than one gauge
    # Group by the index and calculate the average Gauge_precip
    avg_precip = sjoined.groupby(sjoined.index)['Gauge_precip'].mean()
    
    # Update one row in each group with the average value and drop duplicates
    for idx, avg_value in avg_precip.items():
        sjoined.loc[idx, 'Gauge_precip'] = avg_value
    
    sjoined = sjoined[~sjoined.index.duplicated(keep='first')]  # Drop duplicate rows
    
    
    # Load the MRMS shapefile containing polygons 
    mrms_polygons_gdf = gpd.read_file(mrms_shapefile_paths[p])
        
    mrms_polygons_gdf = mrms_polygons_gdf.rename(columns={'geometry': 'mrms_geometry'})
    
    stageIV_geometries_gdf = sjoined.drop(columns=['stageIV_pr','index_right','Latitude','Longitude','Gauge_precip'])
    stageIV_geometries_gdf['stageIV_precip'] = sjoined['stageIV_pr']
    #print(stageIV_geometries_gdf)
        
    # First, perform a spatial join to find the intersection
    
    stageIV_geometry = stageIV_geometries_gdf.set_geometry('geometry')
    mrms_geometry = mrms_polygons_gdf.set_geometry('mrms_geometry')
    
    #print(stageIV_geometries_gdf)
    
    intersection = gpd.sjoin(stageIV_geometry, mrms_geometry , how="inner", op="intersects") #performs the spatial join
    
    # Now, you have a GeoDataFrame containing the intersecting polygons and their attributes from both dataframes.
    # You can select the columns you want to keep and create a new dataframe if needed.
    
    
    # Create a dictionary to store Stage IV polygon IDs and their corresponding geometries
    stageIV_polygons_dict = {index: geometry for index, geometry in zip(stageIV_geometry.index, stageIV_geometry.geometry)}
    
    # Add a new column to store Stage IV polygon IDs in the intersection GeoDataFrame
    intersection['stageIV_polygon_id'] = None
    
    # Iterate over each MRMS polygon in the intersection GeoDataFrame
    for idx, mrms_polygon in intersection.iterrows():
        # Iterate over each Stage IV polygon to find the intersection
        for stageIV_polygon_id, stageIV_polygon_geom in stageIV_polygons_dict.items():
            if mrms_polygon.geometry.intersects(stageIV_polygon_geom):
                # Assign the Stage IV polygon ID to the MRMS polygon
                intersection.at[idx, 'stageIV_polygon_id'] = stageIV_polygon_id
                break
    
    # Now, intersection GeoDataFrame contains the MRMS polygons with a new column 'stageIV_polygon_id' 
    # corresponding to the Stage IV polygon ID they intersect with.
    
    
    #print(intersection)
    mean_precipitation = intersection.groupby('stageIV_polygon_id')['mrms_preci'].mean()
    
    #print(mean_precipitation)
    
    # Concatenate the existing dataframe with the sorted mean precipitation dataframe along the columns axis
    #sjoined_reset_index = sjoined.reset_index(drop=True)
    combined_dataframe = pd.concat([sjoined, mean_precipitation], axis=1)
    
    
    combined_dataframe['stageIV_pr'] = combined_dataframe['stageIV_pr']*25.4 # converts stage IV values to mm from inches
    
    print(combined_dataframe)
    
    
    #print(combined_dataframe)
    
    # Define a function to check if precipitation values are within 90% to 110% of each other's value
    def within_range(val1, val2):
        return 0.9 * val1 <= val2 <= 1.1 * val1 or 0.9 * val2 <= val1 <= 1.1 * val2
    
    # Filter out rows where gauge values are not within 90% to 110% of each other's value
    filtered_df = combined_dataframe[combined_dataframe.apply(lambda row: within_range(row['Gauge_precip'], row['mrms_preci']), axis=1)]
    
    #filtered_df['Bias'] = filtered_df['rainfall']/filtered_df['Precipitation']
    
    gauge_precip = pd.read_csv(gauge_data[p], index_col=0)
    
    filtered_df['date'] = gauge_precip.index[0]
    
    print(filtered_df)
    
    daily_measurements.append(filtered_df)
    
    p=p+1
'''

import multiprocessing

def process_data(p, gauge_data, stageIV_shapefile_paths, mrms_shapefile_paths, coordinates, daily_measurements):
    points_gdf = read_csv_and_create_geodataframe(coordinates, gauge_data[p])
    
    polygons_gdf = gpd.read_file(stageIV_shapefile_paths[p])
    polygons_gdf = polygons_gdf[polygons_gdf['stageIV_pr'] >= 0]
    polygons_gdf = polygons_gdf.to_crs(points_gdf.crs)
    polygons_sindex = polygons_gdf.sindex
    
    points_gdf = points_gdf.reset_index(drop=True)
    sjoined = gpd.sjoin(polygons_gdf, points_gdf, predicate='contains', how='inner')
    avg_precip = sjoined.groupby(sjoined.index)['Gauge_precip'].mean()
    
    for idx, avg_value in avg_precip.items():
        sjoined.loc[idx, 'Gauge_precip'] = avg_value
    
    sjoined = sjoined[~sjoined.index.duplicated(keep='first')]
    
    mrms_polygons_gdf = gpd.read_file(mrms_shapefile_paths[p])
    mrms_polygons_gdf = mrms_polygons_gdf.rename(columns={'geometry': 'mrms_geometry'})
    
    stageIV_geometries_gdf = sjoined.drop(columns=['stageIV_pr','index_right','Latitude','Longitude','Gauge_precip'])
    stageIV_geometries_gdf['stageIV_precip'] = sjoined['stageIV_pr']
    
    stageIV_geometry = stageIV_geometries_gdf.set_geometry('geometry')
    mrms_geometry = mrms_polygons_gdf.set_geometry('mrms_geometry')
    
    intersection = gpd.sjoin(stageIV_geometry, mrms_geometry , how="inner", predicate="intersects")
    
    stageIV_polygons_dict = {index: geometry for index, geometry in zip(stageIV_geometry.index, stageIV_geometry.geometry)}
    
    intersection['stageIV_polygon_id'] = None
    
    for idx, mrms_polygon in intersection.iterrows():
        for stageIV_polygon_id, stageIV_polygon_geom in stageIV_polygons_dict.items():
            if mrms_polygon.geometry.intersects(stageIV_polygon_geom):
                intersection.at[idx, 'stageIV_polygon_id'] = stageIV_polygon_id
                break
    
    mean_precipitation = intersection.groupby('stageIV_polygon_id')['mrms_preci'].mean()
    
    combined_dataframe = pd.concat([sjoined, mean_precipitation], axis=1)
    combined_dataframe['stageIV_pr'] = combined_dataframe['stageIV_pr']*25.4
    
    def within_range(val1, val2):
        return 0.9 * val1 <= val2 <= 1.1 * val1 or 0.9 * val2 <= val1 <= 1.1 * val2
    
    filtered_df = combined_dataframe[combined_dataframe.apply(lambda row: within_range(row['Gauge_precip'], row['mrms_preci']), axis=1)]
    
    gauge_precip = pd.read_csv(gauge_data[p], index_col=0)
    filtered_df['date'] = gauge_precip.index[0]
    
    #daily_measurements.append(filtered_df)
    
    #print(filtered_df)
    
    return filtered_df
    

if __name__ == "__main__":
    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes)
    
    daily_measurements = []
    results=[]
    
    for p in range(len(gauge_data)):
        result = pool.apply_async(process_data, args=(p, gauge_data, stageIV_shapefile_paths, mrms_shapefile_paths, coordinates, daily_measurements))
        results.append(result)
        print(p)
    
    pool.close()
    pool.join()
    
    for result in results:
        processed_data = result.get()
        daily_measurements.append(processed_data)
    
    # Concatenate DataFrames horizontally
    precip_values_2022 = pd.concat(daily_measurements, ignore_index=True)
    
    precip_values_2022.to_csv('/homes/metogra/dbrooks6/wpc_research/final_analysis_data/arkansas_red_2022_data.csv', index=True) # top row is the station IDs
    
    # Now daily_measurements should contain results from all processed data









