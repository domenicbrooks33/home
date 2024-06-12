import pandas as pd
import uscrn
from datetime import datetime, timedelta

df = uscrn.get_data(2022, "hourly", n_jobs=6)  # select year you want data from

unique_values = df['wban'].unique() # use this list to identify the different stations

stations_with_hourlyprecip = df[['wban','utc_time','longitude','latitude','p_calc']] # creates dataframe

# Convert timestamp column to datetime objects
stations_with_hourlyprecip['utc_time'] = pd.to_datetime(stations_with_hourlyprecip['utc_time'])

# Create an empty dictionary to store DataFrames for each station
station_dfs = {}

# Group the data by station_id
grouped = stations_with_hourlyprecip.groupby('wban')

# Create a DataFrame for each station and store it in the dictionary
for station, data in grouped:
    station_dfs[station] = data

list_of_dfs = [] # list to store the dataframes of precip data for each station

lat=[] # stores the latitude for each station
lon=[] # stores the longitude for each station
station_ids=[] # stores the unique station ids
 
i=0
while i<len(station_dfs):

    # Gets a dataframe of a all hours in a year for a given station, as indicated by the unique_values
    # Delete first 11 rows
    station_dfs[unique_values[i]] = station_dfs[unique_values[i]].iloc[12:]

    # Delete last 11 rows
    station_dfs[unique_values[i]] = station_dfs[unique_values[i]].iloc[:-12]

    station_dfs[unique_values[i]] = station_dfs[unique_values[i]].reset_index(drop=True)

    station_id = station_dfs[unique_values[i]]['wban'].unique() # gets the station id to use in the final dataframe

    ########################################################
    station_ids.append(station_id[0])
    
    lat.append(float(station_dfs[unique_values[i]]['latitude'].unique()))
    lon.append(float(station_dfs[unique_values[i]]['longitude'].unique()))
    
    ########################################################

    # Function to assign the same identifier to all rows within each group of 24 rows
    def assign_identifier(group):
        group['Identifier'] = group.index // 24  # Assigning identifier based on group index
        return group

    # creates dataframe with the identifiers
    result = station_dfs[unique_values[i]].groupby(station_dfs[unique_values[0]].index // 24).apply(assign_identifier)



    # Calculate sum of 'Value' column by identifier
    sum_by_identifier = result.groupby('Identifier')['p_calc'].sum().reset_index()

    sum_by_identifier = sum_by_identifier.drop(['Identifier'], axis=1)


    # just renames the precip column to the station id, makes for better organization
    sum_by_identifier.rename(columns={'p_calc': station_id[0]}, inplace=True) 
    
    #print(sum_by_identifier)
    list_of_dfs.append(sum_by_identifier)
    print(i)
    
    i=i+1
    
# Concatenate DataFrames horizontally
stations_with_precip = pd.concat(list_of_dfs, axis=1)


# Start and end dates
start_date = datetime(2022, 1, 2)
end_date = datetime(2022, 12, 31)

# List to store datetime objects
date_list = []

# Loop through dates and add to list
current_date = start_date
while current_date <= end_date:
    date_list.append(current_date)
    current_date += timedelta(days=1)

# Create a DataFrame from the date list with the same index as existing_df
new_df = pd.DataFrame({'Date': date_list}, index=sum_by_identifier.index)

# Add the new DataFrame to the existing DataFrame
#updated_df = pd.concat([stations_with_precip, new_df], axis=1)
stations_with_precip.index = new_df['Date'] 

print(stations_with_precip)


# Export each row to a separate CSV file
for index, row in stations_with_precip.iterrows():
    date_str = index.strftime('%Y-%m-%d')
    # Create a filename based on the index
    filename = f"/homes/metogra/dbrooks6/wpc_research/uscrn_data/2022//raingauges_{date_str}.csv"
    # Create a DataFrame with only the current row
    row_df = pd.DataFrame([row])
    # Save the row DataFrame to a CSV file
    row_df.to_csv(filename, index=True)
'''
# Export DataFrame to CSV file
stations_with_precip.to_csv('rain_gauge_values_2022.csv', index=True) # top row is the station IDs
'''
# Create the dataframe for station coordinates
station_coords = pd.DataFrame({"Station_id": station_ids, "Latitude": lat, "Longitude": lon})
print(station_coords)

station_coords.to_csv('rain_gauge_coordinates.csv', index=True) # top row is the station IDs