import cfgrib
import xarray as xr
import rasterio
from rasterio.transform import from_origin
import glob

# reads in grib file and returns the gridded pixel values
def read_grib_file(file_path):
    # Open the GRIB file
    xds = xr.open_dataset(file_path, engine='cfgrib')


    # Access variables in the GRIB file
    #for var in xds.variables:
        #print(var)
        
    # Access 24 precip data
    precip_data = xds['unknown']
    

    return precip_data # returns the array of precip values
    
    
# function that creates the parameters for the geotiff file for the precipitation data
def array_to_geotiff(array, output_file, transform, crs, extent):
    # Get shape and metadata from the array
    height, width = array.shape
    count = 1  # number of bands
    dtype = array.dtype

    # Write the array to a GeoTIFF file
    with rasterio.open(
        output_file,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=count,
        dtype=dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(array, 1)
        
        # Update the extent if provided
        if extent:
            dst.update_tags(ns='IMAGE_STRUCTURE', bounds=extent)

# Provide the path to your GRIB file
grib_file_path = glob.glob("/homes/metogra/dbrooks6/wpc_research/mrms_raw_data/2022/*.grib2")

#print(grib_file_path[0])

#string = grib_file_path[0][93:101]

#print(string)


i=0
while i < len(grib_file_path):
    precip_data = read_grib_file(grib_file_path[i])
    
    string = grib_file_path[i][93:101]
    
    # Creates the raster data file from the data array in the GRIB file
    
    # Output GeoTIFF file path
    output_geotiff_path = "/homes/metogra/dbrooks6/wpc_research/mrms_raw_tif_files/2022/output_mrms_data_{0}.tif".format(string)
    
    # Assuming you know the coordinate reference system (CRS) and the transform of the data
    # Replace these values with your actual CRS and transform
    crs = '+proj=longlat +datum=WGS84 +no_defs'
    
    # Assuming you know the extent of the data, specify it as [xmin, ymin, xmax, ymax]
    # Replace these values with the actual extent of your data
    extent = [-130.000000, 20.000001,-60.000002,55.000000]
    
    # Define custom transform based on the extent
    transform = rasterio.transform.from_bounds(*extent, precip_data.shape[1], precip_data.shape[0])
    
    # Convert the temperature data array to a GeoTIFF file
    array_to_geotiff(precip_data.values, output_geotiff_path, transform, "EPSG:4326", extent)
    
    i=i+1
    
    print("GeoTIFF file saved successfully:", output_geotiff_path)

