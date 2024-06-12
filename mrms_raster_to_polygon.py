# creates polygons of each pixel in the raster data
import rasterio
import geopandas as gpd
from shapely.geometry import box
from multiprocessing import Pool
from tqdm import tqdm
import glob

# Input raster file path
input_raster_path = glob.glob("/homes/metogra/dbrooks6/wpc_research/mrms_data_rfc/arkansas_red/2022/arkansas*.tif")


#print(string)


i=0
while i<len(input_raster_path):

    # Open the raster file
    with rasterio.open(input_raster_path[i]) as src:
        # Extract meta-data
        transform = src.transform
        crs = src.crs
        pixel_size = src.res[0]  # Assuming square pixels
    
        p=1
        # Generate polygons from raster
        polygons = []
        for ji, window in src.block_windows(1):
            data = src.read(1, window=window)
            #print(p)
            p = p + 1
            for y in range(window.height):
                for x in range(window.width):
                    value = data[y, x]
                    if value is not None:  # You might want to handle nodata values here
                        # Calculate pixel coordinates
                        x_min = transform[2] + (window.col_off + x) * pixel_size
                        y_max = transform[5] - (window.row_off + y) * pixel_size
                        x_max = x_min + pixel_size
                        y_min = y_max - pixel_size
                        
                        # Create the polygon for the pixel
                        polygon = box(x_min, y_min, x_max, y_max)
                        polygons.append({'geometry': polygon, 'mrms_precip': value})
    
    print('done1')
    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(polygons, crs=crs)
    print('done2')
    
    string = input_raster_path[i][92:100]
    # Output shapefile path
    output_shapefile_path = "/homes/metogra/dbrooks6/wpc_research/mrms_data_rfc/arkansas_red/2022/shapefiles/arkansas_mrms_polygons_{0}.shp".format(string)
    
    # Save to shapefile
    gdf.to_file(output_shapefile_path)
    
    print('Successfully saved as:', output_shapefile_path)
    
    i=i+1


# This is the command line code to download the MRMS files from the AWS server
# aws s3 ls --no-sign-request s3://noaa-mrms-pds/CONUS/MultiSensor_QPE_24H_Pass2_00.00/ --recursive | grep 'MRMS_MultiSensor_QPE_24H_Pass2_00.00_2022.*-120000.grib2.gz' | awk '{print "http://noaa-mrms-pds.s3.amazonaws.com/" $4}' | xargs -n 1 wget

# extracts grib from .gz
# for file in /homes/metogra/dbrooks6/wpc_research/mrms_raw_data/2022/*.gz; do gunzip -c "$file" > "${file%.gz}"; done
