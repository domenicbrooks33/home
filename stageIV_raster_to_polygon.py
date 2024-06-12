# creates polygons of each pixel in the raster data
import rasterio
import geopandas as gpd
from shapely.geometry import box
from multiprocessing import Pool
from tqdm import tqdm
import glob

# Input raster file path
input_raster_path = glob.glob("/homes/metogra/dbrooks6/wpc_research/stageIVqpe_data_rfc/cali_nevada_/2022/*.tif")


#print(string)


i=124
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
                        polygons.append({'geometry': polygon, 'stageIV_precip': value})
    
    print('done1')
    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(polygons, crs=crs)
    print('done2')
    
    string = input_raster_path[i][100:108]
    # Output shapefile path
    output_shapefile_path = "/homes/metogra/dbrooks6/wpc_research/stageIVqpe_data_rfc/cali_nevada_/2022/shapefiles/cali_nevada_stageIV_polygons_{0}.shp".format(string)
    
    # Save to shapefile
    gdf.to_file(output_shapefile_path)
    
    print('Successfully saved as:', output_shapefile_path)
    
    i=i+1