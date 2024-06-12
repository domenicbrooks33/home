import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import box
from rasterio.warp import calculate_default_transform, reproject, Resampling
import glob

def reproject_shapefile(shapefile_path, target_crs):
    # Read shapefile
    gdf = gpd.read_file(shapefile_path)
    
    # Reproject to target CRS
    gdf_reproj = gdf.to_crs(target_crs)
    
    return gdf_reproj

def clip_raster(raster_path, shapefile_path, output_path):
    # Open raster dataset
    with rasterio.open(raster_path) as src:
        # Reproject shapefile to match raster CRS
        target_crs = src.crs
        shapefile_reproj = reproject_shapefile(shapefile_path, target_crs)
        
        # Get the extent of the raster
        raster_bounds = src.bounds
        
        # Check if the shapefile intersects with the raster extent
        #if not shapefile_reproj.geometry.intersects(box(*raster_bounds)):
            #raise ValueError("Shapefile does not intersect with raster extent.")
        
        # Extract the geometry from the reprojected shapefile (assuming it has only one feature)
        geom = shapefile_reproj.geometry.values[0]
        
        # Apply the polygon as a mask to the raster data
        out_image, out_transform = mask(src, [geom], crop=True)
        
        # Update the metadata for the clipped raster
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })
        
        # Write the clipped raster to a new GeoTIFF file
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)

raster_path = glob.glob("/homes/metogra/dbrooks6/wpc_research/stageIV_raw_geotiff_data/2022/*.tif")
shapefile_path = "/homes/metogra/dbrooks6/wpc_research/rfc_boundaries/west_gulf.shp"

i=0
while i < len(raster_path):

  string = raster_path[i][84:92]
  #print(string)
  output_raster_path = "/homes/metogra/dbrooks6/wpc_research/stageIVqpe_data_rfc/west_gulf___/2022/west_gulf____stageIV_data_{0}.tif".format(string)

  # Perform the raster clipping
  clip_raster(raster_path[i], shapefile_path, output_raster_path)

  print("Clipping complete. Clipped raster saved to:", output_raster_path)
  i=i+1


            
            
            

