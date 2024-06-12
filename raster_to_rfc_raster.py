# Clips raster data to only be within the RFC boundary

import rasterio
from rasterio.mask import mask
import geopandas as gpd
import glob

def clip_raster(raster_path, shapefile_path, output_path):
    # Open raster dataset
    with rasterio.open(raster_path) as src:
        # Read the shapefile containing the clipping polygon
        shapefile = gpd.read_file(shapefile_path)
        
        # Extract the geometry from the shapefile (assuming it has only one feature)
        geom = shapefile.geometry.values[0]
        
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
# Specify paths to input raster, shapefile, and output raster
raster_path = glob.glob("/homes/metogra/dbrooks6/wpc_research/mrms_raw_tif_files/2022/output*.tif")
shapefile_path = "/homes/metogra/dbrooks6/wpc_research/rfc_boundaries/west_gulf.shp"

i=0
while i < len(raster_path):

  string = raster_path[i][78:86]
  #print(string)
  output_raster_path = "/homes/metogra/dbrooks6/wpc_research/mrms_data_rfc/west_gulf___/2022/west_gulf_____mrms_data_{0}.tif".format(string)

  # Perform the raster clipping
  clip_raster(raster_path[i], shapefile_path, output_raster_path)

  print("Clipping complete. Clipped raster saved to:", output_raster_path)
  i=i+1