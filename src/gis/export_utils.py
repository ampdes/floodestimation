"""
GIS export utilities for flood analysis results.

This module provides functions to export flood masks to various GIS formats:
- GeoTIFF (georeferenced raster)
- Shapefile (vector polygons)
- GeoJSON (vector polygons)
- KML (Google Earth)
"""

import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.transform import from_bounds
import geopandas as gpd
from shapely.geometry import shape, Polygon
import fiona
from pathlib import Path
import json


def export_to_geotiff(
    mask: np.ndarray,
    output_path: str,
    transform: rasterio.Affine,
    crs: str = 'EPSG:4326',
    compression: str = 'lzw',
    nodata: int = 0
):
    """
    Export flood mask to GeoTIFF format.

    Args:
        mask: Binary mask (HW) or float values
        output_path: Path to save GeoTIFF
        transform: Affine transform for georeferencing
        crs: Coordinate reference system (e.g., 'EPSG:4326')
        compression: Compression method ('lzw', 'deflate', 'none')
        nodata: NoData value
    """
    # Ensure mask is 2D
    if mask.ndim == 3:
        mask = mask.squeeze()

    # Determine data type
    if mask.dtype == np.float32 or mask.dtype == np.float64:
        dtype = rasterio.float32
    else:
        dtype = rasterio.uint8

    # Write GeoTIFF
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=mask.shape[0],
        width=mask.shape[1],
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
        compress=compression,
        nodata=nodata
    ) as dst:
        dst.write(mask.astype(dtype), 1)

    print(f"GeoTIFF exported to: {output_path}")


def vectorize_mask(
    mask: np.ndarray,
    transform: rasterio.Affine,
    min_area: float = 100.0
):
    """
    Convert raster mask to vector polygons.

    Args:
        mask: Binary mask (HW)
        transform: Affine transform for coordinates
        min_area: Minimum area threshold (in map units squared)

    Returns:
        List of (geometry, value) tuples
    """
    # Ensure binary
    if mask.max() > 1:
        mask = (mask > 127).astype(np.uint8)

    # Extract shapes
    polygons = []
    for geom, value in shapes(mask, transform=transform):
        if value == 1:  # Only flood pixels
            poly = shape(geom)
            # Filter by area
            if poly.area >= min_area:
                polygons.append(poly)

    return polygons


def export_to_shapefile(
    mask: np.ndarray,
    output_path: str,
    transform: rasterio.Affine,
    crs: str = 'EPSG:4326',
    attributes: dict = None,
    min_area: float = 100.0
):
    """
    Export flood mask to Shapefile format.

    Args:
        mask: Binary mask (HW)
        output_path: Path to save shapefile
        transform: Affine transform for georeferencing
        crs: Coordinate reference system
        attributes: Dictionary of attributes to add to each polygon
        min_area: Minimum area threshold
    """
    # Vectorize mask
    polygons = vectorize_mask(mask, transform, min_area)

    if not polygons:
        print("No polygons found (all areas below threshold)")
        return

    # Create GeoDataFrame
    gdf_data = {'geometry': polygons}

    # Add attributes
    if attributes:
        for key, value in attributes.items():
            if isinstance(value, (list, np.ndarray)):
                # If list/array, must match polygon count
                if len(value) == len(polygons):
                    gdf_data[key] = value
                else:
                    print(f"Warning: Attribute '{key}' length mismatch, skipping")
            else:
                # Single value, apply to all polygons
                gdf_data[key] = [value] * len(polygons)

    # Calculate area and perimeter for each polygon
    gdf_data['area_m2'] = [poly.area for poly in polygons]
    gdf_data['perimeter_m'] = [poly.length for poly in polygons]

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(gdf_data, crs=crs)

    # Save to shapefile
    gdf.to_file(output_path, driver='ESRI Shapefile')

    print(f"Shapefile exported to: {output_path}")
    print(f"  Number of polygons: {len(polygons)}")
    print(f"  Total area: {gdf['area_m2'].sum():.2f} m²")


def export_to_geojson(
    mask: np.ndarray,
    output_path: str,
    transform: rasterio.Affine,
    crs: str = 'EPSG:4326',
    attributes: dict = None,
    min_area: float = 100.0
):
    """
    Export flood mask to GeoJSON format.

    Args:
        mask: Binary mask (HW)
        output_path: Path to save GeoJSON
        transform: Affine transform for georeferencing
        crs: Coordinate reference system
        attributes: Dictionary of attributes
        min_area: Minimum area threshold
    """
    # Vectorize mask
    polygons = vectorize_mask(mask, transform, min_area)

    if not polygons:
        print("No polygons found (all areas below threshold)")
        return

    # Create GeoDataFrame
    gdf_data = {'geometry': polygons}

    # Add attributes
    if attributes:
        for key, value in attributes.items():
            if isinstance(value, (list, np.ndarray)):
                if len(value) == len(polygons):
                    gdf_data[key] = value
            else:
                gdf_data[key] = [value] * len(polygons)

    # Calculate area and perimeter
    gdf_data['area_m2'] = [poly.area for poly in polygons]
    gdf_data['perimeter_m'] = [poly.length for poly in polygons]

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(gdf_data, crs=crs)

    # Save to GeoJSON
    gdf.to_file(output_path, driver='GeoJSON')

    print(f"GeoJSON exported to: {output_path}")
    print(f"  Number of features: {len(polygons)}")


def export_to_kml(
    mask: np.ndarray,
    output_path: str,
    transform: rasterio.Affine,
    crs: str = 'EPSG:4326',
    name: str = 'Flood Extent',
    min_area: float = 100.0
):
    """
    Export flood mask to KML format for Google Earth.

    Args:
        mask: Binary mask (HW)
        output_path: Path to save KML
        transform: Affine transform
        crs: Coordinate reference system
        name: Name for the KML layer
        min_area: Minimum area threshold
    """
    try:
        import simplekml
    except ImportError:
        print("simplekml not installed. Install with: pip install simplekml")
        return

    # Vectorize mask
    polygons = vectorize_mask(mask, transform, min_area)

    if not polygons:
        print("No polygons found")
        return

    # Create KML
    kml = simplekml.Kml()

    # Add polygons
    for i, poly in enumerate(polygons):
        # Convert to WGS84 if needed
        if crs != 'EPSG:4326':
            import pyproj
            from shapely.ops import transform as shapely_transform

            project = pyproj.Transformer.from_crs(crs, 'EPSG:4326', always_xy=True).transform
            poly = shapely_transform(project, poly)

        # Extract coordinates
        coords = list(poly.exterior.coords)

        # Create polygon in KML
        pol = kml.newpolygon(name=f"{name} {i+1}")
        pol.outerboundaryis = coords
        pol.style.polystyle.color = simplekml.Color.changealphaint(100, simplekml.Color.cyan)
        pol.style.polystyle.outline = 1

    # Save KML
    kml.save(output_path)

    print(f"KML exported to: {output_path}")


def load_georeferencing_from_tiff(tiff_path: str):
    """
    Load georeferencing information from a GeoTIFF.

    Args:
        tiff_path: Path to GeoTIFF file

    Returns:
        Dictionary with 'transform', 'crs', 'bounds', 'shape'
    """
    with rasterio.open(tiff_path) as src:
        return {
            'transform': src.transform,
            'crs': src.crs.to_string() if src.crs else None,
            'bounds': src.bounds,
            'shape': (src.height, src.width),
            'res': src.res
        }


def create_transform_from_bounds(
    bounds: tuple,
    width: int,
    height: int
):
    """
    Create affine transform from bounding box and dimensions.

    Args:
        bounds: (left, bottom, right, top)
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        Affine transform
    """
    return from_bounds(*bounds, width, height)


def export_all_formats(
    mask: np.ndarray,
    output_dir: str,
    basename: str,
    transform: rasterio.Affine,
    crs: str = 'EPSG:4326',
    formats: list = None,
    attributes: dict = None,
    min_area: float = 100.0
):
    """
    Export flood mask to multiple formats.

    Args:
        mask: Binary mask (HW)
        output_dir: Directory to save files
        basename: Base name for output files
        transform: Affine transform
        crs: Coordinate reference system
        formats: List of formats to export ['geotiff', 'shapefile', 'geojson', 'kml']
        attributes: Dictionary of attributes
        min_area: Minimum area threshold
    """
    if formats is None:
        formats = ['geotiff', 'shapefile', 'geojson']

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results = {}

    # Export to each format
    if 'geotiff' in formats:
        path = str(Path(output_dir) / f"{basename}.tif")
        export_to_geotiff(mask, path, transform, crs)
        results['geotiff'] = path

    if 'shapefile' in formats:
        path = str(Path(output_dir) / f"{basename}.shp")
        export_to_shapefile(mask, path, transform, crs, attributes, min_area)
        results['shapefile'] = path

    if 'geojson' in formats:
        path = str(Path(output_dir) / f"{basename}.geojson")
        export_to_geojson(mask, path, transform, crs, attributes, min_area)
        results['geojson'] = path

    if 'kml' in formats:
        path = str(Path(output_dir) / f"{basename}.kml")
        export_to_kml(mask, path, transform, crs, basename, min_area)
        results['kml'] = path

    print(f"\nAll formats exported to: {output_dir}")

    return results


if __name__ == "__main__":
    """Test GIS export utilities."""
    print("Testing GIS export utilities...")

    # Create dummy mask
    mask = np.random.randint(0, 2, (1000, 1000), dtype=np.uint8)

    # Create dummy transform (1 meter resolution, starting at 0,0)
    transform = rasterio.transform.from_origin(0, 1000, 1, 1)

    # Test exports
    output_dir = "outputs/gis/test"

    try:
        export_all_formats(
            mask=mask,
            output_dir=output_dir,
            basename="test_flood",
            transform=transform,
            crs='EPSG:4326',
            formats=['geotiff', 'shapefile', 'geojson'],
            attributes={'test': 'value'},
            min_area=10.0
        )

        print("\nGIS export tests completed!")

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
