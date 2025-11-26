"""
Flood extent analysis module.

This module calculates flood extent area from predicted masks using georeferencing
information from the source satellite imagery.
"""

import numpy as np
import rasterio
from rasterio.transform import Affine
import pandas as pd
from pathlib import Path
import json
from typing import Dict, Tuple
import cv2


class FloodExtentAnalyzer:
    """
    Class for analyzing flood extent from segmentation masks.
    """

    def __init__(
        self,
        source_tiff: str = None,
        gsd: float = 3.8,
        crs: str = 'EPSG:4326'
    ):
        """
        Initialize flood extent analyzer.

        Args:
            source_tiff: Path to source GeoTIFF with georeferencing
            gsd: Ground Sample Distance in meters (if source_tiff not provided)
            crs: Coordinate Reference System
        """
        self.source_tiff = source_tiff
        self.gsd = gsd
        self.crs = crs

        # Load georeferencing if source TIFF provided
        if source_tiff and Path(source_tiff).exists():
            self.load_georeferencing()
        else:
            self.transform = None
            self.bounds = None

    def load_georeferencing(self):
        """Load georeferencing information from source TIFF."""
        try:
            with rasterio.open(self.source_tiff) as src:
                self.transform = src.transform
                self.bounds = src.bounds
                self.crs = src.crs.to_string() if src.crs else self.crs
                self.gsd = src.res[0]  # Assume square pixels

            print(f"Loaded georeferencing from {self.source_tiff}")
            print(f"  CRS: {self.crs}")
            print(f"  GSD: {self.gsd} meters")
            print(f"  Bounds: {self.bounds}")

        except Exception as e:
            print(f"Error loading georeferencing: {e}")
            self.transform = None
            self.bounds = None

    def calculate_pixel_area(self, mask: np.ndarray) -> float:
        """
        Calculate area of a single pixel in square meters.

        Args:
            mask: Flood mask (used to check if CRS needs conversion)

        Returns:
            Pixel area in square meters
        """
        # For geographic CRS (lat/lon), need to convert to meters
        # For projected CRS, use GSD directly

        if 'EPSG:4326' in self.crs or 'WGS84' in self.crs:
            # Geographic coordinates - use approximate conversion
            # At equator, 1 degree ≈ 111 km
            # This is approximate and varies with latitude
            pixel_area_m2 = self.gsd * self.gsd
        else:
            # Projected coordinates - use GSD directly
            pixel_area_m2 = self.gsd * self.gsd

        return pixel_area_m2

    def calculate_flood_extent(
        self,
        mask: np.ndarray,
        min_area_threshold: float = 0.0
    ) -> Dict:
        """
        Calculate flood extent from binary mask.

        Args:
            mask: Binary flood mask (HW), where 1 = flooded, 0 = not flooded
            min_area_threshold: Minimum area threshold in m²

        Returns:
            Dictionary with extent statistics
        """
        # Ensure binary
        if mask.max() > 1:
            mask = (mask > 127).astype(np.uint8)

        # Count flooded pixels
        flooded_pixels = np.sum(mask > 0)

        # Calculate pixel area
        pixel_area_m2 = self.calculate_pixel_area(mask)

        # Calculate total area
        total_area_m2 = flooded_pixels * pixel_area_m2
        total_area_km2 = total_area_m2 / 1e6

        # Calculate coverage percentage
        total_pixels = mask.size
        coverage_percent = (flooded_pixels / total_pixels) * 100

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        # Analyze components (skip background label 0)
        components = []
        for i in range(1, num_labels):
            component_pixels = stats[i, cv2.CC_STAT_AREA]
            component_area_m2 = component_pixels * pixel_area_m2

            # Filter by minimum area
            if component_area_m2 >= min_area_threshold:
                components.append({
                    'component_id': i,
                    'pixels': component_pixels,
                    'area_m2': component_area_m2,
                    'area_km2': component_area_m2 / 1e6,
                    'centroid_x': centroids[i][0],
                    'centroid_y': centroids[i][1],
                    'bbox_left': stats[i, cv2.CC_STAT_LEFT],
                    'bbox_top': stats[i, cv2.CC_STAT_TOP],
                    'bbox_width': stats[i, cv2.CC_STAT_WIDTH],
                    'bbox_height': stats[i, cv2.CC_STAT_HEIGHT]
                })

        # Sort components by area
        components = sorted(components, key=lambda x: x['area_m2'], reverse=True)

        # Compile results
        results = {
            'total_area_m2': float(total_area_m2),
            'total_area_km2': float(total_area_km2),
            'flooded_pixels': int(flooded_pixels),
            'total_pixels': int(total_pixels),
            'coverage_percent': float(coverage_percent),
            'pixel_area_m2': float(pixel_area_m2),
            'gsd_meters': float(self.gsd),
            'num_components': len(components),
            'components': components,
            'crs': self.crs
        }

        return results

    def analyze_multiple_masks(
        self,
        mask_paths: list,
        output_csv: str = None
    ) -> pd.DataFrame:
        """
        Analyze multiple flood masks and compile results.

        Args:
            mask_paths: List of paths to mask images
            output_csv: Path to save results CSV (optional)

        Returns:
            DataFrame with results
        """
        results_list = []

        for mask_path in mask_paths:
            # Load mask
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if mask is None:
                print(f"Failed to load {mask_path}")
                continue

            # Analyze
            result = self.calculate_flood_extent(mask)

            # Add filename
            result['filename'] = Path(mask_path).name

            results_list.append(result)

        # Create DataFrame
        df = pd.DataFrame(results_list)

        # Save to CSV if requested
        if output_csv:
            df.to_csv(output_csv, index=False)
            print(f"Results saved to {output_csv}")

        return df

    def create_extent_map(
        self,
        mask: np.ndarray,
        output_path: str = None
    ) -> np.ndarray:
        """
        Create a visualization of flood extent with components labeled.

        Args:
            mask: Binary flood mask
            output_path: Path to save visualization

        Returns:
            Colored extent map
        """
        # Ensure binary
        if mask.max() > 1:
            mask = (mask > 127).astype(np.uint8)

        # Find connected components
        num_labels, labels = cv2.connectedComponents(mask, connectivity=8)

        # Create colored map
        colored = np.zeros((*mask.shape, 3), dtype=np.uint8)

        # Assign colors to components
        np.random.seed(42)
        for i in range(1, num_labels):
            color = np.random.randint(0, 255, 3, dtype=np.uint8)
            colored[labels == i] = color

        # Save if requested
        if output_path:
            cv2.imwrite(output_path, colored)

        return colored

    def print_summary(self, results: Dict):
        """
        Print formatted summary of flood extent analysis.

        Args:
            results: Results dictionary from calculate_flood_extent
        """
        print("\n" + "="*60)
        print("FLOOD EXTENT ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total Flooded Area:     {results['total_area_m2']:,.2f} m²")
        print(f"                        {results['total_area_km2']:.4f} km²")
        print(f"Coverage:               {results['coverage_percent']:.2f}%")
        print(f"Flooded Pixels:         {results['flooded_pixels']:,}")
        print(f"Pixel Resolution:       {results['gsd_meters']:.2f} m")
        print(f"Number of Components:   {results['num_components']}")

        if results['num_components'] > 0:
            print(f"\nTop 5 Components by Area:")
            for i, comp in enumerate(results['components'][:5], 1):
                print(f"  {i}. Area: {comp['area_m2']:,.2f} m² ({comp['area_km2']:.4f} km²)")

        print("="*60 + "\n")


def analyze_flood_from_prediction(
    prediction_mask_path: str,
    source_tiff_path: str = None,
    gsd: float = 3.8,
    output_dir: str = "outputs/gis",
    min_area_threshold: float = 100.0
) -> Dict:
    """
    Convenience function to analyze flood extent from a prediction mask.

    Args:
        prediction_mask_path: Path to predicted flood mask
        source_tiff_path: Path to source GeoTIFF (optional)
        gsd: Ground Sample Distance if source TIFF not available
        output_dir: Directory to save outputs
        min_area_threshold: Minimum area threshold in m²

    Returns:
        Analysis results dictionary
    """
    # Create analyzer
    analyzer = FloodExtentAnalyzer(
        source_tiff=source_tiff_path,
        gsd=gsd
    )

    # Load mask
    mask = cv2.imread(prediction_mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise ValueError(f"Failed to load mask from {prediction_mask_path}")

    # Analyze
    results = analyzer.calculate_flood_extent(mask, min_area_threshold)

    # Print summary
    analyzer.print_summary(results)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save results to JSON
    output_json = Path(output_dir) / "extent_analysis.json"
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_json}")

    # Create extent map
    extent_map_path = Path(output_dir) / "extent_components.png"
    analyzer.create_extent_map(mask, str(extent_map_path))
    print(f"Extent map saved to {extent_map_path}")

    return results


if __name__ == "__main__":
    """Test flood extent analysis."""
    import argparse

    parser = argparse.ArgumentParser(description='Analyze flood extent')
    parser.add_argument('--mask', type=str, help='Path to flood mask')
    parser.add_argument('--source_tiff', type=str, help='Path to source GeoTIFF')
    parser.add_argument('--gsd', type=float, default=3.8, help='Ground Sample Distance')
    parser.add_argument('--output_dir', type=str, default='outputs/gis', help='Output directory')

    args = parser.parse_args()

    if args.mask:
        # Analyze specific mask
        results = analyze_flood_from_prediction(
            prediction_mask_path=args.mask,
            source_tiff_path=args.source_tiff,
            gsd=args.gsd,
            output_dir=args.output_dir
        )
    else:
        # Test with dummy data
        print("No mask provided. Creating test data...")

        # Create dummy mask
        mask = np.zeros((1000, 1000), dtype=np.uint8)
        mask[200:400, 300:600] = 1
        mask[600:800, 100:300] = 1

        # Create analyzer
        analyzer = FloodExtentAnalyzer(gsd=3.8)

        # Analyze
        results = analyzer.calculate_flood_extent(mask)

        # Print summary
        analyzer.print_summary(results)
