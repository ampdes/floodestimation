"""
Flood depth estimation module.

This module estimates flood depth using DEM data and flood extent masks.
The approach assumes flat water surfaces and estimates depth based on elevation differences.
"""

import numpy as np
import rasterio
from scipy import ndimage
from pathlib import Path
import cv2
from typing import Tuple, Dict


class FloodDepthEstimator:
    """
    Class for estimating flood depth from DEM and flood masks.
    """

    def __init__(
        self,
        dem_path: str,
        min_depth_threshold: float = 0.1,
        max_depth_threshold: float = 20.0
    ):
        """
        Initialize flood depth estimator.

        Args:
            dem_path: Path to DEM file (GeoTIFF)
            min_depth_threshold: Minimum depth threshold in meters
            max_depth_threshold: Maximum reasonable depth in meters
        """
        self.dem_path = dem_path
        self.min_depth_threshold = min_depth_threshold
        self.max_depth_threshold = max_depth_threshold

        # Load DEM
        self.load_dem()

    def load_dem(self):
        """Load DEM from file."""
        if not Path(self.dem_path).exists():
            raise FileNotFoundError(f"DEM file not found: {self.dem_path}")

        with rasterio.open(self.dem_path) as src:
            self.dem = src.read(1)
            self.dem_transform = src.transform
            self.dem_crs = src.crs
            self.dem_nodata = src.nodata

        print(f"DEM loaded: {self.dem.shape}")
        print(f"  Elevation range: {self.dem.min():.2f} to {self.dem.max():.2f} m")

    def estimate_depth_boundary_method(
        self,
        flood_mask: np.ndarray,
        buffer_distance: int = 5
    ) -> np.ndarray:
        """
        Estimate flood depth using boundary elevation method.

        This method assumes water surface elevation equals the average elevation
        at the flood boundary, then calculates depth as boundary_elevation - ground_elevation.

        Args:
            flood_mask: Binary flood mask (must match DEM dimensions)
            buffer_distance: Distance (in pixels) from boundary to sample elevations

        Returns:
            Depth map in meters
        """
        # Ensure flood mask matches DEM dimensions
        if flood_mask.shape != self.dem.shape:
            print(f"Resizing flood mask from {flood_mask.shape} to {self.dem.shape}")
            flood_mask = cv2.resize(
                flood_mask,
                (self.dem.shape[1], self.dem.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        # Ensure binary
        if flood_mask.max() > 1:
            flood_mask = (flood_mask > 127).astype(np.uint8)

        # Find boundary of flood
        # Erode mask to find inner boundary
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(flood_mask, kernel, iterations=1)
        boundary = flood_mask - eroded

        # Dilate boundary to get buffer zone
        boundary_buffer = cv2.dilate(boundary, kernel, iterations=buffer_distance)

        # Extract elevations at boundary
        boundary_elevations = self.dem[boundary_buffer > 0]

        # Remove NoData values
        if self.dem_nodata is not None:
            boundary_elevations = boundary_elevations[boundary_elevations != self.dem_nodata]

        if len(boundary_elevations) == 0:
            print("Warning: No valid boundary elevations found")
            return np.zeros_like(flood_mask, dtype=np.float32)

        # Calculate water surface elevation (use percentile to avoid outliers)
        water_surface_elevation = np.percentile(boundary_elevations, 75)

        print(f"Estimated water surface elevation: {water_surface_elevation:.2f} m")

        # Calculate depth = water_surface - ground_elevation
        depth_map = np.zeros_like(flood_mask, dtype=np.float32)
        depth_map[flood_mask > 0] = water_surface_elevation - self.dem[flood_mask > 0]

        # Apply thresholds
        depth_map[depth_map < self.min_depth_threshold] = 0
        depth_map[depth_map > self.max_depth_threshold] = self.max_depth_threshold

        # Handle NoData
        if self.dem_nodata is not None:
            depth_map[self.dem == self.dem_nodata] = 0

        return depth_map

    def estimate_depth_component_method(
        self,
        flood_mask: np.ndarray
    ) -> np.ndarray:
        """
        Estimate flood depth using connected component method.

        Each connected flood component is assumed to have a flat water surface
        at the maximum elevation within that component.

        Args:
            flood_mask: Binary flood mask (must match DEM dimensions)

        Returns:
            Depth map in meters
        """
        # Ensure flood mask matches DEM dimensions
        if flood_mask.shape != self.dem.shape:
            print(f"Resizing flood mask from {flood_mask.shape} to {self.dem.shape}")
            flood_mask = cv2.resize(
                flood_mask,
                (self.dem.shape[1], self.dem.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        # Ensure binary
        if flood_mask.max() > 1:
            flood_mask = (flood_mask > 127).astype(np.uint8)

        # Find connected components
        num_labels, labels = cv2.connectedComponents(flood_mask, connectivity=8)

        # Initialize depth map
        depth_map = np.zeros_like(flood_mask, dtype=np.float32)

        # Process each component
        for label_id in range(1, num_labels):
            component_mask = (labels == label_id)

            # Get elevations in this component
            component_elevations = self.dem[component_mask]

            # Remove NoData
            if self.dem_nodata is not None:
                component_elevations = component_elevations[component_elevations != self.dem_nodata]

            if len(component_elevations) == 0:
                continue

            # Water surface = max elevation in component (conservative estimate)
            # Or use percentile for more realistic estimate
            water_surface = np.percentile(component_elevations, 90)

            # Calculate depth for this component
            component_depth = water_surface - self.dem[component_mask]

            # Apply to depth map
            depth_map[component_mask] = component_depth

        # Apply thresholds
        depth_map[depth_map < self.min_depth_threshold] = 0
        depth_map[depth_map > self.max_depth_threshold] = self.max_depth_threshold

        # Handle NoData
        if self.dem_nodata is not None:
            depth_map[self.dem == self.dem_nodata] = 0

        return depth_map

    def calculate_depth_statistics(
        self,
        depth_map: np.ndarray,
        flood_mask: np.ndarray
    ) -> Dict:
        """
        Calculate statistics from depth map.

        Args:
            depth_map: Depth map in meters
            flood_mask: Binary flood mask

        Returns:
            Dictionary with depth statistics
        """
        # Get depth values in flooded areas
        flooded_depths = depth_map[flood_mask > 0]

        # Remove zeros
        flooded_depths = flooded_depths[flooded_depths > 0]

        if len(flooded_depths) == 0:
            return {
                'mean_depth_m': 0.0,
                'median_depth_m': 0.0,
                'max_depth_m': 0.0,
                'min_depth_m': 0.0,
                'std_depth_m': 0.0,
                'num_pixels_with_depth': 0
            }

        stats = {
            'mean_depth_m': float(np.mean(flooded_depths)),
            'median_depth_m': float(np.median(flooded_depths)),
            'max_depth_m': float(np.max(flooded_depths)),
            'min_depth_m': float(np.min(flooded_depths)),
            'std_depth_m': float(np.std(flooded_depths)),
            'percentile_25': float(np.percentile(flooded_depths, 25)),
            'percentile_75': float(np.percentile(flooded_depths, 75)),
            'percentile_95': float(np.percentile(flooded_depths, 95)),
            'num_pixels_with_depth': len(flooded_depths)
        }

        return stats

    def save_depth_map(
        self,
        depth_map: np.ndarray,
        output_path: str
    ):
        """
        Save depth map as GeoTIFF.

        Args:
            depth_map: Depth map in meters
            output_path: Path to save GeoTIFF
        """
        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            height=depth_map.shape[0],
            width=depth_map.shape[1],
            count=1,
            dtype=rasterio.float32,
            crs=self.dem_crs,
            transform=self.dem_transform,
            nodata=0.0
        ) as dst:
            dst.write(depth_map.astype(np.float32), 1)

        print(f"Depth map saved to {output_path}")

    def visualize_depth_map(
        self,
        depth_map: np.ndarray,
        output_path: str = None
    ):
        """
        Create visualization of depth map.

        Args:
            depth_map: Depth map in meters
            output_path: Path to save visualization
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Depth map
        im1 = ax1.imshow(depth_map, cmap='Blues', vmin=0, vmax=depth_map.max())
        ax1.set_title('Flood Depth Map')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, label='Depth (meters)')

        # Histogram
        depths_nonzero = depth_map[depth_map > 0]
        if len(depths_nonzero) > 0:
            ax2.hist(depths_nonzero, bins=50, edgecolor='black')
            ax2.set_xlabel('Depth (meters)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Depth Distribution')
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def print_depth_summary(self, stats: Dict):
        """
        Print formatted summary of depth statistics.

        Args:
            stats: Statistics dictionary
        """
        print("\n" + "="*60)
        print("FLOOD DEPTH ESTIMATION SUMMARY")
        print("="*60)
        print(f"Mean Depth:        {stats['mean_depth_m']:.2f} m")
        print(f"Median Depth:      {stats['median_depth_m']:.2f} m")
        print(f"Max Depth:         {stats['max_depth_m']:.2f} m")
        print(f"Min Depth:         {stats['min_depth_m']:.2f} m")
        print(f"Std Deviation:     {stats['std_depth_m']:.2f} m")
        print(f"\nPercentiles:")
        print(f"  25th:            {stats['percentile_25']:.2f} m")
        print(f"  75th:            {stats['percentile_75']:.2f} m")
        print(f"  95th:            {stats['percentile_95']:.2f} m")
        print(f"\nPixels with depth: {stats['num_pixels_with_depth']:,}")
        print("="*60 + "\n")


def estimate_flood_depth(
    flood_mask_path: str,
    dem_path: str,
    output_dir: str = "outputs/gis",
    method: str = "boundary"
) -> Tuple[np.ndarray, Dict]:
    """
    Convenience function to estimate flood depth.

    Args:
        flood_mask_path: Path to flood mask
        dem_path: Path to DEM file
        output_dir: Directory to save outputs
        method: Estimation method ('boundary' or 'component')

    Returns:
        Tuple of (depth_map, statistics)
    """
    # Create estimator
    estimator = FloodDepthEstimator(dem_path)

    # Load flood mask
    flood_mask = cv2.imread(flood_mask_path, cv2.IMREAD_GRAYSCALE)

    if flood_mask is None:
        raise ValueError(f"Failed to load flood mask from {flood_mask_path}")

    # Estimate depth
    if method == "boundary":
        depth_map = estimator.estimate_depth_boundary_method(flood_mask)
    elif method == "component":
        depth_map = estimator.estimate_depth_component_method(flood_mask)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Calculate statistics
    stats = estimator.calculate_depth_statistics(depth_map, flood_mask)

    # Print summary
    estimator.print_depth_summary(stats)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save depth map
    depth_map_path = Path(output_dir) / "flood_depth.tif"
    estimator.save_depth_map(depth_map, str(depth_map_path))

    # Save visualization
    viz_path = Path(output_dir) / "flood_depth_visualization.png"
    estimator.visualize_depth_map(depth_map, str(viz_path))

    # Save statistics
    import json
    stats_path = Path(output_dir) / "depth_statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"Statistics saved to {stats_path}")

    return depth_map, stats


if __name__ == "__main__":
    """Test flood depth estimation."""
    import argparse

    parser = argparse.ArgumentParser(description='Estimate flood depth')
    parser.add_argument('--mask', type=str, help='Path to flood mask')
    parser.add_argument('--dem', type=str, help='Path to DEM file')
    parser.add_argument('--output_dir', type=str, default='outputs/gis', help='Output directory')
    parser.add_argument('--method', type=str, default='boundary', choices=['boundary', 'component'],
                        help='Estimation method')

    args = parser.parse_args()

    if args.mask and args.dem:
        # Estimate depth
        depth_map, stats = estimate_flood_depth(
            flood_mask_path=args.mask,
            dem_path=args.dem,
            output_dir=args.output_dir,
            method=args.method
        )
    else:
        print("Usage: python depth_estimation.py --mask MASK_PATH --dem DEM_PATH")
        print("\nThis module requires:")
        print("  1. A flood mask (binary image)")
        print("  2. A Digital Elevation Model (DEM) covering the same area")
        print("\nThe DEM must have the same spatial extent as the flood mask")
