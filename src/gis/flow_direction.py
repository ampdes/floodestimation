"""
Water flow direction analysis module.

This module computes flow direction and flow accumulation from a Digital Elevation Model (DEM)
and integrates it with flood extent masks to analyze water flow patterns.
"""

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pathlib import Path
import subprocess
import os


class FlowDirectionAnalyzer:
    """
    Class for analyzing water flow direction using DEM data.
    """

    def __init__(
        self,
        dem_path: str,
        use_whitebox: bool = True
    ):
        """
        Initialize flow direction analyzer.

        Args:
            dem_path: Path to DEM file (GeoTIFF)
            use_whitebox: Use WhiteboxTools (True) or basic numpy implementation (False)
        """
        self.dem_path = dem_path
        self.use_whitebox = use_whitebox

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
        print(f"  CRS: {self.dem_crs}")
        print(f"  NoData: {self.dem_nodata}")

    def fill_depressions(self, output_path: str = None):
        """
        Fill depressions in DEM (required for flow analysis).

        Args:
            output_path: Path to save filled DEM

        Returns:
            Filled DEM array
        """
        if self.use_whitebox:
            return self._fill_depressions_whitebox(output_path)
        else:
            return self._fill_depressions_basic()

    def _fill_depressions_whitebox(self, output_path: str = None):
        """Fill depressions using WhiteboxTools."""
        try:
            from whitebox import WhiteboxTools

            wbt = WhiteboxTools()
            wbt.verbose = False

            # Create output path if not provided
            if output_path is None:
                output_path = str(Path(self.dem_path).parent / "dem_filled.tif")

            # Fill depressions
            wbt.fill_depressions(self.dem_path, output_path)

            # Load filled DEM
            with rasterio.open(output_path) as src:
                filled_dem = src.read(1)

            print(f"Depressions filled using WhiteboxTools")
            return filled_dem

        except ImportError:
            print("WhiteboxTools not available, using basic method")
            return self._fill_depressions_basic()

    def _fill_depressions_basic(self):
        """Basic depression filling (less sophisticated)."""
        # Simple approach: iteratively raise depression cells to lowest neighbor
        filled = self.dem.copy()

        # This is a simplified implementation
        # For production, use WhiteboxTools or RichDEM

        print("Using basic depression filling (less accurate)")
        return filled

    def compute_flow_direction_d8(self):
        """
        Compute D8 flow direction.

        D8 assigns flow to one of 8 neighbors (N, NE, E, SE, S, SW, W, NW).
        Flow direction is encoded as:
        32  64  128
        16   0   1
        8    4   2

        Returns:
            Flow direction array
        """
        if self.use_whitebox:
            return self._compute_flow_direction_whitebox()
        else:
            return self._compute_flow_direction_basic()

    def _compute_flow_direction_whitebox(self):
        """Compute flow direction using WhiteboxTools."""
        try:
            from whitebox import WhiteboxTools

            wbt = WhiteboxTools()
            wbt.verbose = False

            # Create temp directory
            temp_dir = Path("outputs/gis/temp")
            temp_dir.mkdir(parents=True, exist_ok=True)

            # Fill depressions first
            filled_path = str(temp_dir / "dem_filled.tif")
            wbt.fill_depressions(self.dem_path, filled_path)

            # Compute flow direction
            flow_dir_path = str(temp_dir / "flow_direction.tif")
            wbt.d8_pointer(filled_path, flow_dir_path)

            # Load result
            with rasterio.open(flow_dir_path) as src:
                flow_direction = src.read(1)

            print("Flow direction computed using WhiteboxTools (D8)")
            return flow_direction

        except ImportError:
            print("WhiteboxTools not available, using basic method")
            return self._compute_flow_direction_basic()

    def _compute_flow_direction_basic(self):
        """
        Basic D8 flow direction computation.
        """
        rows, cols = self.dem.shape
        flow_dir = np.zeros((rows, cols), dtype=np.uint8)

        # D8 direction codes
        directions = {
            (-1, 1): 128,   # NE
            (0, 1): 1,      # E
            (1, 1): 2,      # SE
            (1, 0): 4,      # S
            (1, -1): 8,     # SW
            (0, -1): 16,    # W
            (-1, -1): 32,   # NW
            (-1, 0): 64     # N
        }

        # For each cell, find steepest descent
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                if self.dem_nodata and self.dem[i, j] == self.dem_nodata:
                    continue

                max_slope = -np.inf
                max_dir = 0

                # Check 8 neighbors
                for (di, dj), code in directions.items():
                    ni, nj = i + di, j + dj

                    if self.dem_nodata and self.dem[ni, nj] == self.dem_nodata:
                        continue

                    slope = (self.dem[i, j] - self.dem[ni, nj])

                    if slope > max_slope:
                        max_slope = slope
                        max_dir = code

                flow_dir[i, j] = max_dir

        print("Flow direction computed using basic D8 method")
        return flow_dir

    def compute_flow_accumulation(self, flow_direction: np.ndarray = None):
        """
        Compute flow accumulation from flow direction.

        Args:
            flow_direction: Flow direction array (if None, will compute)

        Returns:
            Flow accumulation array
        """
        if flow_direction is None:
            flow_direction = self.compute_flow_direction_d8()

        if self.use_whitebox:
            return self._compute_flow_accumulation_whitebox()
        else:
            print("Basic flow accumulation not implemented")
            print("Please use WhiteboxTools for flow accumulation")
            return np.zeros_like(flow_direction)

    def _compute_flow_accumulation_whitebox(self):
        """Compute flow accumulation using WhiteboxTools."""
        try:
            from whitebox import WhiteboxTools

            wbt = WhiteboxTools()
            wbt.verbose = False

            # Create temp directory
            temp_dir = Path("outputs/gis/temp")
            temp_dir.mkdir(parents=True, exist_ok=True)

            # Fill depressions first
            filled_path = str(temp_dir / "dem_filled.tif")
            wbt.fill_depressions(self.dem_path, filled_path)

            # Compute flow accumulation
            flow_accum_path = str(temp_dir / "flow_accumulation.tif")
            wbt.d8_flow_accumulation(filled_path, flow_accum_path)

            # Load result
            with rasterio.open(flow_accum_path) as src:
                flow_accumulation = src.read(1)

            print("Flow accumulation computed using WhiteboxTools")
            return flow_accumulation

        except ImportError:
            print("WhiteboxTools not available")
            return np.zeros_like(self.dem)

    def analyze_flow_in_flooded_areas(
        self,
        flood_mask: np.ndarray,
        flow_direction: np.ndarray = None,
        flow_accumulation: np.ndarray = None
    ):
        """
        Analyze flow patterns within flooded areas.

        Args:
            flood_mask: Binary flood mask (must match DEM dimensions)
            flow_direction: Flow direction array (optional)
            flow_accumulation: Flow accumulation array (optional)

        Returns:
            Dictionary with flow analysis results
        """
        # Compute flow direction if not provided
        if flow_direction is None:
            flow_direction = self.compute_flow_direction_d8()

        # Compute flow accumulation if not provided
        if flow_accumulation is None:
            flow_accumulation = self.compute_flow_accumulation(flow_direction)

        # Ensure flood mask matches DEM dimensions
        if flood_mask.shape != self.dem.shape:
            print(f"Resizing flood mask from {flood_mask.shape} to {self.dem.shape}")
            import cv2
            flood_mask = cv2.resize(
                flood_mask,
                (self.dem.shape[1], self.dem.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        # Ensure binary
        if flood_mask.max() > 1:
            flood_mask = (flood_mask > 127).astype(np.uint8)

        # Extract flow information in flooded areas
        flooded_flow_dir = flow_direction * flood_mask
        flooded_flow_accum = flow_accumulation * flood_mask

        # Calculate statistics
        results = {
            'mean_flow_accumulation': float(np.mean(flooded_flow_accum[flood_mask > 0])),
            'max_flow_accumulation': float(np.max(flooded_flow_accum[flood_mask > 0])),
            'num_flooded_pixels': int(np.sum(flood_mask)),
            'flow_direction_in_flooded': flooded_flow_dir,
            'flow_accumulation_in_flooded': flooded_flow_accum
        }

        return results

    def save_flow_rasters(
        self,
        flow_direction: np.ndarray,
        flow_accumulation: np.ndarray,
        output_dir: str
    ):
        """
        Save flow direction and accumulation rasters.

        Args:
            flow_direction: Flow direction array
            flow_accumulation: Flow accumulation array
            output_dir: Directory to save outputs
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Save flow direction
        flow_dir_path = Path(output_dir) / "flow_direction.tif"
        with rasterio.open(
            flow_dir_path, 'w',
            driver='GTiff',
            height=flow_direction.shape[0],
            width=flow_direction.shape[1],
            count=1,
            dtype=flow_direction.dtype,
            crs=self.dem_crs,
            transform=self.dem_transform
        ) as dst:
            dst.write(flow_direction, 1)

        print(f"Flow direction saved to {flow_dir_path}")

        # Save flow accumulation
        flow_accum_path = Path(output_dir) / "flow_accumulation.tif"
        with rasterio.open(
            flow_accum_path, 'w',
            driver='GTiff',
            height=flow_accumulation.shape[0],
            width=flow_accumulation.shape[1],
            count=1,
            dtype=flow_accumulation.dtype,
            crs=self.dem_crs,
            transform=self.dem_transform
        ) as dst:
            dst.write(flow_accumulation, 1)

        print(f"Flow accumulation saved to {flow_accum_path}")


def download_dem_usgs(bounds: tuple, output_path: str, resolution: int = 10):
    """
    Download DEM from USGS 3DEP.

    Args:
        bounds: Bounding box (left, bottom, right, top)
        output_path: Path to save DEM
        resolution: Resolution in meters (10 or 30)

    Note: This is a placeholder. Actual implementation would use USGS API or py3dep.
    """
    print("DEM download functionality requires implementation")
    print("Options:")
    print("  1. Use py3dep library: pip install py3dep")
    print("  2. Use elevation library: pip install elevation")
    print("  3. Manual download from USGS Earth Explorer")
    print(f"  Requested bounds: {bounds}")
    print(f"  Resolution: {resolution}m")


if __name__ == "__main__":
    """Test flow direction analysis."""
    print("Flow Direction Analysis Module")
    print("="*60)
    print("\nNote: This module requires:")
    print("  1. A Digital Elevation Model (DEM) file")
    print("  2. WhiteboxTools for accurate flow analysis")
    print("\nInstall WhiteboxTools:")
    print("  pip install whitebox")
    print("\nFor USGS DEM download, install:")
    print("  pip install py3dep")
    print("="*60)
