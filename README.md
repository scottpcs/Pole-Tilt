# Cylinder Tilt Detection

A Python tool for accurately estimating the tilt of cylindrical objects (such as lightposts and traffic signs) from point cloud data, even in the presence of significant outliers and environmental noise.

## Author

**Scott Thielman**  
Email: scott.thielman@gmail.com  
Phone: +1 206-229-1796

## Overview

This project provides a robust pipeline for analyzing 3D point cloud data to:
- Detect and segment the ground plane
- Isolate cylindrical objects from background elements
- Accurately estimate the tilt angle and direction of detected cylinders
- Visualize results with reference coordinate systems

The solution is designed to handle real-world challenges including:
- Significant outlier points from foreground objects
- Uneven terrain and ground surfaces
- Occlusions and partial object visibility
- Noise in measurement data

## Installation

### Requirements
- Python 3.7+
- Open3D
- NumPy
- scikit-learn
- matplotlib

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/cylinder-tilt-detection.git
cd cylinder-tilt-detection

# Create and activate virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
# Run with default settings (synthetic test data)
python pole_tilt.py

# Process a specific point cloud file
python pole_tilt.py --input my_pointcloud.ply
```

### Command Line Options

```
--input FILE       Input point cloud file (PLY, PCD, etc.)
--tilt ANGLE       Tilt angle for synthetic data (degrees)
--direction ANGLE  Tilt direction for synthetic data (degrees)
--no-vis           Disable visualization
--voxel-size SIZE  Voxel size for downsampling
```

### ZED Camera Integration

To use with ZED X camera data:

1. Install the ZED SDK following the instructions at: https://www.stereolabs.com/docs/installation/
2. Update the `load_from_zed_svo` function with your SVO file path:

```python
# Example usage with ZED camera data
python pole_tilt.py --input path/to/zed_recording.svo
```

## Technical Approach

The cylinder tilt estimation process follows these steps:

1. **Data Loading & Preprocessing**
   - Load point cloud data (from file or ZED camera)
   - Optional downsampling for large point clouds
   - Statistical outlier removal

2. **Ground Plane Detection**
   - RANSAC-based plane segmentation with horizontal preference
   - Ground points extraction and normal vector calculation

3. **Cylinder Segmentation**
   - DBSCAN clustering to isolate the cylinder
   - Automatic parameter adjustment for clustering

4. **Axis Fitting & Tilt Estimation**
   - PCA for initial axis estimation
   - RANSAC refinement to handle outliers
   - Calculation of tilt angle from vertical reference
   - Determination of azimuth (tilt direction)

5. **Visualization**
   - Color-coded point cloud display
   - Axis visualization (green = cylinder axis, blue = vertical reference)
   - Ground plane visualization
   - Coordinate system reference display

## Results Interpretation

The script provides the following key measurements:

- **Cylinder Axis Vector**: The 3D direction vector of the cylinder's central axis
- **Tilt Angle**: Angle between the cylinder axis and vertical (in degrees)
- **Tilt Direction (Azimuth)**: Compass direction of the tilt (in degrees)
  - 0° = East (positive X)
  - 90° = North (positive Y)
  - 180° or -180° = West (negative X)
  - -90° = South (negative Y)

### Visualization Guide

The 3D visualization includes:
- **Red points**: Inlier points used for fitting the cylinder axis
- **Multicolored points**: Cylinder point cloud with cluster coloring
- **Green line**: Fitted cylinder axis
- **Blue line**: Vertical reference
- **Yellow points**: Detected ground plane
- **Gray grid**: Horizontal reference grid
- **Coordinate axes**: Red/Green/Blue axes showing global X/Y/Z directions

## Parameters & Customization

The following parameters can be adjusted for your specific use case:

- **Ground Detection**
  - `distance_threshold`: Maximum distance for RANSAC plane fitting
  - `enforce_horizontal`: Force detection of horizontal ground planes

- **Cylinder Segmentation**
  - `eps`: DBSCAN clustering distance threshold
  - `min_points`: Minimum points to form a cluster

- **Axis Fitting**
  - `min_samples`: Minimum percentage of points for RANSAC fitting
  - `residual_threshold`: Maximum distance for inlier determination

## License

[MIT License](LICENSE)

## Acknowledgements

This project was developed to address the challenge of accurately measuring the tilt of cylindrical structures in urban environments using 3D computer vision techniques.