import numpy as np
import open3d as o3d
from sklearn.linear_model import RANSACRegressor
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import argparse


def create_synthetic_data(tilt_angle=15.0, tilt_direction=45.0, 
                          cylinder_points=1400, ground_points=500, 
                          outlier_points=200, noise_level=0.05,
                          ground_noise=0.1):
    """
    Create synthetic point cloud data of a tilted cylinder with ground plane and outliers
    
    Args:
        tilt_angle: Tilt angle in degrees from vertical
        tilt_direction: Direction of tilt in degrees (azimuth)
        cylinder_points: Number of points for the cylinder
        ground_points: Number of points for the ground plane
        outlier_points: Number of outlier points (foreground objects)
        noise_level: General noise level for cylinder radius
        ground_noise: Noise level for ground plane (simulates sidewalks, etc.)
    """
    print(f"Generating cylinder with {tilt_angle}° tilt in direction {tilt_direction}°...")
    
    # Convert angles to radians
    tilt_rad = np.radians(tilt_angle)
    direction_rad = np.radians(tilt_direction)
    
    # Create rotation matrix for the tilt
    # First rotate around Y axis by direction angle, then around the rotated X axis by tilt angle
    cy = np.cos(direction_rad)
    sy = np.sin(direction_rad)
    ct = np.cos(tilt_rad)
    st = np.sin(tilt_rad)
    
    # Direction rotation (around Z axis)
    Rz = np.array([
        [cy, -sy, 0],
        [sy, cy, 0],
        [0, 0, 1]
    ])
    
    # Tilt rotation (around rotated X axis)
    Rx = np.array([
        [1, 0, 0],
        [0, ct, -st],
        [0, st, ct]
    ])
    
    # Combined rotation
    R = Rz @ Rx
    
    # Create points for a vertical cylinder
    points = []
    
    # Cylinder points
    for i in range(cylinder_points):
        height = np.random.uniform(0, 10)
        angle = np.random.uniform(0, 2 * np.pi)
        radius = 0.5 + np.random.normal(0, noise_level)  # Add some noise to radius
        
        # Points on a vertical cylinder
        x_vert = radius * np.cos(angle)
        y_vert = radius * np.sin(angle)
        z_vert = height
        
        # Apply rotation to tilt the cylinder
        point_vert = np.array([x_vert, y_vert, z_vert])
        point_tilted = R @ point_vert
        
        points.append(point_tilted)
    
    # Add ground plane with realistic noise
    print(f"Adding ground plane with {ground_points} points...")
    
    # Create a more realistic ground with varying elevation (simulates sidewalks, etc.)
    for i in range(ground_points):
        x = np.random.uniform(-5, 5)
        y = np.random.uniform(-5, 5)
        
        # Distance from center affects noise to simulate features like sidewalks
        dist_from_center = np.sqrt(x**2 + y**2)
        
        # Add structured noise - lines and patterns in the ground
        structured_noise = 0
        
        # Simple sidewalk simulation - slightly raised area in a strip
        if 2.0 < x < 3.0:
            structured_noise += 0.1
            
        # Add some random bumps to simulate uneven terrain
        if np.random.random() < 0.1:  # 10% chance of a bump
            structured_noise += np.random.uniform(0, 0.2)
            
        # Distance-based noise (gradual slope away from center)
        slope_noise = dist_from_center * 0.01
            
        # Random noise component
        random_noise = np.random.normal(0, ground_noise)
        
        # Combine noise components
        z = structured_noise + slope_noise + random_noise
        
        points.append([x, y, z])
    
    # Add structured outliers (like a nearby object)
    print(f"Adding structured outliers ({outlier_points} points)...")
    for i in range(outlier_points):
        # Create a small box-like structure near the cylinder
        x = np.random.uniform(2, 3)
        y = np.random.uniform(2, 3)
        z = np.random.uniform(1, 3)
        points.append([x, y, z])
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    
    print(f"Generated point cloud with {len(points)} points")
    
    return pcd


def remove_statistical_outlier(pcd, nb_neighbors=20, std_ratio=2.0):
    """
    Remove outliers using statistical outlier removal
    """
    print("Removing statistical outliers...")
    cleaned_pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return cleaned_pcd


def segment_ground_plane(pcd, distance_threshold=0.1, ransac_n=3, num_iterations=1000, enforce_horizontal=True):
    """
    Segment and remove ground plane
    
    Args:
        pcd: Input point cloud
        distance_threshold: Maximum distance a point can be from the plane model
        ransac_n: Number of points to sample for RANSAC
        num_iterations: Number of RANSAC iterations
        enforce_horizontal: If True, enforce that ground is roughly horizontal
        
    Returns:
        Tuple of (non_ground_pcd, ground_normal, ground_points, plane_model)
    """
    print("Segmenting ground plane...")
    
    if enforce_horizontal:
        # First attempt to find points with low z-values (potential ground)
        points = np.asarray(pcd.points)
        z_values = points[:, 2]
        z_min, z_max = np.min(z_values), np.max(z_values)
        z_threshold = z_min + 0.2  # Consider lowest 20cm as potential ground
        
        # Select points near ground level
        potential_ground_indices = np.where(z_values < z_threshold)[0]
        
        if len(potential_ground_indices) > 10:  # Need enough points for plane fitting
            potential_ground_pcd = pcd.select_by_index(potential_ground_indices)
            
            # Try to fit plane to these points
            try:
                plane_model, inliers = potential_ground_pcd.segment_plane(
                    distance_threshold=distance_threshold,
                    ransac_n=ransac_n,
                    num_iterations=num_iterations
                )
                
                # Check if the plane is roughly horizontal (normal has strong z component)
                a, b, c, d = plane_model
                normal = np.array([a, b, c])
                normal = normal / np.linalg.norm(normal)
                
                # If normal is pointing down, flip it
                if normal[2] < 0:
                    normal = -normal
                    plane_model = (-a, -b, -c, -d)
                
                if normal[2] > 0.8:  # Normal is mostly vertical (cos(36°) ≈ 0.8)
                    print("Found horizontal ground plane using height-based selection")
                    
                    # Map inliers back to original point cloud indices
                    original_inliers = potential_ground_indices[inliers]
                    
                    # Get ground and non-ground points
                    non_ground_indices = list(set(range(len(pcd.points))) - set(original_inliers))
                    non_ground_pcd = pcd.select_by_index(non_ground_indices)
                    ground_pcd = pcd.select_by_index(original_inliers)
                    ground_points = np.asarray(ground_pcd.points)
                    
                    print(f"Ground plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
                    print(f"Ground normal vector: [{a:.2f}, {b:.2f}, {c:.2f}]")
                    
                    return non_ground_pcd, normal, ground_points, plane_model
            except Exception as e:
                print(f"Error in horizontal plane fitting: {e}")
    
    # Fall back to standard plane segmentation if horizontal enforcement fails
    # or if enforce_horizontal is False
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )
    
    # Get points that are not part of the ground plane
    non_ground_indices = list(set(range(len(pcd.points))) - set(inliers))
    non_ground_pcd = pcd.select_by_index(non_ground_indices)
    
    # Get ground points
    ground_pcd = pcd.select_by_index(inliers)
    ground_points = np.asarray(ground_pcd.points)
    
    # Extract ground normal vector
    a, b, c, d = plane_model
    ground_normal = np.array([a, b, c])
    ground_normal = ground_normal / np.linalg.norm(ground_normal)
    
    # If normal is pointing down, flip it for consistency
    if ground_normal[2] < 0:
        ground_normal = -ground_normal
        plane_model = (-a, -b, -c, -d)
        a, b, c, d = plane_model
    
    print(f"Ground plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    print(f"Ground normal vector: [{a:.2f}, {b:.2f}, {c:.2f}]")
    
    return non_ground_pcd, ground_normal, ground_points, plane_model


def cluster_points(pcd, eps=0.5, min_points=100):
    """
    Cluster points using DBSCAN to separate different objects
    """
    print("Clustering points...")
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    
    if len(set(labels)) <= 1:
        print(f"Warning: Only found {len(set(labels))} clusters (including noise)")
        if -1 in labels and len(set(labels)) == 1:
            print("Only noise points found. Adjusting clustering parameters...")
            return cluster_points(pcd, eps=eps*1.5, min_points=max(10, min_points//2))
    
    # Get the largest cluster (assuming it's the cylinder)
    if -1 in labels:  # -1 is noise
        label_counts = np.bincount(labels[labels >= 0]) if any(labels >= 0) else np.array([])
    else:
        label_counts = np.bincount(labels)
        
    if len(label_counts) == 0:
        print("No clusters found. Returning original point cloud.")
        return pcd
        
    largest_cluster_label = np.argmax(label_counts)
    cylinder_indices = np.where(labels == largest_cluster_label)[0]
    cylinder_pcd = pcd.select_by_index(cylinder_indices)
    
    print(f"Largest cluster has {len(cylinder_indices)} points")
    
    # Visualize the clusters with different colors (for debugging)
    max_label = labels.max()
    print(f"Found {max_label + 1 if max_label >= 0 else 0} clusters")
    
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0  # Set noise points to black
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    
    return cylinder_pcd


def fit_cylinder_axis(pcd):
    """
    Fit the cylinder axis using PCA and RANSAC for refinement
    """
    print("Fitting cylinder axis...")
    
    # Convert to numpy array
    points = np.asarray(pcd.points)
    
    if len(points) < 10:
        print("Too few points to fit axis reliably")
        return np.array([0, 0, 1]), np.mean(points, axis=0), points
    
    # Step 1: Use PCA to get initial axis estimate
    pca = PCA(n_components=3)
    pca.fit(points)
    initial_axis = pca.components_[0]  # First principal component
    
    # Ensure the axis points up (positive Z)
    if initial_axis[2] < 0:
        initial_axis = -initial_axis
    
    # Step 2: Project points onto a plane perpendicular to the axis
    center = np.mean(points, axis=0)
    
    # Create basis for the plane perpendicular to the axis
    # First find any vector perpendicular to the axis
    if abs(initial_axis[0]) > abs(initial_axis[1]):
        perp1 = np.array([initial_axis[2], 0, -initial_axis[0]])
    else:
        perp1 = np.array([0, initial_axis[2], -initial_axis[1]])
    perp1 = perp1 / np.linalg.norm(perp1)
    
    # Second perpendicular vector using cross product
    perp2 = np.cross(initial_axis, perp1)
    perp2 = perp2 / np.linalg.norm(perp2)
    
    # Project points onto the plane defined by perp1 and perp2
    centered_points = points - center
    projection_perp1 = np.dot(centered_points, perp1)
    projection_perp2 = np.dot(centered_points, perp2)
    
    # Step 3: Use RANSAC to fit a circle (or actually, to identify points that form a circle)
    # In a perfect cylinder projection, points should be at roughly constant distance from center
    point_projections = np.column_stack((projection_perp1, projection_perp2))
    distances = np.sqrt(np.sum(point_projections**2, axis=1))
    
    # We'll use RANSAC to find the cylinder radius, ignoring outliers
    try:
        # Create dummy X and y for RANSAC
        X = np.ones((len(distances), 1))
        y = distances
        
        ransac = RANSACRegressor(min_samples=0.3, residual_threshold=0.1)
        ransac.fit(X, y)
        
        # Get inlier mask and estimated radius
        inlier_mask = ransac.inlier_mask_
        estimated_radius = ransac.predict([[1]])[0]
        
        print(f"Estimated cylinder radius: {estimated_radius:.3f}")
        
        # Use inliers to refine the axis
        inlier_points = points[inlier_mask]
        
        if len(inlier_points) < 10:
            print("Too few inliers, using initial PCA axis")
            refined_axis = initial_axis
        else:
            # Refine axis with PCA on inliers
            pca = PCA(n_components=3)
            pca.fit(inlier_points)
            refined_axis = pca.components_[0]
            
            # Ensure the axis points up (positive Z)
            if refined_axis[2] < 0:
                refined_axis = -refined_axis
    except Exception as e:
        print(f"RANSAC fitting failed: {e}")
        print("Using initial PCA axis")
        refined_axis = initial_axis
        inlier_points = points
    
    return refined_axis, center, inlier_points


def calculate_tilt_angle(axis_vector, reference_vector=np.array([0, 0, 1])):
    """
    Calculate tilt angle between cylinder axis and reference vector (usually vertical)
    """
    # Normalize vectors
    axis_vector = axis_vector / np.linalg.norm(axis_vector)
    reference_vector = reference_vector / np.linalg.norm(reference_vector)
    
    # Calculate angle in radians using dot product
    cos_angle = np.dot(axis_vector, reference_vector)
    angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    
    # Calculate tilt direction (azimuth)
    # Project axis vector onto horizontal plane
    proj = axis_vector - np.dot(axis_vector, reference_vector) * reference_vector
    if np.linalg.norm(proj) > 1e-6:  # Check if projection is non-zero
        proj = proj / np.linalg.norm(proj)
        azimuth_rad = np.arctan2(proj[1], proj[0])
        azimuth_deg = np.degrees(azimuth_rad)
    else:
        azimuth_deg = 0.0
    
    return angle_deg, azimuth_deg


def visualize_results(pcd, axis_vector, center_point, inlier_points=None, distance=10.0, 
                   ground_plane=None):
    """
    Visualize the point cloud and the fitted cylinder axis
    
    Args:
        pcd: Point cloud of the cylinder
        axis_vector: Fitted cylinder axis vector
        center_point: Center point of the cylinder
        inlier_points: Points used for axis fitting
        distance: Length of visualization lines
        ground_plane: Tuple of (ground_points, plane_model) if available
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Add the original point cloud
    vis.add_geometry(pcd)
    
    # Add inlier points in a different color if available
    if inlier_points is not None:
        inlier_pcd = o3d.geometry.PointCloud()
        inlier_pcd.points = o3d.utility.Vector3dVector(inlier_points)
        inlier_pcd.paint_uniform_color([1, 0, 0])  # Red color
        vis.add_geometry(inlier_pcd)
    
    # Create a line segment representing the cylinder axis
    line = o3d.geometry.LineSet()
    line_points = np.array([
        center_point - axis_vector * distance / 2,
        center_point + axis_vector * distance / 2
    ])
    line.points = o3d.utility.Vector3dVector(line_points)
    line.lines = o3d.utility.Vector2iVector(np.array([[0, 1]]))
    line.colors = o3d.utility.Vector3dVector(np.array([[0, 1, 0]]))  # Green line
    vis.add_geometry(line)
    
    # Create a vertical reference line
    ref_line = o3d.geometry.LineSet()
    ref_line_points = np.array([
        center_point - np.array([0, 0, 1]) * distance / 2,
        center_point + np.array([0, 0, 1]) * distance / 2
    ])
    ref_line.points = o3d.utility.Vector3dVector(ref_line_points)
    ref_line.lines = o3d.utility.Vector2iVector(np.array([[0, 1]]))
    ref_line.colors = o3d.utility.Vector3dVector(np.array([[0, 0, 1]]))  # Blue line
    vis.add_geometry(ref_line)
    
    # Visualize ground plane if available
    if ground_plane is not None:
        ground_points, plane_model = ground_plane
        
        if ground_points is not None and len(ground_points) > 0:
            # Add original ground points
            ground_pcd = o3d.geometry.PointCloud()
            ground_pcd.points = o3d.utility.Vector3dVector(ground_points)
            ground_pcd.paint_uniform_color([0.8, 0.8, 0.0])  # Yellow
            vis.add_geometry(ground_pcd)
        
        if plane_model is not None:
            # Create a grid mesh to represent the ground plane
            a, b, c, d = plane_model
            
            # Create a horizontal grid with world coordinates for reference
            grid_size = distance * 2
            grid_points = []
            grid_density = 20  # Number of points in each direction
            
            for i in range(-grid_density//2, grid_density//2):
                for j in range(-grid_density//2, grid_density//2):
                    x_coord = i * grid_size / grid_density
                    y_coord = j * grid_size / grid_density
                    # Calculate z based on plane equation: ax + by + cz + d = 0
                    # Therefore z = -(ax + by + d) / c
                    
                    # Avoid division by zero
                    if abs(c) > 1e-6:
                        z_coord = -(a * x_coord + b * y_coord + d) / c
                    else:
                        # If c is close to zero, place the grid at z=0
                        z_coord = 0
                        
                    grid_points.append([x_coord, y_coord, z_coord])
            
            # Create a point cloud for the grid
            grid_pcd = o3d.geometry.PointCloud()
            grid_pcd.points = o3d.utility.Vector3dVector(np.array(grid_points))
            grid_pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Gray
            vis.add_geometry(grid_pcd)
            
            # Add XYZ coordinate axes at origin for reference
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=distance/2, origin=[0, 0, 0])
            vis.add_geometry(coord_frame)

    # Set camera viewpoint for better visualization
    opt = vis.get_render_option()
    opt.background_color = np.array([1, 1, 1])  # White background
    opt.point_size = 3.0  # Larger points
    
    # Run visualization
    vis.run()
    vis.destroy_window()


def process_point_cloud(pcd, visualize=True, voxel_size=None):
    """
    Process point cloud to estimate cylinder tilt
    """
    # Step 1: Optional downsampling for large point clouds
    if voxel_size is not None and voxel_size > 0:
        print(f"Downsampling with voxel size {voxel_size}...")
        pcd = pcd.voxel_down_sample(voxel_size)
    
    # Step 2: Remove statistical outliers
    pcd = remove_statistical_outlier(pcd)
    
    # Step 3: Segment ground plane
    non_ground_pcd, ground_normal, ground_points, plane_model = segment_ground_plane(pcd)
    
    # Step 4: Cluster points to isolate the cylinder
    cylinder_pcd = cluster_points(non_ground_pcd)
    
    # Step 5: Fit cylinder axis
    axis_vector, center_point, inlier_points = fit_cylinder_axis(cylinder_pcd)
    
    # Step 6: Calculate tilt angle
    tilt_angle, tilt_direction = calculate_tilt_angle(axis_vector)
    
    # Print results
    print("\nResults:")
    print(f"Cylinder axis vector: [{axis_vector[0]:.4f}, {axis_vector[1]:.4f}, {axis_vector[2]:.4f}]")
    print(f"Tilt angle: {tilt_angle:.2f} degrees")
    print(f"Tilt direction (azimuth): {tilt_direction:.2f} degrees")
    
    # Visualize results
    if visualize:
        ground_plane = (ground_points, plane_model)
        visualize_results(cylinder_pcd, axis_vector, center_point, inlier_points, 
                         ground_plane=ground_plane)
    
    return tilt_angle, tilt_direction, axis_vector


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Cylinder tilt estimation from point cloud data.')
    parser.add_argument('--input', type=str, help='Input point cloud file (PLY, PCD, etc.)')
    parser.add_argument('--tilt', type=float, default=20.0, help='Tilt angle for synthetic data (degrees)')
    parser.add_argument('--direction', type=float, default=45.0, help='Tilt direction for synthetic data (degrees)')
    parser.add_argument('--no-vis', action='store_true', help='Disable visualization')
    parser.add_argument('--voxel-size', type=float, default=None, help='Voxel size for downsampling')
    
    args = parser.parse_args()
    
    # Create or load point cloud
    if args.input:
        print(f"Loading point cloud from {args.input}...")
        pcd = o3d.io.read_point_cloud(args.input)
    else:
        print(f"Running single synthetic data test (default):")
        pcd = create_synthetic_data(tilt_angle=args.tilt, tilt_direction=args.direction)
    
    # Process the point cloud
    tilt_angle, tilt_direction, axis_vector = process_point_cloud(
        pcd, 
        visualize=not args.no_vis,
        voxel_size=args.voxel_size
    )


if __name__ == "__main__":
    main()