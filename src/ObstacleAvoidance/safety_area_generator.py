
from shapely.geometry import Polygon, Point, MultiPolygon

from shapely.ops import unary_union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

def generate_safety_bounds(init_pose, end_pose, 
                           length, width, offset, 
                           objects_list=None, dist_to_docks=2.0):
    """
    Generates safety bounds polygon with circular cut-outs removed.

    Ensures:
    - Both start and end pose are inside the resulting safety area.
    - Margin dist_to_docks to the outer safety area edge.
    - Cut-outs may split area; only returns the part containing both start and end.

    Parameters:
        init_pose (np.ndarray): [x, y, psi]
        end_pose (np.ndarray): [x, y, psi]
        length (float): Proposed length of safety rectangle (along centerline)
        width (float): Proposed width of safety rectangle (perpendicular to centerline)
        offset (float): Perpendicular offset of safety rectangle
        objects_list (list of [x, y, radius]): List of circles to subtract
        dist_to_docks (float): Minimum margin from start/end to any edge

    Returns:
        shapely.geometry.Polygon: Final safety area polygon with cut-outs
    """
    # --- STEP 1: Compute outer safety rectangle with margin checks ---

    x_start, y_start = init_pose[:2]
    x_end, y_end = end_pose[:2]
    
    # Centerline and perpendicular direction
    vec = np.array([x_end - x_start, y_end - y_start])
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("Start and end pose are the same; cannot define centerline.")
    direction = vec / norm
    perp_direction = np.array([-direction[1], direction[0]])
    
    # Margin checks
    start_vec = np.array([x_start, y_start])
    end_vec = np.array([x_end, y_end])
    midpoint = (start_vec + end_vec) / 2

    start_rel = start_vec - midpoint
    end_rel = end_vec - midpoint

    proj_start_long = np.abs(np.dot(start_rel, direction))
    proj_end_long = np.abs(np.dot(end_rel, direction))
    max_proj_long = max(proj_start_long, proj_end_long)

    proj_start_lat = np.abs(np.dot(start_rel, perp_direction))
    proj_end_lat = np.abs(np.dot(end_rel, perp_direction))
    max_proj_lat = max(proj_start_lat, proj_end_lat)

    required_half_length = max_proj_long + dist_to_docks
    required_half_width = max_proj_lat + dist_to_docks + abs(offset)

    half_length = max(length / 2, required_half_length)
    half_width = max(width / 2, required_half_width)

    center = midpoint + offset * perp_direction

    # Safety area corners
    p1 = center + half_length * direction + half_width * perp_direction
    p2 = center + half_length * direction - half_width * perp_direction
    p3 = center - half_length * direction - half_width * perp_direction
    p4 = center - half_length * direction + half_width * perp_direction
    safety_corners = np.array([p1, p2, p3, p4])

    safety_polygon = Polygon(safety_corners)

    if not objects_list:
        return safety_polygon

    # --- STEP 2: Subtract all circular objects ---
    circle_polygons = [Point(obj[0], obj[1]).buffer(obj[2]) for obj in objects_list]
    all_circles_union = unary_union(circle_polygons)

    cutout_polygon = safety_polygon.difference(all_circles_union)

    if cutout_polygon.is_empty:
        raise ValueError("Resulting safety area is empty after subtraction.")

    # --- STEP 3: Ensure both poses are inside ---
    start_point = Point(x_start, y_start)
    end_point = Point(x_end, y_end)

    if isinstance(cutout_polygon, Polygon):
        if cutout_polygon.contains(start_point) and cutout_polygon.contains(end_point):
            return cutout_polygon
        else:
            raise ValueError("Resulting area does not contain both start and end pose.")

    elif hasattr(cutout_polygon, 'geoms'):
        for poly in cutout_polygon.geoms:
            if poly.contains(start_point) and poly.contains(end_point):
                return poly
        raise ValueError("No single connected area contains both start and end pose after subtraction.")

    else:
        raise ValueError("Unsupported geometry type after subtraction.")

def plot_safety_polygon_and_poses(init_pose, end_pose, safety_polygon):
    """
    Plot start and end poses as arrows and safety area as polygon(s) in NED frame.

    North is plotted vertically (y-axis), East horizontally (x-axis).
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')

    def transform_coords(coords):
        """Swap x and y for NED to plotting (East, North)."""
        coords = np.array(coords)
        return np.column_stack((coords[:,1], coords[:,0]))

    def plot_polygon(polygon, **kwargs):
        # Exterior
        exterior_coords = transform_coords(polygon.exterior.coords)
        ax.add_patch(MplPolygon(exterior_coords, closed=True, **kwargs))
        # Interiors (holes)
        for interior in polygon.interiors:
            interior_coords = transform_coords(interior.coords)
            ax.add_patch(MplPolygon(interior_coords, closed=True, facecolor='white', edgecolor='black', alpha=0.5))

    # --- Plot safety area ---
    if isinstance(safety_polygon, Polygon):
        plot_polygon(safety_polygon, fill=True, alpha=0.2, edgecolor='black', facecolor='lightblue')
    elif isinstance(safety_polygon, MultiPolygon):
        for poly in safety_polygon.geoms:
            plot_polygon(poly, fill=True, alpha=0.2, edgecolor='black', facecolor='lightblue')
    else:
        raise ValueError("safety_polygon must be Polygon or MultiPolygon")

    # --- Plot start and end positions as points in East-North ---
    ax.plot(init_pose[1], init_pose[0], 'go', label='Start Pose')  # swap
    ax.plot(end_pose[1], end_pose[0], 'ro', label='End Pose')      # swap

    # --- Plot arrows for heading ---
    arrow_scale = 1.0
    def ned_arrow_components(psi):
        # For NED psi, direction in North-East frame: (cos(psi), sin(psi))
        # When plotting East-North, swap → (sin(psi), cos(psi))
        return np.sin(psi), np.cos(psi)
    
    dx_init, dy_init = ned_arrow_components(init_pose[2])
    dx_end, dy_end = ned_arrow_components(end_pose[2])

    ax.arrow(init_pose[1], init_pose[0], arrow_scale * dx_init, arrow_scale * dy_init,
             head_width=0.3, head_length=0.5, fc='green', ec='green')
    ax.arrow(end_pose[1], end_pose[0], arrow_scale * dx_end, arrow_scale * dy_end,
             head_width=0.3, head_length=0.5, fc='red', ec='red')

    # --- Axis labels and grid ---
    ax.set_xlabel('East [m]')
    ax.set_ylabel('North [m]')
    ax.grid(True)
    ax.legend()
    ax.set_title('Safety Area with Dock Cut-out and Start/End Poses (NED Frame)')

    plt.show()

def sample_object_from_area(safety_polygon, init_pose, end_pose, margin=1.0, max_radius=5.0):
    """
    Sample a random point within the safety polygon and return a circle object.
    
    Parameters:
        safety_polygon (shapely.geometry.Polygon): The safety area polygon.
        init_pose (np.ndarray): [x, y, psi] of the start pose.
        end_pose (np.ndarray): [x, y, psi] of the end pose.
        margin (float): Safety margin to keep from start and end poses.
        max_radius (float): Maximum radius for the sampled circle.

    Returns:
        list: [x, y, radius] of the sampled circle object.
    """
    minx, miny, maxx, maxy = safety_polygon.bounds
    while True:
        x_sample = np.random.uniform(minx, maxx)
        y_sample = np.random.uniform(miny, maxy)
        candidate_point = Point(x_sample, y_sample)

        if safety_polygon.contains(candidate_point):
            # Compute Euclidean distances to start and end points
            distance_to_end = np.hypot(x_sample - end_pose[0], y_sample - end_pose[1])
            distance_to_start = np.hypot(x_sample - init_pose[0], y_sample - init_pose[1])
            
            # Set safety margin
            clipped_max_radius = min(max(distance_to_end - margin, 0.1), 
                                     max(distance_to_start - margin, 0.1), 
                                     max_radius)
            
            # Sample radius within allowed range
            radius = np.random.uniform(0.5, clipped_max_radius)
            
            return [x_sample, y_sample, radius]

if __name__ == "__main__":
    from shapely.geometry import LineString, Point
    # Example usage
    init_pose = np.array([0, 0, 0])  # [x, y, psi]
    end_pose = np.array([10, 0, -np.pi])  # [x, y, psi]
    length = 10.0
    width = 10.0
    offset = 1.0

    obstacles = [
        [2, 6, 2.5],  # Circle at (2, 0) with radius 0.5
        [5, 0, 0.5],  # Circle at (5, 0) with radius 0.5
        [8, 0, 0.5]   # Circle at (8, 0) with radius 0.5
    ]

    safety_polygon = generate_safety_bounds(init_pose, end_pose, length, width, offset, objects_list=None, dist_to_docks=2.0)
    # vessel_pose = np.array([0, 10, 0])

    # Example: assume you want N random points
    N = 10
    sampled_points = []

    while len(sampled_points) < N:
        sampled_points.append(sample_object_from_area(safety_polygon, init_pose, end_pose, margin=1.0, max_radius=10.0))


    objects_list = sampled_points
    safety_polygon = generate_safety_bounds(init_pose, end_pose, length, width, offset, objects_list, dist_to_docks=2.0)
    # print(lookahead_points)
    plot_safety_polygon_and_poses(init_pose, end_pose, safety_polygon)
