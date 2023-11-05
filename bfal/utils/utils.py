import torch
import numpy as np

def points_aligned_by_axis(points, axis_value, y_axis=True, th=0.5):
    
    for (x, y, *_) in points:
        t_axis = y if y_axis else x # target axis
        # check if target axis is out of bounds within the given axis value plus its threshold
        if not is_value_within(t_axis, axis_value, tolerance=th):
            return False
    
    return True

def normalize_vector(vector):
    # Calculate the magnitude (length) of the vector
    magnitude = torch.sqrt(vector[0] ** 2 + vector[1] ** 2)

    # Avoid division by zero by checking if the magnitude is nonzero
    if magnitude != 0:
        # Calculate the normalized components
        normalized_x = vector[0] / magnitude
        normalized_y = vector[1] / magnitude
        return (normalized_x, normalized_y)
    else:
        # Handle the case where the vector has zero magnitude (it cannot be normalized)
        raise ValueError("Cannot normalize a vector with zero magnitude.")

def get_distance_of_2_points(pt1, pt2):
        return torch.sqrt(torch.pow(pt2[0] - pt1[0], 2) + torch.pow(pt2[1] - pt1[1], 2))

def curveness_difference(points):
    """
        This computes the curveness of given points base on their order.
        
        Calculation is done by comparing the distance between the first and last point and the sum of distance of each
        points with respect to their order.
    """
    if len(points) <= 1:
        return -1

    straight_distance = get_distance_of_2_points(points[0], points[-1])

    if len(points) == 2:
        return straight_distance

    total_distance = 0
    prev_point = None
    for (x, y, *_) in points:
        pt = (x, y)
        if prev_point != None:
            total_distance += get_distance_of_2_points(prev_point, pt)
        
        prev_point = pt

    return torch.abs(total_distance - straight_distance)


def extend_line_to_y(x1, y1, x2, y2, Y_desired):
    # Calculate the slope of the line
    m = (y2 - y1) / (x2 - x1)
    
    # Calculate the X-coordinate where the line intersects Y_desired
    x = (Y_desired - y1) / m + x1
    
    return x

def midpoint(point1, point2):
    """
    Calculate the midpoint between two points.

    Args:
        point1 (tuple): The coordinates of the first point (x1, y1).
        point2 (tuple): The coordinates of the second point (x2, y2).

    Returns:
        tuple: The coordinates of the midpoint.
    """
    x1, y1 = point1
    x2, y2 = point2

    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2

    return (mid_x, mid_y)


def center_of_circular_point(points):
    sum_x = 0.0
    sum_y = 0.0

    # Iterate through the points and accumulate the sum of x and y coordinates
    for point in points:
        sum_x += point[0]
        sum_y += point[1]

    # Calculate the mean (average) x and y coordinates
    mean_x = sum_x / len(points)
    mean_y = sum_y / len(points)

    return torch.tensor((mean_x, mean_y)).to('cuda')

def find_intersection(point1, point2, point3, point4):
    # Convert points to NumPy arrays
    p1 = np.array(point1)
    p2 = np.array(point2)
    p3 = np.array(point3)
    p4 = np.array(point4)

    # Create direction vectors of the two lines
    dir1 = p2 - p1
    dir2 = p4 - p3

    # Check if the lines are parallel
    if np.cross(dir1, dir2) == 0:
        return None  # Lines are parallel, no intersection

    # Calculate t values for each line
    t1 = np.cross(p3 - p1, dir2) / np.cross(dir1, dir2)
    t2 = np.cross(p3 - p1, dir1) / np.cross(dir1, dir2)
    
    # Check if the intersection point is within the line segments
    if 0 <= t1 <= 1 and 0 <= t2 <= 1:
        intersection = p1 + t1 * dir1
        return tuple(intersection)

    return None  # Intersection is outside the line segments


def crop_9_16(image):
    # Calculate the dimensions for the 9:16 aspect ratio
    height = image.shape[0]
    width = int(height * 9 / 16)  # 9:16 aspect ratio

    # Calculate the cropping area to center the image
    start_x = (image.shape[1] - width) // 2
    end_x = start_x + width

    # Crop the image to the calculated dimensions
    cropped_image = image[:, start_x:end_x]
    
    return cropped_image

def is_value_within(value: float, ref_value: float, tolerance: float, equals=True) -> bool:
    # check if value is within the range of ref value plus minus tolerance
    if equals:
        return (ref_value - tolerance) <= value <= (ref_value + tolerance)
    
    return (ref_value - tolerance) < value < (ref_value + tolerance)