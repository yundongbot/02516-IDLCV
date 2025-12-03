import random
from scipy.ndimage import center_of_mass, label
from skimage import measure
from scipy.ndimage import distance_transform_edt
import numpy as np

def generate_clicks(mask, num_clicks=2, include_background_clicks=True, sampling_strategy='centroid'):
    pos_clicks = []
    neg_clicks = []

    if sampling_strategy == 'centroid':
        centroid_list = centroid_points(mask)
        if len(centroid_list) == 0:
            return np.array(pos_clicks)
        while len(pos_clicks) < num_clicks:
            for c_x, c_y in centroid_list:
                if len(pos_clicks) < num_clicks:
                    pos_clicks.append([c_x, c_y])
                else:
                    break
            if len(pos_clicks) < num_clicks:
                for c_x, c_y in centroid_list:
                    x_click, y_click = add_random_offset(c_x, c_y, mask.shape)
                    pos_clicks.append([x_click, y_click])
                    if len(pos_clicks) >= num_clicks:
                        break
    elif sampling_strategy == 'random':
        pos_clicks += random_points_within_object(mask, num_clicks)
    elif sampling_strategy == 'boundary':
        pos_clicks += boundary_points(mask, num_clicks)
    else:
        raise ValueError("Invalid sampling strategy")

    # Background clicks
    if include_background_clicks:
        neg_clicks += random_points_in_background(mask, num_clicks, min_distance=5)

    return np.array(pos_clicks), np.array(neg_clicks)

def add_random_offset(c_x, c_y, shape, offset_range=5):
    x_offset = random.randint(-offset_range, offset_range)
    y_offset = random.randint(-offset_range, offset_range)
    x_click = c_x + x_offset
    y_click = c_y + y_offset
    # Ensure the click is within the image bounds
    height, width = shape
    x_click = min(max(x_click, 0), width - 1)
    y_click = min(max(y_click, 0), height - 1)
    return x_click, y_click

def centroid_points(mask):
    mask_binary = mask > 0.5
    labeled_mask, num_features = label(mask_binary)
    centroids = []
    for region_label in range(1, num_features + 1):
        region = (labeled_mask == region_label)
        if np.sum(region) == 0:
            continue
        c_y, c_x = center_of_mass(region)
        centroids.append((int(c_x), int(c_y)))
    return centroids

def random_points_within_object(mask, num_clicks):
    y_coords, x_coords = np.where(mask > 0.5)
    if len(x_coords) == 0:
        return []  # No foreground pixels found
    # Create a list of indices and shuffle them
    indices = list(range(len(x_coords)))
    random.shuffle(indices)
    selected_clicks = []
    for idx in indices[:num_clicks]:
        selected_clicks.append([x_coords[idx], y_coords[idx]])
    return selected_clicks

def boundary_points(mask, num_clicks):
    contours = measure.find_contours(mask, level=0.5)
    if not contours:
        return []  # No contours found
    boundary = np.vstack(contours)
    num_boundary_points = len(boundary)
    if num_boundary_points == 0:
        return []
    # Create a list of indices and shuffle them
    indices = list(range(num_boundary_points))
    random.shuffle(indices)
    selected_clicks = []
    for idx in indices[:num_clicks]:
        y, x = boundary[idx]
        selected_clicks.append([int(x), int(y)])
    return selected_clicks

def random_points_in_background(mask, num_clicks, min_distance=5):
    background_mask = (mask <= 0.5).astype(np.uint8)
    distance_map = distance_transform_edt(background_mask)
    far_background_mask = (distance_map >= min_distance)
    y_coords, x_coords = np.where(far_background_mask)
    if len(x_coords) == 0:
        return []  # No suitable background pixels found
    # Create a list of indices and shuffle them
    indices = list(range(len(x_coords)))
    random.shuffle(indices)
    selected_clicks = []
    for idx in indices[:num_clicks]:
        selected_clicks.append([x_coords[idx], y_coords[idx]])
    return selected_clicks
