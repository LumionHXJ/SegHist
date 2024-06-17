from typing import Tuple, Sequence, Union, Optional

import numpy as np
from shapely.geometry import Polygon, JOIN_STYLE
import cv2

from mmocr.utils.polygon_utils import poly_make_valid, offset_polygon

def iou(poly1, poly2):
     poly1 = Polygon(poly1)
     poly2 = Polygon(poly2)
     return poly1.intersection(poly2).area / poly1.union(poly2).area

def get_distance(polygon: np.ndarray,
                      shrink_ratio: float,
                      ) -> float:
    """
    Compute the shrinkage distance of a polygon with respect to a given 
    shrink ratio. This function is in reference to the PSENet approach.

    ALERT! distance is compute by A(1-r)/L not A(1-r^2)/L

    Args:
        polygon (np.ndarray): An array representing the vertices of the polygon.
        The shape of the array should be (num_points, 2), where each row represents 
        the (x, y) coordinates of a vertex.
        shrink_ratio (float): The ratio by which the polygon is to be shrunk. 
        It's a value less than 1, where 1 means no shrinkage.

    Returns:
        distance(float): The calculated distance by which the polygon should 
        be shrunk.
    """
    poly = polygon.reshape(-1, 2)    
    poly_obj = Polygon(poly)
    area = poly_obj.area
    peri = poly_obj.length
    distance = area * (1 - shrink_ratio) / (peri + 1e-5)
    return distance

def expand_poly(
        polygon: np.ndarray,
        shrink_ratio: float,
        stretch_ratio: float,
    ) -> np.ndarray:
    """Generate text instance kernels according to a shrink ratio.

    Args:
        polygon (np.ndarray): array of text polygons.

    Returns:
        polygon after expansion by TKS.
    """
    poly = polygon.copy().reshape(-1, 2).astype(np.float32)
    distance = get_distance(poly, shrink_ratio)

    # stretching on horizental
    poly[:, 0] *= stretch_ratio

    # no splits happen in expansion
    poly = poly_make_valid(Polygon(poly))
    expand_poly = np.array(poly.buffer(distance, 
                                       ).exterior.coords)
    expand_poly = expand_poly.reshape(-1, 2).astype(np.float32)
    expand_poly[:, 0] /= stretch_ratio

    return expand_poly 

def stretch_kernel(
        polygon: np.ndarray,
        shrink_ratio: float,
        stretch_ratio: float,
    ) -> np.ndarray:
    poly = polygon.copy().reshape(-1, 2).astype(np.float32)
    
    # get shrink distance before stretching
    distance = get_distance(poly, shrink_ratio)

    # stretch on x-axis
    poly[:, 0] *= stretch_ratio

    # shrink poly
    shrunk_poly = Polygon(poly).buffer(-distance)

    # if splits into multiple parts
    if not isinstance(shrunk_poly, Polygon):
        return np.array([]).reshape(0,2)
    
    shrunk_poly = np.array(shrunk_poly.exterior.coords)

    # shrink to NULL
    if len(shrunk_poly) == 0:
        return shrunk_poly
    
    shrunk_poly = shrunk_poly.reshape(-1, 2).astype(np.float32)
    shrunk_poly[:, 0] /= stretch_ratio
    return shrunk_poly 

def unstretch_kernel(poly_pts: np.ndarray,
                     shrink_ratio: float,
                     stretch_ratio: float,
                     refinement: bool = True,
                     unclip_ratio: float = 0,
                     refine_epoch: int = 30,
                     step_size: float = 1.0,
                     tolerance: float = 0.4) -> np.ndarray:
    """Unclip a polygon either adaptively or by fixed ratio. 
    Only used in postprocessor.

    Args:
        poly_pts (np.ndarray): The polygon points.
        shrink_ratio(float): r used in module loss.
        refinement(bool): whether doing refinement, if `refinement=false`, 
        then unclip polygons by fixed ratio `unclip_ratio`.

    Returns:
        np.ndarray: The expanded polygon points.
    """
    poly_pts = poly_pts.copy().reshape(-1, 2)
    poly = poly_pts.astype(np.float32)

    if refinement:
        # unclip adaptively
        _, (_x, _y), _ = cv2.minAreaRect(poly)
        poly[:, 0] *= stretch_ratio
        _, (_kx, _ky), _ = cv2.minAreaRect(poly)

        # adaptive distance: distance nearly same as shrink
        # b is approximate (Maintaining rotation symmetry)
        a = 4 * (1 / stretch_ratio + 1) - 4 / stretch_ratio * (1 - shrink_ratio)
        b = 2 * (_x + _y) - 2 / stretch_ratio * (_kx + _ky) * (1 - shrink_ratio)
        c = - _x * _y * (1 - shrink_ratio)
        distance = (- b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
        assert distance >= 0, 'dilate should have d > 0'
        step_size = max(distance / 2, step_size)
    else:
        # by fixed ratio
        p = Polygon(poly)
        distance = p.area * unclip_ratio / p.length
        poly[:, 0] *= stretch_ratio
        refine_epoch = 0

    poly = poly_make_valid(Polygon(poly))
    expand_poly = poly.buffer(distance, )
    expand_poly = np.array(expand_poly.exterior.coords)
    expand_poly[:, 0] /= stretch_ratio

    greater = None
    for _ in range(refine_epoch):
        # get shrink distance from newly recovered polygon
        distance_0 = get_distance(expand_poly, shrink_ratio) 

        if distance_0 > distance + tolerance:
            if greater is not None and not greater:
                # scale step
                step_size /= 2
            greater = True
            distance += step_size                
        elif distance_0 < distance - tolerance:
            if greater:
                # scale step
                step_size /= 2
            greater = False
            distance -= step_size
            distance = max(distance, 0)
        else:
            break 
        expand_poly = poly.buffer(distance)
        expand_poly = np.array(expand_poly.exterior.coords) 
        expand_poly[:, 0] /= stretch_ratio  

    return expand_poly

def align_polygon(polygon: np.ndarray, stride: int) -> np.ndarray:
    return (polygon / stride) - (stride - 1) / (2 * stride)

def fill_hole(binary_image):
    floodfilled = binary_image.copy()

    h, w = binary_image.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    cv2.floodFill(floodfilled, mask, (0, 0), 255)
    floodfilled_inv = cv2.bitwise_not(floodfilled)
    out_image = binary_image | floodfilled_inv

    return out_image
