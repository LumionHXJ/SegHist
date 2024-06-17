from typing import Sequence, Sequence, Union
import warnings

import numpy as np
from numpy.typing import *
import cv2
from sklearn.cluster import KMeans

def dist(p1: NDArray, p2: NDArray):
    """Compute Euclid distance between p1 and p2.

    Args:
        p1(NDArray) and p2(NDArray) have same shape (..., k).    
    """
    return np.sqrt(np.sum((p1-p2)**2, axis=-1))

def norm(p: NDArray):
    """Perform L2-normalization on the given array.

    This function normalizes the input array `p` using the L2 norm, also known
    as the Euclidean norm. The L2 norm is calculated as the square root of the
    sum of the squared elements of `p`. The normalization process scales the 
    elements of `p` so that the length of the resultant vector is 1. This is 
    commonly used in machine learning and statistics to normalize the input 
    features or data points.

    Parameters:
    p (NDArray): A numpy array of any shape, where the normalization is applied 
    along the last dimension.

    Returns:
    NDArray: The L2-normalized array, having the same shape as the input array `p`.

    Example:
    >>> import numpy as np
    >>> p = np.array([[1, 2, 3], [4, 5, 6]])
    >>> norm(p)
    array([[0.26726124, 0.53452248, 0.80178373],
           [0.45584231, 0.56980288, 0.68376346]])

    Note:
    The function assumes that the input array `p` is not the zero vector, as the L2
    norm of a zero vector is undefined.
    """
    return p / (np.sqrt(np.sum(p ** 2, axis=-1)))

def section_iou(l1: NDArray, l2:NDArray):
    """
    Calculate the Intersection over Union (IoU) of two one-dimensional sections.
    Each sections contains k points, but only edge points are contributed.

    This function computes the IoU of two line segments, l1 and l2. Each segment 
    is represented by a series of points. The IoU is calculated as the length of 
    the intersection of the two segments divided by the length of their union. 

    The segments are defined in an unordered manner, meaning that for each 
    segment, the start and end points are not necessarily in increasing order.

    Args:
        l1 (NDArray): An array representing the first segment, shape (k, ). 
        l2 (NDArray): An array representing the second segment, shape (k, ). 

    Returns:
        float: The IoU of the two segments. The value ranges from 
        0 (no overlap) to 1 (full overlap).

    Example:
    >>> import numpy as np
    >>> l1 = np.array([1, 3, 2])
    >>> l2 = np.array([2, 4, 3])
    >>> section_iou(l1, l2)
    0.3333333333333333

    Note:
    The function includes a small constant (1e-4) in the denominator to avoid 
    division by zero in case the union of the segments has zero length.
    """
    less = (np.min(l1), np.min(l2))
    greater = (np.max(l1), np.max(l2))
    _iou = max(0, (np.min(greater) - np.max(less)) / (np.max(greater) - np.min(less) + 1e-4))
    return _iou

def uniform_curve_sampling(curve: NDArray, points: int):
    """Uniformly sample a specified number of points on a given curve.

    This function takes a curve represented by a series of points and samples
    a fixed number of points from it in a uniform manner, based on the 
    cumulative length of the curve. It guarantees that the starting and 
    ending points of the curve are included in the sampled points.

    Args:
        curve (NDArray): A numpy array representing the curve. The array 
        should have the shape (k, 2), where k is the number of points in 
        the curve and each point is a 2D coordinate (x, y).
        points (int): The number of points to sample from the curve.

    Returns:
        NDArray: A numpy array of the uniformly sampled points with the shape
        (p, 2), where p is equal to the 'points' argument.

    The function works by first calculating the length of each segment of the 
    curve, then accumulating these lengths to find the total length of the curve. 
    It then determines the positions along the curve where the uniformly spaced 
    points should be, and interpolates these points based on the nearest segments 
    in the original curve.
    """
    k = curve.shape[0] # points in original curve
    
    segment_length = dist(curve, np.concatenate([curve[0:1], curve[:-1]], axis=0)) # (k, )
    accumulate_length = np.cumsum(segment_length) # (k, )
    curve_length = accumulate_length[-1]

    sample_results = np.zeros((points, 2))
    sample_results[0] = curve[0]
    for p in range(1, points):
        curr_length = curve_length * p / (points - 1)

        # default return value v in (a(i-1), a(i)]
        curr_segment = np.searchsorted(accumulate_length, curr_length)  
        curr_segment = min(k-1, curr_segment) # precision problem in division may ocurr

        theta = (accumulate_length[curr_segment] - curr_length) / segment_length[curr_segment]
        sample_results[p] = theta * curve[curr_segment-1] + (1-theta) * curve[curr_segment]

    return sample_results

def extra_sampling(array: NDArray, extra_points: int):
    """Doing extra sampling to `array`, sample `extra_points` uniformly on each side.
    
    Args:
        arrray(NDArray): supporting two-dim array.
    """
    result = []

    # sample each pair of neighboring elements
    for i in range(len(array) - 1):
        samples = np.linspace(array[i], array[i + 1], extra_points + 1, endpoint=False)[1:]
        result.extend(samples)

    # adding last element
    result.append(array[-1])

    return np.array(result)

def compute_text_direction(polygon: NDArray):
    """Return normalized direction vector of a text region.
    Direction vector is along the positive Y-axis.
    """
    vec1 = polygon[len(polygon)//2-1] - polygon[0]
    vec1 = vec1 if vec1[1] >= 0 else -vec1
    vec2 = polygon[len(polygon)//2] - polygon[-1]
    vec2 = vec2 if vec2[1] >= 0 else -vec2
    mean =  (vec1 + vec2) / 2
    return norm(mean)

def find_top_bottom(polygon: NDArray):
    """Receive reordered polygon, find its top and bottom.
    Returns:
        in top-bottom order
    """
    line0 = np.array([polygon[0], polygon[-1]])
    line1 = np.array(polygon[len(polygon)//2-1:len(polygon)//2+1])
    if np.mean(line0[:, 1]) < np.mean(line1[:, 1]):
        return line0, line1
    else:
        return line1, line0
    
class ImageToolkits:
    """ImageToolkits class can achieve the following functionalities:
        1. Separate single-line body text and double-line annotations in 
        historical document images, and output corresponding JSON annotations.
        2. Rectify the polygon representation of text lines, where the first 
        n points correspond to one long edge, the last n points correspond to 
        the opposite long edge, and the two long edges are joined end-to-end.
        3. Calculate the aspect ratio of text lines (defined as the ratio of 
        the long edge to the short edge).
        4. Calculate the vertical aspect ratio of text lines in historical 
        documents (defined as the ratio of the vertical edge to the horizontal 
        edge).
        5. Calculate the text center line(compared to the text kernel, there 
        is no shrinkage along text direction).
        6. Check the text line orientation in historical documents.
    
    Args:
        polygons(Sequence[NDArray]): Text regions in the image.
        image_shape(NDArray): The shape of the image in (height, width).
        image_path(str): Path to the image.
        texts(Sequence[str]): Text annotations for regions, aiding in text line 
        localization.
        points(int): Number of samples taken along each long edge to determine 
        the length of the short edge.
        cluster_thresh(float): Determines whether the historical document 
        contains only single-line body text.
        shrink_ratio(Union[float, Sequence]): The width of the text central region 
        is 1/r times the width of the text region; using a single value indicates 
        the same shrinkage ratio for both single-line body text and double-line 
        annotations, while using two values indicates different shrinkage ratios.
        reorder(bool): Whether it is necessary to rearrange the annotation order of 
        polygons. If the image is not a document image, please pass false.
    """
    SINGLE_ENTRY = 0
    DOUBLE_ENTRY = 1 # having thinner width
    TO_BE_DETERMINED = -1
    def __init__(self, 
                 polygons: Sequence[NDArray],
                 image_shape: NDArray = None,
                 image_path: str = None,
                 texts: Sequence[str] = None,
                 points: int = 30,
                 cluster_thresh: float = 1.3,
                 shrink_ratio: Union[float, Sequence] = 3,
                 reorder: bool = False):
        self.image_shape = image_shape
        self.image_path = image_path
        self.polygons = polygons
        self.points = points
        self.texts = texts
        self.num_instance = len(polygons)
        self.cluster_thresh = cluster_thresh
        self.shrink_ratio = np.array(shrink_ratio) if isinstance(shrink_ratio, Sequence) \
            else np.array((shrink_ratio, shrink_ratio))
        for r in self.shrink_ratio:
            assert r > 1, 'Centerline must have proper shrink ratio r > 1.'
        self.reorder = reorder

    @classmethod
    def fitting2reorder(cls, poly, m=0, k=5):
        """
        Args:
            m: extra sampling
            k: degree of fitting polynominal

        Returns:
            fit_mse: fitting error.
            mse: fitting error of each side.
            polynominal: coefficient of polynominal.
        """
        fit_mse = []
        mse = []
        polynomimal = []
        for _ in range(len(poly) // 2):
            # part 1: fetch out each curve in same order(top to down or vise versa)
            curve_a = poly[_: _+len(poly)//2]
            curve_b = np.concatenate([poly[_+len(poly)//2: ], poly[: _]], axis=0)
            curve_b = curve_b[::-1]

            # part 2: extra sampling
            curve_a = extra_sampling(curve_a, m)
            curve_b = extra_sampling(curve_b, m)
            
            # part 3: fitting with polynominal: x = f(y)
            poly_eff_a = np.polyfit(curve_a[:, 1], curve_a[:, 0], k)
            poly_eff_b = np.polyfit(curve_b[:, 1], curve_b[:, 0], k)

            poly_a = np.poly1d(poly_eff_a)
            poly_b = np.poly1d(poly_eff_b)

            # part 4: fitting
            fit_aj = poly_a(curve_a[:, 1])
            fit_bj = poly_b(curve_b[:, 1])

            # part 5: compute fitting error
            mse_a = np.sum((fit_aj - curve_a[:, 0]) ** 2)
            mse_b = np.sum((fit_bj - curve_b[:, 0]) ** 2)
            
            # part 6: append return list
            polynomimal.append((poly_a, poly_b))
            mse.append((mse_a, mse_b))
            fit_mse.append(mse_a + mse_b)

        return fit_mse, mse, polynomimal
    
    def reorder_polygon(self, 
                        k: int = 5,
                        line_ratio: float = 5.0,
                        extra_points: int = 2):
        """Reorder all polygons and find out each long curve. The process keep the 
        order between `self.polygons` unchanged.

        If the instance is less-point annotated and hard to determine longerside,
        then save it and wait for the help of overdetermined results.

        Long curve will be save in attribute `self.polygons`, `self.polygons` is a 
        list of np.array, which first half represents a long curve.

        Args:
            k(int): the degree of polyfit
            line_ratio(float): if the long length is `line_ratio` times as long as short 
            one, the long curve can be determined.
            extra_points(int): extra points to sample when fitting the curve.
        """
        if self.reorder:
            return
        
        reordered_polygon = []
        to_be_determined = []
        to_be_determined_index = []
        text_direction = [] # end-start, (0, 1)

        for i, poly in enumerate(self.polygons):
            assert len(poly) % 2 == 0 and len(poly) >= 4, \
                f'polygon must contains 2k(at least 4) points but receive {poly}.'

            # two-point(line) annotation: cannot determined
            if len(poly) == 4:
                
                # dist compute: 0->3, 1->2
                dist_1 = np.sum(dist(poly[:2], poly[2:][::-1]))
                
                # dist compute: 0->1, 3->2
                dist_2 = np.sum(dist(np.array([poly[0], poly[-1]]), poly[1:3]))
                
                # if one set of sides is significantly longer 
                if dist_1 > dist_2 * line_ratio:
                    result = np.concatenate([poly[1:], poly[:1]], axis=0)
                    text_direction.append(compute_text_direction(result))
                    reordered_polygon.append(result)
                elif dist_2 > dist_1 * line_ratio:
                    result = poly
                    text_direction.append(compute_text_direction(result))
                    reordered_polygon.append(result)
                else:
                    # if the results cannot be determined now, reorder it later
                    # Note: keep the order inside polygons
                    to_be_determined.append(poly)
                    to_be_determined_index.append(i)
                    reordered_polygon.append([])
                continue

            # promise no underdetermined problem
            if len(poly) <= 2 * (k + 1):
                m = max(extra_points, np.ceil((len(poly)//2 - (k + 1)) / (len(poly) // 2 - 1)).astype(np.int32))
            else:
                m = extra_points

            fit_mse, mse, polynomial = self.fitting2reorder(poly, m, k)
                    
            min_fit_mse = np.argmin(fit_mse)
            result = np.concatenate([poly[min_fit_mse:], poly[:min_fit_mse]], axis=0)

            # try refining if result isn't ideal
            if not (1/3 < mse[min_fit_mse][0] / (mse[min_fit_mse][1] + 1e-6) < 3) and \
                fit_mse[min_fit_mse] > 10:
                # deleting side with imbalance points
                argmin = np.array(mse[min_fit_mse]).argmin()
                if argmin == 0:
                    to_delete = poly[min_fit_mse: min_fit_mse+len(poly)//2]
                else:
                    # second curve no need to reverse
                    to_delete = np.concatenate([poly[min_fit_mse+len(poly)//2: ], poly[: min_fit_mse]], axis=0)
                
                # remove two points with lowest fitting error
                fit_error = polynomial[min_fit_mse][argmin](to_delete[:, 1]) - to_delete[:, 0]
                del_points = fit_error.argsort()[:2]

                # construct new poly
                avail_index = np.ones(to_delete.shape[:1], dtype=bool)
                avail_index[del_points] = False
                if argmin == 0:
                    another_curve = np.concatenate([poly[min_fit_mse+len(poly)//2: ], poly[: min_fit_mse]], axis=0)
                    poly = np.concatenate([to_delete[avail_index], another_curve], axis=0)
                else:
                    another_curve = poly[min_fit_mse: min_fit_mse+len(poly)//2]
                    poly = np.concatenate([another_curve, to_delete[avail_index]], axis=0)
                fit_mse_n, mse_n, _ = self.fitting2reorder(poly, m, k)
                min_fit_mse_n = np.argmin(fit_mse_n)

                image_path = getattr(self, "image_path", "<UNK_IMG>")
                text = self.texts[i] if isinstance(getattr(self, "texts", None), list) and i < len(self.texts) else "<UNK>"
                if np.min(fit_mse) / (np.min(fit_mse_n) + 1e-3) > 5:
                    result = np.concatenate([poly[min_fit_mse_n:], poly[:min_fit_mse_n]], axis=0)
                    action = "replacing"
                else:
                    action = "keep"
                print(f"{image_path} {text}: {action} old {np.min(fit_mse)} by {'new' if action == 'replacing' else 'rejecting new'} {np.min(fit_mse_n)}")

            text_direction.append(compute_text_direction(result))
            reordered_polygon.append(result)
        
        document_direction = np.mean(text_direction, axis=0) # no need to normalize again
        for i, poly in zip(to_be_determined_index, to_be_determined):
            direct1 = compute_text_direction(poly)
            direct2 = compute_text_direction(np.concatenate([poly[1:], poly[:1]], axis=0))
            if np.sum(direct1 * document_direction) > np.sum(direct2 * document_direction):
                reordered_polygon[i] = poly
                text_direction.append(direct1)
            else:
                reordered_polygon[i] = np.concatenate([poly[1:], poly[:1]], axis=0)
                text_direction.append(direct2)

        self.polygons = reordered_polygon
        self.direction = norm(np.mean(text_direction, axis=0))

        self.check_polygon_order()
    
    def determine_short_length(self,
                               points_a: NDArray, 
                               points_b: NDArray):
        """Determine the shorter curve of polygon.

        Args:
            points_a(NDArray): shape-like (n, p, 2).
            points_b(NDArray): shape-like (n, p, 2).
            points_a and points_b are return value of function `uniform_curve_sampling`.  

        Returns:
            short side length(NDArray): (n, )  
        """

        raw_dist = dist(points_a, points_b) # n, p

        # using IQR identify outliers
        q1 = np.percentile(raw_dist, 25, axis=-1) # n,
        q3 = np.percentile(raw_dist, 75, axis=-1)
        iqr = q3 - q1
        lower_bound = q1 - 1 * iqr
        upper_bound = q3 + 1 * iqr

        # need to filter outliers one by one (due to numbers of outliers isn't same)
        mean = np.zeros((len(points_a), )) # n,
        for i, raw in enumerate(raw_dist):
            refine_dist = raw[(lower_bound[i] <= raw) & (raw <= upper_bound[i])]
            mean[i] = np.mean(refine_dist, axis=-1)
        return mean

    def clustering_polygons(self, 
                            shorter_length: NDArray):
        """Clustering polygons through shorter length by KMeans.
        Returns:
            label(NDArray): (n, )
        """
        kmeans = KMeans(n_clusters=2, n_init=3).fit(shorter_length.reshape(-1, 1))
        
        cluster_center = kmeans.cluster_centers_
        label = kmeans.labels_
        # switching label when single entries are assigned smaller width
        if cluster_center[self.SINGLE_ENTRY] < cluster_center[self.DOUBLE_ENTRY]:
            label = np.where(label==self.SINGLE_ENTRY, self.DOUBLE_ENTRY, self.SINGLE_ENTRY)
            cluster_center = cluster_center[::-1]

        # if the center of two clusters is close enough, merge them!
        if cluster_center[self.SINGLE_ENTRY] < cluster_center[self.DOUBLE_ENTRY] * self.cluster_thresh:
            label = np.ones_like(label) * self.SINGLE_ENTRY
            cluster_center = (cluster_center[self.SINGLE_ENTRY],)
        else:
            # if keeping two cluster, using reading order to refine the result
            determined = np.zeros_like(label)

            # step 1: picking out double entry
            for i in range(len(label)-1):
                if determined[i]:
                    continue
                if label[i] == self.DOUBLE_ENTRY and label[i+1] == self.DOUBLE_ENTRY:
                    y_less = (np.min(self.polygons[i][:, 1]), np.min(self.polygons[i+1][:, 1]))
                    y_greater = (np.max(self.polygons[i][:, 1]), np.max(self.polygons[i+1][:, 1]))
                    _iou = max(0, (np.min(y_greater) - np.max(y_less)) / (np.max(y_greater) - np.min(y_less) + 1e-4))
                    if _iou > 0.5:
                        determined[i] = determined[i+1] = True

            # step 2: recompute cluster center
            if np.any(determined):
                # keeping the minimum for maximize the gap between clsuter center.
                cluster_center[self.DOUBLE_ENTRY] = min(np.mean(shorter_length[determined == True]),
                                                        cluster_center[self.DOUBLE_ENTRY])                     
            
            # step 3: transformation according to reading order
            # continuous text line in same column cannot have same label
            for i in range(len(label)-1):
                if determined[i] and determined[i+1]:
                    continue
                
                # step 1: find bottom line of line[i]
                _, bottom_0 = find_top_bottom(self.polygons[i])

                # step 2: find top line of line[i+1]
                top_1, _ = find_top_bottom(self.polygons[i+1])

                # step 3: projecting mid point of top and bottom line, prog = a·b/|b|
                top_proj = np.dot(np.mean(top_1, axis=0), self.direction)
                bot_proj = np.dot(np.mean(bottom_0, axis=0), self.direction)

                # step 4: computing normal vector of text direction
                normal_vector = np.array([self.direction[1], -self.direction[0]])

                # step 5: compute projection, avoid changing text blocks
                _iou = section_iou(np.array([np.dot(top_1[0], normal_vector), np.dot(top_1[1], normal_vector)]),
                            np.array([np.dot(bottom_0[0], normal_vector), np.dot(bottom_0[1], normal_vector)]))

                # step 6: if in the same column (not switching blocks)
                if bot_proj < top_proj and _iou > 0.1:
                    # step 7: promising continuous text line in same column cannot have same label
                    if determined[i]:
                        label[i + 1] = self.SINGLE_ENTRY if label[i]==self.DOUBLE_ENTRY else self.DOUBLE_ENTRY
                    elif determined[i+1]:
                        label[i] = self.SINGLE_ENTRY if label[i+1]==self.DOUBLE_ENTRY else self.DOUBLE_ENTRY
                    else:
                        if shorter_length[i] < shorter_length[i+1]:
                            label[i] = self.DOUBLE_ENTRY
                            label[i+1] = self.SINGLE_ENTRY
                        else:
                            label[i+1] = self.DOUBLE_ENTRY
                            label[i] = self.SINGLE_ENTRY
                    determined[i] = determined[i+1] = True
            
            # step 4: recompute cluster center
            # using the mean of determined one to get the new center
            if np.any((label == self.SINGLE_ENTRY) & (determined == True)):
                cluster_center[self.SINGLE_ENTRY] = max(np.mean(shorter_length[(label == self.SINGLE_ENTRY) & (determined == True)]),
                                                        cluster_center[self.SINGLE_ENTRY])
            if np.any((label == self.DOUBLE_ENTRY) & (determined == True)):
                cluster_center[self.DOUBLE_ENTRY] = min(np.mean(shorter_length[(label == self.DOUBLE_ENTRY) & (determined == True)]),
                                                        cluster_center[self.DOUBLE_ENTRY])

            # step 5: using recompute center to determine the remain
            for i in range(len(label)):
                if determined[i]:
                    continue
                label[i] = self.SINGLE_ENTRY if abs(shorter_length[i] - cluster_center[self.SINGLE_ENTRY]) \
                            < abs(shorter_length[i] - cluster_center[self.DOUBLE_ENTRY]) else self.DOUBLE_ENTRY
                determined[i] = True
                
        return label, cluster_center
    
    def compute_centerline(self,
                           points_a: NDArray, 
                           points_b: NDArray):
        """Computing centerline using sampled points and concatenate them nose
        to tail.
        Different shrink ratio may be used on single entries and double 
        entries.

        Args:
            points_a(NDArray): shape-like (n, p, 2).
            points_b(NDArray): shape-like (n, p, 2).
            points_a and points_b are return value of function `uniform_curve_sampling`.    
            r(float): shrink ratio of centerline, the area will shrink to 1/r respect to
                original polygon.

        Returns:
            center_a, center_b(NDArray): having same shape as points_a, points_b.
                center_a is the edge of centerline that near a, vice versa.
        """
        r = self.shrink_ratio[self.labels][:, np.newaxis, np.newaxis] # (n, 1, 1)
        theta = 0.5  -  1 / (2 * r) 

        # (n, p, 2)
        center_a = points_a * (1 - theta) + points_b * theta
        center_b = points_a * theta + points_b * (1 - theta)
        return np.concatenate([center_a, center_b[:, ::-1]], axis=1) # (n, 2p, 2)
        
    def preprocess(self):
        self.reorder_polygon()
        sample_a = np.zeros((self.num_instance, self.points, 2))
        sample_b = np.zeros((self.num_instance, self.points, 2))
        for i, poly in enumerate(self.polygons):
            curve_a, curve_b = poly[:len(poly)//2], poly[len(poly)//2:][::-1]
            sample_a[i] = uniform_curve_sampling(curve_a, self.points)
            sample_b[i] = uniform_curve_sampling(curve_b, self.points)
        shortside_length = self.determine_short_length(sample_a, sample_b)
        self.labels, self.cluster_center = self.clustering_polygons(shortside_length)
        self.preprocess = True

    def get_length(self, curve):
        segment_length = dist(curve, np.concatenate([curve[0:1], curve[:-1]], axis=0)) # (k, )
        accumulate_length = np.cumsum(segment_length) # (k, )
        curve_length = accumulate_length[-1]
        return curve_length
    
    def vertical_aspect_ratio(self):
        self.reorder_polygon()
        sample_a = np.zeros((self.num_instance, self.points, 2))
        sample_b = np.zeros((self.num_instance, self.points, 2))
        longside_length = np.zeros((self.num_instance, ))
        for i, poly in enumerate(self.polygons):
            curve_a, curve_b = poly[:len(poly)//2], poly[len(poly)//2:][::-1]
            sample_a[i] = uniform_curve_sampling(curve_a, self.points)
            sample_b[i] = uniform_curve_sampling(curve_b, self.points)
            longside_length[i] = (self.get_length(curve_a) + self.get_length(curve_b)) / 2
        shortside_length = self.determine_short_length(sample_a, sample_b)
        return longside_length / shortside_length
    
    def aspect_ratio(self):
        self.reorder_polygon()
        sample_a = np.zeros((self.num_instance, self.points, 2))
        sample_b = np.zeros((self.num_instance, self.points, 2))
        longside_length = np.zeros((self.num_instance, ))
        for i, poly in enumerate(self.polygons):
            curve_a, curve_b = poly[:len(poly)//2], poly[len(poly)//2:][::-1]
            sample_a[i] = uniform_curve_sampling(curve_a, self.points)
            sample_b[i] = uniform_curve_sampling(curve_b, self.points)
            longside_length[i] = (self.get_length(curve_a) + self.get_length(curve_b)) / 2
        shortside_length = self.determine_short_length(sample_a, sample_b)
        return np.where(longside_length>shortside_length, 
                        longside_length / shortside_length,
                        shortside_length / longside_length)
    
    def process(self):
        self.reorder_polygon()

        sample_a = np.zeros((self.num_instance, self.points, 2))
        sample_b = np.zeros((self.num_instance, self.points, 2))
        for i, poly in enumerate(self.polygons):
            curve_a, curve_b = poly[:len(poly)//2], poly[len(poly)//2:][::-1]
            sample_a[i] = uniform_curve_sampling(curve_a, self.points)
            sample_b[i] = uniform_curve_sampling(curve_b, self.points)

        if getattr(self, "preprocess", False):
            shortside_length = self.determine_short_length(sample_a, sample_b)
            self.labels, self.cluster_center = self.clustering_polygons(shortside_length)
            
        self.center_line = self.compute_centerline(sample_a, sample_b) # n, 2p, 2

    def generate_kernelmap(self):
        """Generate text center line map for single-line and double-line, respectively.
        """
        if not getattr(self, 'image_shape', False):
            warnings.warn('object don\'t have image_shape attr, cannot generate maps.')
            return None, None
        kernel_single = np.zeros(self.image_shape, dtype=np.uint8)
        kernel_double = np.zeros(self.image_shape, dtype=np.uint8)
        cv2.fillPoly(kernel_single, 
                     self.center_line[self.labels==self.SINGLE_ENTRY].astype(np.int32), 
                     255)
        cv2.fillPoly(kernel_double, 
                     self.center_line[self.labels==self.DOUBLE_ENTRY].astype(np.int32), 
                     255)
        return kernel_single, kernel_double
    
    def check_polygon_order(self):
        """Checking polygon order after reorder polygons. 
        """
        for poly in self.polygons:
            direction = compute_text_direction(poly)
            if np.dot(direction, self.direction) < np.cos(np.pi/6):
                print(f'may find fault direction in {getattr(self, "image_path", "<UNK_IMG>")}, \
                      direction difference: {np.dot(direction, self.direction)}')
                
    def output_json(self):
        '''Adding labels to annotations when preprocessing.  
        Returning a list of dict that will behave as `data['instances']`.     
        '''
        results = []
        assert self.texts is not None, 'text is none, json cannot be creates.'
        for poly, label, text in zip(self.polygons, self.labels, self.texts):
            results.append(dict(
                ignore=False,
                text=text,
                bbox_label=int(label),
                polygon=poly.reshape(-1).astype(int).tolist()
            ))
        return results