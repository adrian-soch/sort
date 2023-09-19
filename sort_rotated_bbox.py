"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016 Alex Bewley alex@dynamicdetection.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import numpy as np
from filterpy.kalman import KalmanFilter
from shapely.geometry import Polygon
from shapely import affinity, vectorized

import timeit


def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def iou_rotated_bbox(poly1, poly2):
    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return intersection / union


def state2polygon(state) -> Polygon:
    ratio = np.maximum(0.0, state[3])
    half_width = np.sqrt(state[2]*ratio)/2.0
    half_height = (half_width*2 / state[3])/2.0
    center_x = state[0]
    center_y = state[1]
    angle = state[4]

    # Create a rectangle polygon centered at the origin
    rect = Polygon([(-half_width, -half_height), (-half_width, half_height),
                   (half_width, half_height), (half_width, -half_height)])

    # Rotate the polygon by the angle
    rotated_rect = affinity.rotate(rect, angle, use_radians=True)

    # Translate the polygon to the center coordinates
    return affinity.translate(rotated_rect, center_x, center_y)


def iou_rotated_bbox_matrix(bb_test_batch, bb_gt_batch):
    # Assume that bb_test_batch and bb_gt_batch are numpy arrays of shape (M, 5) and (N, 5)
    # where M and N are the batch sizes and 5 is the number of state parameters
    # Return a numpy array of shape (M, N) containing the IoU values for each pair of bounding boxes

    # Convert the state parameters to polygon objects using vectorized operations
    ratio = np.maximum(0.0, bb_test_batch[:, 3])
    width = np.sqrt(bb_test_batch[:, 2] * ratio)
    height = width / bb_test_batch[:, 3]
    center_x = bb_test_batch[:, 0]
    center_y = bb_test_batch[:, 1]
    angle = bb_test_batch[:, 4]

    # Create a rectangle polygon centered at the origin
    rect = Polygon([(-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5), (0.5, -0.5)])

    # Rotate and translate the rectangle polygon for each bounding box
    poly1 = [affinity.translate(affinity.rotate(
        rect, a, use_radians=True), x, y) for x, y, a in zip(center_x, center_y, angle)]

    # Repeat the same process for the ground truth bounding boxes
    ratio = np.maximum(0.0, bb_gt_batch[:, 3])
    width = np.sqrt(bb_gt_batch[:, 2] * ratio)
    height = width / bb_gt_batch[:, 3]
    center_x = bb_gt_batch[:, 0]
    center_y = bb_gt_batch[:, 1]
    angle = bb_gt_batch[:, 4]

    rect = Polygon([(-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5), (0.5, -0.5)])
    poly2 = [affinity.translate(affinity.rotate(
        rect, a, use_radians=True), x, y) for x, y, a in zip(center_x, center_y, angle)]

    # Initialize an empty matrix to store the IoU values
    iou_matrix = np.zeros((len(poly1), len(poly2)))

    # Loop over each pair of polygons and compute the intersection and union areas
    for i in range(len(poly1)):
        for j in range(len(poly2)):
            intersection = poly1[i].intersection(poly2[j]).area
            union = poly1[i].union(poly2[j]).area
            iou_matrix[i][j] = intersection / union

    # Return the IoU matrix
    return iou_matrix


def iou_rotated_bbox_matrix_optimized(bb_test_batch, bb_gt_batch):
    # Assume that bb_test_batch and bb_gt_batch are numpy arrays of shape (M, 5) and (N, 5)
    # where M and N are the batch sizes and 5 is the number of state parameters
    # Return a numpy array of shape (M, N) containing the IoU values for each pair of bounding boxes

    # Convert the state parameters to polygon objects using vectorized operations
    ratio = np.maximum(0.0, bb_test_batch[:, 3])
    width = np.sqrt(bb_test_batch[:, 2] * ratio)
    height = width / bb_test_batch[:, 3]
    center_x = bb_test_batch[:, 0]
    center_y = bb_test_batch[:, 1]
    angle = bb_test_batch[:, 4]

    # Create a rectangle polygon centered at the origin
    rect = Polygon([(-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5), (0.5, -0.5)])

    # Rotate and translate the rectangle polygon for each bounding box
    # Use list comprehensions instead of loops to create a list of polygons
    poly1 = [affinity.translate(affinity.rotate(
        rect, a, use_radians=True), x, y) for x, y, a in zip(center_x, center_y, angle)]

    # Repeat the same process for the ground truth bounding boxes
    ratio = np.maximum(0.0, bb_gt_batch[:, 3])
    width = np.sqrt(bb_gt_batch[:, 2] * ratio)
    height = width / bb_gt_batch[:, 3]
    center_x = bb_gt_batch[:, 0]
    center_y = bb_gt_batch[:, 1]
    angle = bb_gt_batch[:, 4]

    rect = Polygon([(-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5), (0.5, -0.5)])
    poly2 = [affinity.translate(affinity.rotate(
        rect, a, use_radians=True), x, y) for x, y, a in zip(center_x, center_y, angle)]

    # Initialize an empty matrix to store the IoU values
    iou_matrix = np.zeros((len(poly1), len(poly2)))

    # Loop over each pair of polygons and compute the intersection and union areas
    # Use numpy vectorization instead of nested loops to calculate the areas
    # Use shapely.vectorized functions instead of shapely.geometry functions to operate on arrays of polygons
    intersection = vectorized.intersects(poly1, poly2).astype(
        float) * vectorized.intersection_area(poly1, poly2)
    union = vectorized.union_area(poly1, poly2)

    # Avoid division by zero by adding a small epsilon value to the denominator
    epsilon = 1e-10
    iou_matrix = intersection / (union + epsilon)

    # Return the IoU matrix
    return iou_matrix


class KalmanBoxTracker(object):
    """
    This class represents the internel state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, z):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=9, dim_z=5)

        self.kf.F = np.array([[1, 0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0, 1],
                              [0, 0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 1]])

        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[5:, 5:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[5:, 5:] *= 0.01

        self.kf.x[:5] = np.expand_dims(z, axis=1)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, z):
        """
        Updates the state vector with observed z.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(z)

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        # If area + velocity of the area is negative, make the velovity negative
        # this needs to be done for any state that must be != 0
        if ((self.kf.x[7] + self.kf.x[2]) <= 0):
            self.kf.x[7] *= 0.0
        self.kf.predict()
        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.kf.x[:5])
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.kf.x[:5]


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    det_poly_array = []
    for det in detections:
        det_poly_array.append(state2polygon(det))

    trk_poly_array = []
    for trk in trackers:
        trk_poly_array.append(state2polygon(trk))

    for d in range(len(detections)):
        for t in range(len(trackers)):
            iou_matrix[d, t] = iou_rotated_bbox(det_poly_array[d], trk_poly_array[t])

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if (iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.01):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.dim_z = 5
        self.dim_trk_out = self.dim_z + 1  # some of the states + trk ID

    def update(self, dets):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()
            trk[:] = [pos[0], pos[1], pos[2], pos[3], pos[4]]
            if (np.any(np.isnan(pos))):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trks, self.iou_threshold)

        # update matched trackers with assigned detections
        for t, trk in enumerate(self.trackers):
            if (t not in unmatched_trks):
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                trk.update(dets[d, :][0])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            if ((trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
                # +1 as MOT benchmark requires positive
                ret.append(np.append(d, trk.id + 1).reshape(1, -1))
            i -= 1
            # remove dead tracklet
            if (trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if (len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 6))
