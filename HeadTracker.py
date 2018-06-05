import numpy as np
from collections import deque
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cdist
import cv2
import warnings
from core import plot_mask

import time

# params
DIST_RESTRICT = 100
MID_FIELD_MARGIN = 100
ENABLE_PREDICT = False
def _one_to_one_mapping(dist_matrix):
    """
    :param dist_matrix:
    :return:
    returns an array of shape (n_true_heads,2), stores the minimal distances and corresponding index in each column
    """
    x, y = dist_matrix.shape
    ret = np.ones((x, 2), dtype=np.float) * np.inf
    mask = np.ones_like(dist_matrix)
    while np.max(mask) > 0:
        idx, idy = np.unravel_index(np.argmin(np.where(mask == 1, dist_matrix, np.inf), axis=None),
                                    dist_matrix.shape)
        mask[:, idy] = 0
        ret[idx, 0] = dist_matrix[idx, idy]
        ret[idx, 1] = int(idy)
        mask[idx, :] = 0
    return ret


class HeadTracker(object):
    def __init__(self, threshold=DIST_RESTRICT):
        self.track_threshold = threshold
        self.head_tracks = list()
        self.inn = 0
        self.out = 0


    def update(self, frame, proposal_mask):
        if len(self.head_tracks) == 0:
            for i,j in np.argwhere(proposal_mask == 1):
                ht = self.HeadTrack(frame, (i,j))
                self.head_tracks.append(ht)
            return
        head_track_list = np.zeros(len(self.head_tracks), dtype=np.bool)
        if np.any(proposal_mask):
            dist_matrix = cdist(np.argwhere(proposal_mask == 1), np.array(list(map(lambda x: x.lastPoint(), self.head_tracks))))
            dist_matrix = _one_to_one_mapping(dist_matrix)
            for i in range(len(np.argwhere(proposal_mask == 1))):
                if dist_matrix[i, 0] < self.track_threshold:
                    head_track_list[int(dist_matrix[i, 1])] = 1
                    self.head_tracks[int(dist_matrix[i, 1])].update(frame, tuple(np.argwhere(proposal_mask == 1)[i]))
                else:
                    point = tuple(np.argwhere(proposal_mask == 1)[i])
                    if MID_FIELD_MARGIN < point[0] < 240 - MID_FIELD_MARGIN:
                       warnings.warn('Trace start from midfield. Abondoned!')
                       continue
                    ht = self.HeadTrack(frame, point)
                    self.head_tracks.append(ht)

        for i in range(len(head_track_list)-1, -1, -1):
            if head_track_list[i]==0:
                ret = self.head_tracks[i].update(frame)
                if ret:
                    if self.head_tracks[i].is_valid():
                        if MID_FIELD_MARGIN < self.head_tracks[i].path[-1][0] < 240 - MID_FIELD_MARGIN:
                            warnings.warn('Trace vanished in midfield. Abondoned!')
                        else:
                            if min(self.head_tracks[i].path[0][0],self.head_tracks[i].path[-1][0]) > MID_FIELD_MARGIN or \
                                max(self.head_tracks[i].path[0][0], self.head_tracks[i].path[-1][0]) < 240 -MID_FIELD_MARGIN :
                               warnings.warn('Trace started and vanished in same side. Abondoned!')
                            else:
                                if self.head_tracks[i].path[0][0] - self.head_tracks[i].path[-1][0] > 0:
                                    self.inn += 1
                                else:
                                    self.out += 1
                    del self.head_tracks[i]


    def get_valid_tracks(self):
        return list(filter(lambda x: x.is_valid(), self.head_tracks))

    class HeadTrack(object):
        def __init__(self, frame, point, intrusion=2, omission=6):
            self.path = deque([], maxlen=600)
            self.omission = omission
            self.intrusion = intrusion
            self.omission_cnt = 0
            self.intrusion_cnt = 0
            self.valid = False
            if ENABLE_PREDICT:
                self.tracker = cv2.TrackerMedianFlow_create()
                bbox = point[1] - 35, point[0] - 35, 70, 70
                self.tracker.init(frame, bbox)
            self.update(frame, point)



        def lastPoint(self):
            """
            calculate the distance of a given point and the last point of an existing track, for evaluating
            which track the new point should belong to
            :param point: a tuple that contains the x and y coordinates of a new point
            :return:  a float that indicates the Euclid distance
            """
            if len(self.path) < 1:
                raise Exception('Track not initialized!')
            else:
                if self.path[-1] is not None:
                    last_point = self.path[-1]
                else:
                    raise RuntimeError('last point is None')
                return last_point

        def tracker_predict(self, frame, point):
            if point:
                bbox = point[1] - 35, point[0] - 35, 70, 70
                self.tracker.init(frame, bbox)
            else:
                if MID_FIELD_MARGIN < self.path[-1][0] < 240 - MID_FIELD_MARGIN:
                    ok, bbox = self.tracker.update(frame)
                    if ok:
                        point = (int(bbox[0] + bbox[2] // 2), int(bbox[1] + bbox[3] // 2))
                        if euclidean(point,self.lastPoint()) > DIST_RESTRICT:
                            point = None
                else:
                    self.omission_cnt = self.omission
            return point

        def update(self, frame, point=None):
            """
            update a track and return true if it dies
            :return: boolean True if the track should be destroyed
            """
            if self.valid and ENABLE_PREDICT:
                point = self.tracker_predict(frame, point)
            if point:
                self.path.append(point)
                if self.valid:
                    self.omission_cnt = 0
                else:
                    self.intrusion_cnt += 1
                    if self.intrusion_cnt >= self.intrusion:
                        self.valid = True
                return False
            else:
                self.omission_cnt += 1
                if self.omission_cnt >= self.omission:
                    return True
                if not self.valid:
                    self.intrusion_cnt = 0
                    return False

        def is_valid(self):
            return self.valid

        def __str__(self):
            return '.'.join(map(lambda x: '(%s,%s)' % x, self.path))