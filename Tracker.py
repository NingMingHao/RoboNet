#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 15:25:52 2020

@author: minghao
"""

from __future__ import print_function
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
from filterpy.kalman import KalmanFilter
import cv2
from Car_info import Car_info

car_info = Car_info()
def iou_rotate_calculate(box1, box2):
    """
    compute IOU between two bboxes in the form [center_x, center_y, w, h, angle(degrees)]

    """
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    r1 = ((box1[0], box1[1]), (box1[2], box1[3]), box1[4])
    r2 = ((box2[0], box2[1]), (box2[2], box2[3]), box2[4])

    int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
    if int_pts is not None:
        order_pts = cv2.convexHull(int_pts, returnPoints=True)

        int_area = cv2.contourArea(order_pts)

        iou = int_area / (area1 + area2 - int_area)
    else:
        iou = 0.0
    return iou


def convert_pos_angle_to_bbox(pos_angle):
  """
  Takes a pos_angle in the form [center_x, center_y, angle) and returns it in the form
    [center_x, center_y, w, h, angle(degrees)]
  """
  w = car_info.width
  h = car_info.length
  return np.array([pos_angle[0], pos_angle[1], w, h, pos_angle[2]])
    


class KalmanBoxTracker(object):
    """
    This class represents the internel state of individual tracked objects observed as bbox.
    bbox:[x, y, angle],     x,y:meter;  angle: degree
    """
    count = 0
    def __init__(self, car_id=0):
        """
        Initialises a tracker using initial bounding box.
        """
        #define constant velocity model
        self.id = car_id
        self.dt = 1
        self.scale_for_angle = 0.01
        self.kf = KalmanFilter(dim_x=6, dim_z=3)
        self.kf.F = np.array([[1,0,0,self.dt,0,0],[0,1,0,0,self.dt,0],[0,0,1,0,0,self.dt],[0,0,0,1,0,0],  [0,0,0,0,1,0],[0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0]])
        
        self.kf.P[3:,3:] *= 10. #give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10
        
        self.kf.Q = np.array( [ [0,0,0,0.1,0,0], [0,0,0,0,0.1,0], [0,0,0,0,0,0.1], [0.1,0,0,0.1,0,0], [0,0.1,0,0,0.1,0], [0,0,0.1,0,0,0.1] ] )
        
        self.have_initilized = False


    def init_tracker(self, bbox):
        self.have_initilized = True
        self.kf.x[:3] = bbox[:,None]
        self.kf.x[3:] = 0
        self.kf.x[2] *= self.scale_for_angle
        self.time_since_update = 0
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self,bbox):
        """
        Updates the state vector with observed bbox.
        """
        if not self.have_initilized:
            self.init_tracker(bbox)
        else:
            self.time_since_update = 0
            self.history = []
            self.hits += 1
            self.hit_streak += 1
            
            current_angle = self.get_needed_state()[2]
            new_angle = bbox[2]
            if (new_angle-current_angle) >= 0:
                new_angle -= ( (new_angle-current_angle) // 180 ) * 360
            else:
                new_angle += ( (current_angle-new_angle) // 180 ) * 360
            bbox[2] = new_angle*self.scale_for_angle
            self.kf.update(bbox[:,None])
            self.kf.x[2,0] %= (360*self.scale_for_angle)

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if self.have_initilized:
            self.kf.predict()
            self.kf.x[2,0] %= (360*self.scale_for_angle)
            self.age += 1
            if(self.time_since_update>0):
              self.hit_streak = 0
            self.time_since_update += 1
            self.history.append( self.get_needed_state() )
            return self.history[-1]
        else:
            return None

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        if self.have_initilized:
            return self.kf.x.copy()
        else:
            return None
    
    def get_needed_state(self):
        if self.have_initilized:
            ret = self.kf.x.copy()[:3,0]
            ret[2] /= self.scale_for_angle
            return ret
        else:
            return None


def associate_detections_to_trackers(bbox2ID_matrix, pos_heading_info, trackers, iou_threshold = 0.35):
    for d,det in enumerate(pos_heading_info):
        for t,trk in enumerate(trackers):
            if trk is not None:
                det_box = convert_pos_angle_to_bbox(det)
                trk_box = convert_pos_angle_to_bbox(trk)
                bbox2ID_matrix[d,t] += iou_rotate_calculate(det_box,trk_box)
              
    matched_indices = linear_assignment(-bbox2ID_matrix)
    
    unmatched_detections = []
    for d,det in enumerate(pos_heading_info):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t,trk in enumerate(trackers):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)
    
    #filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if(bbox2ID_matrix[m[0],m[1]]<iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)
    
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)



class Tracker(object):
  def __init__(self,max_age=10,min_hits=1):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.trackers = [KalmanBoxTracker(car_id=0), KalmanBoxTracker(car_id=1), KalmanBoxTracker(car_id=2), KalmanBoxTracker(car_id=3)]
    self.frame_count = 0

  def update(self,bbox2ID_matrix, pos_heading_info):
    self.frame_count += 1
    #get predicted locations from existing trackers.
    trks = []
    for t in range(len(self.trackers)):
        pos = self.trackers[t].predict()
        trks.append(pos)
      
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(bbox2ID_matrix, pos_heading_info, trks)
    
    ret = []
    #update matched trackers with assigned detections
    for t,trk in enumerate(self.trackers):
        if(t not in unmatched_trks):
            d = matched[np.where(matched[:,1]==t)[0][0],0]
            trk.update(pos_heading_info[d])
          
    for trk in self.trackers:
        if trk.have_initilized:
            if(trk.time_since_update > self.max_age):
                trk.have_initilized = False
        d = trk.get_needed_state()
        ret.append(d)
        # if((trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
        #     ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
        #remove dead tracklet
    return ret
    

if __name__ == '__main__':
    tracker = Tracker() #create instance of the SORT tracker



