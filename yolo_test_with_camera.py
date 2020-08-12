#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 22:53:53 2020

@author: minghao
"""

import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from yolo_tiny import YOLO
from PIL import Image
import cv2
import numpy as np
from timeit import default_timer as timer
from coordinate_transformer import Camera
from Car_info import Car_info
from keras.models import load_model
from sklearn.utils.linear_assignment_ import linear_assignment
from Tracker import Tracker


# Data Unseen
video_input_path = 'test_video.mp4'

video_output_path = ""

is_to_detect_image = False


roi_expand_ratio = 1.2

# bbox=[top, left, bottom, right, class, scores]
def get_expanded_bbox(bbox):
    center_x = (bbox[1] + bbox[3]) / 2
    center_y = (bbox[0] + bbox[2]) / 2
    
    width = bbox[3] - bbox[1]
    height = bbox[2] - bbox[0]
    
    new_lu_x = int(max(center_x - width*roi_expand_ratio/2, 0))
    new_lu_y = int(max(center_y - height*roi_expand_ratio/2, 0))
    new_rd_x = int(min(center_x + width*roi_expand_ratio/2, 1280))
    new_rd_y = int(min(center_y + height*roi_expand_ratio/2, 1024))
    
    return (new_lu_y, new_lu_x, new_rd_y, new_rd_x, bbox[-2], bbox[-1])


def bbox_tlbr_ltrb(bbox):
    ret_bbox = bbox.copy()
    ret_bbox[1], ret_bbox[0] = ret_bbox[0], ret_bbox[1]
    ret_bbox[3], ret_bbox[2] = ret_bbox[2], ret_bbox[3]
    return ret_bbox


def transform_roi_bboxs(orig_startx, orig_starty, orig_w, orig_h, bboxs, resized_width, resized_height):
    ret_bboxs = []
    for bbox in bboxs:
        ret_bboxs.append( [int(orig_startx+bbox[0]*orig_w/resized_width), int(orig_starty+bbox[1]*orig_h/resized_height),
                           int(orig_startx+bbox[2]*orig_w/resized_width), int(orig_starty+bbox[3]*orig_h/resized_height), bbox[-2], bbox[-1]] )
    return ret_bboxs
        
    
def transform_roi_angle(xc, yc, orig_w, orig_h, predicted_x, predicted_y, camera):
    ### TODO: maybe can use mag as the confidence
    from_predicted_mag = (predicted_x**2 + predicted_y**2)**0.5
    predicted_x = predicted_x * orig_w
    predicted_y = predicted_y * orig_h
    predicted_mag = (predicted_x**2 + predicted_y**2)**0.5
    prediction = (predicted_x/predicted_mag, predicted_y/predicted_mag)
    
    startxw, startyw = camera.solve_wc_position(xc, yc, 0)
    endxw, endyw = camera.solve_wc_position(xc+prediction[0], yc+prediction[1], 0)
    
    prediction_deg = np.degrees( np.arctan2(endyw-startyw, endxw-startxw) ) % 360
    return prediction_deg
         
            
def calc_P_bbox2ID(bboxs, roi_features_bboxs, bbox_classes, roi_feature_classes):
    ### 0:B1, 1:B2, 2:R1, 3:R2
    to_add_inds = {'Red':[2,3],
                   'Blue':[0,1],
                   'Armor_Blue1':[0],
                   'Armor_Blue2':[1],
                   'Armor_Red1':[2],
                   'Armor_Red2':[3],
                   'Tail_Blue':[0,1],
                   'Tail_Red':[2,3]}
    weight_color = 0.3
    weight_armor = 0.6
    bbox2ID_matrix = np.zeros((len(bboxs),4),dtype=np.float32)
    scores_of_true_box = []
    for i, bbox in enumerate(bboxs):
        tmp_score_of_true_box = 0
        for class_ind, score in enumerate(bbox[-1]):
            tmp_score_of_true_box += score
            for j in to_add_inds.get(bbox_classes[class_ind], []):
                bbox2ID_matrix[i,j] += weight_color * score
        scores_of_true_box.append(tmp_score_of_true_box)
        
    for i in range(len(roi_features_bboxs)):
        for bbox in roi_features_bboxs[i]:
            for class_ind, score in enumerate(bbox[-1]):
                for j in to_add_inds.get(roi_feature_classes[class_ind], []):
                    bbox2ID_matrix[i,j] += weight_armor * score * scores_of_true_box[i]
    # return bbox2ID_matrix / (1e-10 + bbox2ID_matrix.sum(axis=1)[:,None])
    return bbox2ID_matrix


def project_roi_bboxs_to_wc(roi_features_bboxs, roi_feature_classes, camera):
    height_dict = {'Armor':car_info.armor_board_height,
                   'Tail':car_info.tail_lightbar_height,
                   'Wheel':car_info.wheel_height}
    ret_list = []
    for i_feature_bbox in roi_features_bboxs:
        i_class = i_feature_bbox[-2]
        i_score = i_feature_bbox[-1][roi_feature_classes.index(i_class)]
        if 'Armor' in i_class:
            key_for_height = 'Armor'
        elif 'Tail' in i_class:
            key_for_height = 'Tail'
        else:
            key_for_height = 'Wheel'
        xw, yw = camera.solve_wc_position( (i_feature_bbox[0]+i_feature_bbox[2])/2, (i_feature_bbox[1]+i_feature_bbox[3])/2, -height_dict[key_for_height] )
        ret_list.append((xw, yw, i_class, i_score))
    return ret_list
        

def calc_distance(point1, point2):
    return ( (point1[0]-point2[0])**2 + (point1[1]-point2[1])**2 )**0.5


def shift_point_position(from_x, from_y, longi_dis, lati_dis, angle):
    cos_val = np.cos(np.radians(angle))
    sin_val = np.sin(np.radians(angle))
    to_x = from_x + longi_dis*cos_val - lati_dis*sin_val
    to_y = from_y + longi_dis*sin_val + lati_dis*cos_val
    return to_x, to_y

 
def generate_potential_points(roi_locations_wc, angle, guess_center, r_thre):
    ret_list = []
    sign_for_armor = [(0,1), (1,0), (0,-1), (-1,0)]
    sign_for_wheel = [(1,1), (1,-1), (-1,1), (-1,-1)]
    for i_roi in roi_locations_wc:
        xw, yw, i_class, i_score = i_roi
        if 'Armor' in i_class:
            for i_sign in sign_for_armor:
                i_to_x, i_to_y = shift_point_position(xw, yw, i_sign[0]*car_info.armor_dis_longi, i_sign[1]*car_info.armor_dis_lat, angle)
                if calc_distance( (i_to_x, i_to_y), guess_center ) < r_thre:
                    ret_list.append( (i_to_x, i_to_y, i_score) )
        
        elif 'Wheel' in i_class:
            for i_sign in sign_for_wheel:
                i_to_x, i_to_y = shift_point_position(xw, yw, i_sign[0]*car_info.wheel_dis_longi, i_sign[1]*car_info.wheel_dis_lat, angle)
                if calc_distance( (i_to_x, i_to_y), guess_center ) < r_thre:
                    ret_list.append( (i_to_x, i_to_y, i_score) )
        else: ###TODO: sometimes the tail is partly detected
            pass
            # i_to_x, i_to_y = shift_point_position(xw, yw, car_info.tail_dis, 0, angle)
            # if calc_distance( (i_to_x, i_to_y), guess_center ) < r_thre:
            #     ret_list.append( (i_to_x, i_to_y, i_score) )
    return np.array(ret_list)


def cluster_center_point(potential_points, guess_center, r_thre):
    ### return x,y, position_score
    if len(potential_points) == 0:
        return (guess_center[0], guess_center[1], 0)
    elif len(potential_points) == 1:
        return potential_points[0]
    else:
        mean_x = (potential_points[:,0] * potential_points[:,2]).sum() / potential_points[:,2].sum()
        mean_y = (potential_points[:,1] * potential_points[:,2]).sum() / potential_points[:,2].sum()
        if calc_distance((mean_x, mean_y), guess_center) < 1e-3:
            return (mean_x, mean_y, potential_points[:,2].sum())
        dist_list = []
        for i_point in potential_points:
            if calc_distance( (i_point[0], i_point[1]), (mean_x, mean_y) ) < r_thre:
                dist_list.append(i_point)
        return cluster_center_point(np.array(dist_list), (mean_x, mean_y), r_thre)
                
    
        
def locate_one_roi(roi_features_bboxs, roi_feature_classes, angle, guess_center, camera):
    roi_locations_wc = project_roi_bboxs_to_wc(roi_features_bboxs, roi_feature_classes, camera)
    
    potential_points = generate_potential_points(roi_locations_wc, angle, guess_center, 0.3)
    clustered_center = cluster_center_point(potential_points, guess_center, 0.1)
    return clustered_center, angle ##TODO: try to refine the angle based on the detected roi features
    
    
def process_one_image(yolo_red_blue, yolo_roi_feature, angle_model, image, frame, camera):
    warped_image = camera.warp_image(frame)
    image, bboxs_list = yolo_red_blue.detect_image(image)
    ### bbox = (top, left, bottom, right, predicted_class, scores)
    
    stored_roi_images = []
    stored_roi_features_bboxs = []
    stored_angles = []
    stored_guess_centers = []
    
    for i, tmp_bbox in enumerate(bboxs_list):
        tmp_bbox = get_expanded_bbox(tmp_bbox)
        tmp_roi = frame[tmp_bbox[0]:tmp_bbox[2], tmp_bbox[1]:tmp_bbox[3]]
        tmp_roi_height = tmp_bbox[2] - tmp_bbox[0]
        tmp_roi_width = tmp_bbox[3] - tmp_bbox[1]
        
        tmp_resized_roi_for_feature = cv2.resize(tmp_roi, (288,288))
        tmp_resized_roi_for_angle = cv2.resize( cv2.cvtColor(tmp_roi, cv2.COLOR_BGR2GRAY), (96,96) ) / 255 - 0.5
        
        tmp_roi_image, tmp_roi_bbox_list = yolo_roi_feature.detect_image( Image.fromarray( cv2.cvtColor(tmp_resized_roi_for_feature, cv2.COLOR_BGR2RGB)) )
        tmp_roi_bboxs_at_image_coord = transform_roi_bboxs(tmp_bbox[1], tmp_bbox[0], tmp_roi_width, tmp_roi_height, tmp_roi_bbox_list, 288, 288) ##(l,t,r,b,c,s)
        # tmp_roi_image, tmp_roi_bbox_list = yolo_roi_feature.detect_image( Image.fromarray( cv2.cvtColor(tmp_roi, cv2.COLOR_BGR2RGB)) )
        # tmp_roi_bboxs_at_image_coord = transform_roi_bboxs(tmp_bbox[1], tmp_bbox[0], tmp_roi_width, tmp_roi_height, tmp_roi_bbox_list, tmp_roi_width, tmp_roi_height) ##(l,t,r,b,c,s)
        
        stored_roi_images.append(tmp_roi_image)
        stored_roi_features_bboxs.append(tmp_roi_bboxs_at_image_coord)
        
        predicted_x,  predicted_y = angle_model.predict(tmp_resized_roi_for_angle[None,:,:,None])[0]
        xp = tmp_bbox[1]*0.5 + tmp_bbox[3]*0.5
        yp = tmp_bbox[0]*0.33 + tmp_bbox[2]*0.67
        stored_angles.append( transform_roi_angle(xp, yp, tmp_roi_width, tmp_roi_height, predicted_x, predicted_y, camera) )
        stored_guess_centers.append( camera.solve_wc_position(xp, yp, 0) )
    
    
    bbox2ID_matrix = calc_P_bbox2ID(bboxs_list, stored_roi_features_bboxs, yolo_red_blue.class_names, yolo_roi_feature.class_names)
    # print(bbox2ID_matrix)
    
    pos_heading_info = []
    for i, i_roi_feature_bboxs in enumerate(stored_roi_features_bboxs):
        i_center, i_angle = locate_one_roi(i_roi_feature_bboxs, yolo_roi_feature.class_names, stored_angles[i], stored_guess_centers[i], camera)
        pos_heading_info.append( (i_center[0], i_center[1], i_angle) )
        
    for i,tmp_roi_image in enumerate(stored_roi_images):
        cv2.imshow('roi%d'%(bbox2ID_matrix[i].argmax()), cv2.cvtColor(np.asarray(tmp_roi_image), cv2.COLOR_RGB2BGR))
        
    result = np.asarray(image)
    result = cv2.cvtColor(result,cv2.COLOR_RGB2BGR)
    
    warped_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB)
    return result, warped_image, bbox2ID_matrix, np.array(pos_heading_info)
    
    
def detect_img(yolo_red_blue, yolo_roi_feature, angle_model):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            frame = np.array(image, dtype=np.uint8)
            camera = Camera(image_for_calibration=frame, n_keypoints=6)
            
            result, warped_image, bbox2ID_matrix, pos_heading_info = process_one_image(yolo_red_blue, yolo_roi_feature, angle_model, image, frame, camera)
            if len(pos_heading_info) > 0:
                matched_inds = linear_assignment(-bbox2ID_matrix)
                for i_detection, i_ID in matched_inds:
                    i_color = yolo_roi_feature.colors[i_ID+1]
                    pt1_x = pos_heading_info[i_detection][0]*100
                    pt1_y = 100*pos_heading_info[i_detection][1]
                    pt2_x, pt2_y = shift_point_position(pt1_x, pt1_y, 30, 0, pos_heading_info[i_detection][2])
                    print( (int(pt1_x),int(pt1_y)), (int(pt2_x),int(pt2_y)), i_color )
                    warped_image = cv2.arrowedLine(warped_image, (int(pt1_x),int(pt1_y)), (int(pt2_x),int(pt2_y)), i_color, 4)
            cv2.imshow('result', result)
            warped_image = cv2.cvtColor(warped_image, cv2.COLOR_RGB2BGR)
            cv2.imshow('warped_image', warped_image)
            cv2.waitKey(1)
    # yolo_red_blue.close_session()
    # yolo_roi_feature.close_session()
    # angle_model.close_session()


def detect_video(yolo_red_blue, yolo_roi_feature, angle_model, video_path, output_path=""):
    vid = cv2.VideoCapture(video_path)
    tracker = Tracker(max_age=4)
    return_value, frame = vid.read()
    camera = Camera(image_for_calibration=frame, n_keypoints=4)
    # camera = Camera(n_keypoints=6)
    
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    
    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    
    while True:
        return_value, frame = vid.read()
        image = Image.fromarray( cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) )
        
        ### bbox_list: bbox=[top, left, botton, right, class]
        result, warped_image, bbox2ID_matrix, pos_heading_info  = process_one_image(yolo_red_blue, yolo_roi_feature, angle_model, image, frame, camera)
        
        tracked_pos_heading_info = tracker.update(bbox2ID_matrix, pos_heading_info)
        
        for i, i_pos_heading in enumerate(tracked_pos_heading_info):
            if i_pos_heading is not None:
                print(i, i_pos_heading)
                i_color = yolo_roi_feature.colors[i+1]
                pt1_x = i_pos_heading[0]*100
                pt1_y = i_pos_heading[1]*100
                pt2_x, pt2_y = shift_point_position(pt1_x, pt1_y, 30, 0, i_pos_heading[2])
                # print( (int(pt1_x),int(pt1_y)), (int(pt2_x),int(pt2_y)), i_color )
                warped_image = cv2.arrowedLine(warped_image, (int(pt1_x),int(pt1_y)), (int(pt2_x),int(pt2_y)), i_color, 4)
                    
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.imshow("result", result)
        warped_image = cv2.cvtColor(warped_image, cv2.COLOR_RGB2BGR)
        cv2.imshow('warped_image', warped_image)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    # yolo_red_blue.close_session()
    # yolo_roi_feature.close_session()
    # angle_model.close_session()
    
    
    
if __name__ == '__main__':
    car_info = Car_info()
    """
    "model_path": 'model_data/RedBlueDetect.h5',
        "anchors_path": 'model_data/tiny_yolo_anchors.txt',
        "classes_path": 'model_data/robomaster_classes.txt',
        "score" : 0.25,
        "iou" : 0.4,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }
    """
    
    yolo_red_blue = YOLO(model_path='model_data/RedBlueDetect.h5',
                         anchors_path = 'model_data/yolo_anchors_robot.txt',
                         classes_path = 'model_data/robomaster_classes.txt',
                         score = 0.5, iou=0.2, model_image_size=(416,416), gpu_num=1)
    print('load Red Blue model')
    
    yolo_roi_feature = YOLO(model_path='model_data/ROIFeature.h5',
                         anchors_path = 'model_data/yolo_anchors_roi.txt',
                         classes_path = 'model_data/roi_classes.txt',
                         score = 0.3, iou=0.05, model_image_size=(288,288), gpu_num=1)
    print('load ROI Feature model')
    
    angle_model = load_model('model_data/AngleEstimator.h5')
    # INPUT SHAPE = (96,96,1)
    print('load Angle model')
    
    if is_to_detect_image:
        detect_img(yolo_red_blue, yolo_roi_feature, angle_model)
    else:
        detect_video(yolo_red_blue, yolo_roi_feature, angle_model, video_input_path, video_output_path)
        