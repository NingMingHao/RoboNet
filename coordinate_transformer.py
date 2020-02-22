#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 16:32:40 2020

@author: minghao
"""


## Reference link: https://www.cnblogs.com/aoru45/p/9781540.html
## Offical link: https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#solvepnp
import cv2
import numpy as np

class Camera(object):    
    def __init__(self, use_distortion=True, image_for_calibration=None, n_keypoints=6):
        """
        Parameters
        ----------
        use_distortion : boolen, optional
            whether or not to consider the distortion of the camera. The default is True.
        image_for_calibration : np.array, optional
            the image array for find n keypoints. The default is None. If it's None, then use the default image
        n_keypoints : TYPE, optional
            number of keypoints, the more the number is ,the more roboust the transformation matrix will be. The default is 4.
        """
        
        self.intrinsic_matrix = np.matrix([[841.841799249915, 0, 662.107871894200],
                                      [0, 840.413766437409, 518.864211233054],
                                      [0, 0, 1]])
        if use_distortion:
            self.distortion = np.array([-0.1103,0.0754,0,0], dtype=np.float32)
        else:
            self.distortion = np.array([0,0,0,0], dtype=np.float32)
        
        self.whole_size = (800,500)
        
        assert n_keypoints in [4,6]
        if n_keypoints==4:
            self.scene_points_at_world_coordinates_cm = np.array([(120,225,0), (700,100,0), (100,400,0), (700,500,0)], dtype=np.float32)
            self.have_stored_points = [(246,492), (794,233), (492,791), (1268,385)]
        else:
            self.scene_points_at_world_coordinates_cm = np.array([(120,225,0), (350,100,0), (700,100,0), 
                                                  (100,400,0), (450,400,0), (700,500,0)], dtype=np.float32)
            self.have_stored_points = [(248,494), (498,297), (792,232), (491,790), (997,429), (1269,384)]
            
        self.n_keypoints = n_keypoints
        self.scene_points_at_world_coordinates_m = self.scene_points_at_world_coordinates_cm / 100
        
        
        if image_for_calibration is None:
            ### Use pre image
            scene_image_path = '/Users/minghao/Documents/University/Robomaster/labelled_data/merged_data/JPEGImages/2020_1_4_17_17_26_805.mp4#t=85.833333.jpg'
            self.scene_image = cv2.imread(scene_image_path)
            self.show_scene_image = self.scene_image.copy()
            
        else:
            ### select new points
            self.scene_image = image_for_calibration.copy()
            self.show_scene_image = self.scene_image.copy()
            self.have_stored_points = []
            
            self.select_n_points() # will get n_keypoints points for calculating transformation matrix
        
        ### TODO: maybe I can refine it into subpixel accuracy
        self.scene_points_at_pixel_coordinates = np.array(self.have_stored_points, dtype=np.float32)
        
        self.calculate_transformation_matrix()
        
    def left_btn_callback(self, event,x,y,flags,param):
        if event==cv2.EVENT_LBUTTONDBLCLK:
            self.have_stored_points.append((x,y))
            self.draw_circles()
            
    def draw_circles(self):
        self.show_scene_image = self.scene_image.copy()
        for (x,y) in self.have_stored_points:
            cv2.circle(self.show_scene_image,(x,y),6,(255,0,0),-1)
            
    def select_n_points(self):
        cv2.imshow('scene_image', self.show_scene_image)
        cv2.waitKey(10)
        cv2.setMouseCallback('scene_image',self.left_btn_callback)
        
        print('select %d keypoints'%self.n_keypoints)
        confirm_flag = False
        self.draw_circles()
        while( (not confirm_flag) or (len(self.have_stored_points) < self.n_keypoints) ):
            cv2.imshow('scene_image', self.show_scene_image)
            tmp_key = cv2.waitKey(100)
            if tmp_key&0xFF==27: #esc
                print('Cancelled')
                break
            elif tmp_key&0xFF==8: #退格键
                if len(self.have_stored_points) > 0:
                    self.have_stored_points.pop(-1)
                    self.draw_circles()
                    
            elif tmp_key&0xFF==13: #Enter
                confirm_flag = True
            else:
                pass
        cv2.destroyWindow('scene_image')
        # cv2.destroyAllWindows()
        
        
    def warp_image(self, image):
        warped_image = cv2.warpPerspective(image, self.homo_matrix, dsize=self.whole_size)
        return warped_image
    
    def calculate_transformation_matrix(self):
        self.homo_matrix, _= cv2.findHomography(self.scene_points_at_pixel_coordinates, self.scene_points_at_world_coordinates_cm)
        self.warped_scene_image = self.warp_image(self.scene_image)
        
        retval, self.rvecs_from_wc_to_cc, self.tvecs_from_wc_to_cc = cv2.solvePnP(self.scene_points_at_world_coordinates_m, self.scene_points_at_pixel_coordinates, 
                                            self.intrinsic_matrix, self.distortion, flags=cv2.SOLVEPNP_ITERATIVE)
        
        
        ### Matrix from world coordinates to camera coordinates
        rotation_matrix_from_wc_to_cc = cv2.Rodrigues(self.rvecs_from_wc_to_cc)[0]
        matrix_from_wc_to_cc = np.zeros((4,4))
        matrix_from_wc_to_cc[:3,:3] = rotation_matrix_from_wc_to_cc
        matrix_from_wc_to_cc[:3,3] = self.tvecs_from_wc_to_cc[:,0]
        matrix_from_wc_to_cc[3,3] = 1
        self.matrix_from_wc_to_cc = np.matrix(matrix_from_wc_to_cc)
        
        
        ### Matrix from camera coordinates to world coordinates
        self.matrix_from_cc_to_wc = self.matrix_from_wc_to_cc.I
        self.tvecs_from_cc_to_wc = self.matrix_from_cc_to_wc[:3,3]
        self.rvecs_from_cc_to_wc = cv2.Rodrigues(self.matrix_from_cc_to_wc[:3,:3])[0]
        
        print('The position of the camera at the world coordinates is :\n', self.tvecs_from_cc_to_wc)
        h_of_camera = -self.tvecs_from_cc_to_wc[2,0]
        assert ( (h_of_camera>1.5) and (h_of_camera<2) )
        
        self.matrix_H34 = self.intrinsic_matrix * self.matrix_from_wc_to_cc[:3,:]


    ### Solve actual_position matrix
    def solve_wc_position(self, xp, yp, zw):
        tmp_a = xp*self.matrix_H34[2,0] - self.matrix_H34[0,0]
        tmp_b = xp*self.matrix_H34[2,1] - self.matrix_H34[0,1]
        tmp_c = yp*self.matrix_H34[2,0] - self.matrix_H34[1,0]
        tmp_d = yp*self.matrix_H34[2,1] - self.matrix_H34[1,1]
        
        tmp_e = (self.matrix_H34[0,2] - xp*self.matrix_H34[2,2])*zw + self.matrix_H34[0,3] - xp*self.matrix_H34[2,3]
        tmp_f = (self.matrix_H34[1,2] - yp*self.matrix_H34[2,2])*zw + self.matrix_H34[1,3] - yp*self.matrix_H34[2,3]
        matrix_A = np.matrix( [[tmp_a, tmp_b],
                               [tmp_c, tmp_d]])
        vector_b = np.matrix([[tmp_e],
                              [tmp_f]])
        
        solved_result = matrix_A.I * vector_b
        
        xw = solved_result[0,0]
        yw = solved_result[1,0]
        
        # print((xw, yw))
        return (xw, yw)
    
    def project_wc_to_cc(self, xw, yw, zw):
        vector_wc = np.matrix([xw, yw, zw, 1]).T
        vector_cc = self.matrix_H34 * vector_wc
        xp = vector_cc[0,0] / vector_cc[2,0]
        yp = vector_cc[1,0] / vector_cc[2,0]
        return (xp, yp)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    camera = Camera()
    fig_pixel_coordinate = plt.figure(1)
    scene_image_rgb = cv2.cvtColor(camera.scene_image, cv2.COLOR_BGR2RGB)
    plt.imshow(scene_image_rgb)
    
    fig_world_coorinate = plt.figure(2)
    warped_scene_image_rgb = cv2.cvtColor(camera.warped_scene_image, cv2.COLOR_BGR2RGB)
    plt.imshow(warped_scene_image_rgb)
    
    
    while True:
        tmp_input = fig_pixel_coordinate.ginput(1)
        if len(tmp_input):
            tmpx, tmpy = tmp_input[0]
            h_str = input('please input the height of the object, it should be negative: ')
            h_eval = eval(h_str)
            tmpxw, tmpyw = camera.solve_wc_position(tmpx, tmpy, h_eval)
            
            plt.figure(1)
            tmp_scene_image = cv2.circle(scene_image_rgb.copy(), (int(tmpx),int(tmpy)),6,(255,0,0),-1)
            plt.imshow(tmp_scene_image)
            plt.pause(0.2)
            
            plt.figure(2)
            tmp_warped_scene_image = cv2.circle(warped_scene_image_rgb.copy(), (int(tmpxw*100),int(tmpyw*100)),6,(255,0,0),-1)
            plt.imshow(tmp_warped_scene_image)
            plt.pause(0.2)
        
        
    
    
    
    