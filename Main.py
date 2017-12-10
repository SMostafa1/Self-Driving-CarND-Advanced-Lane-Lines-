import Utilities
import os , sys
import cv2
from moviepy.editor import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import glob
##############################Compute the camera calibration matrix and distortion coefficients given a set of chessboard images###########################
# mtx,dist,newcameramtx = Utilities.CalibrateCamera()
################################################Process Video Images#########################################
# ####1-loop on all images
# Inputpath = "test_images"
# OutputPath = "C:/Self Driving/CarND-Advanced-Lane-Lines/"
# dirs = os.listdir(Inputpath)
# UnDistortionArr =[]
# x=0
# for file in dirs:
#   if file.startswith("frame"):
#     x = x+1
#     file = Inputpath + '/' +file
#     # print (file)
#     img = mpimg.imread(file)
#     undistortionImg = Utilities.Undistortion(img,mtx,dist,newcameramtx)
#     Binaryimg = Utilities.ColorandGradient(undistortionImg)
#     binary_warped, unwarped, m, m_inv = Utilities.corners_unwarp(9, 6, Binaryimg)
#     #################################################Find lines#########################################################
#     left_fit, right_fit, left_lane_inds, right_lane_inds, nonzerox, nonzeroy, out_img, left_fitx, right_fitx, margin, ploty = Utilities.Findlines(binary_warped)
#     left_fit_new, right_fit_new, left_lane_inds, right_lane_inds = Utilities.polyfit_using_prev_fit(binary_warped,left_fit,right_fit)
#     ##################################################Measuring Curveature##############################################
#     left_curverad, right_curverad, center_dist = Utilities.calc_curv_rad_and_center_dist(binary_warped,left_fit_new,right_fit_new,left_lane_inds,right_lane_inds)
#     #################################################Draw lane##########################################################
#     results = Utilities.draw_lane(img,binary_warped,left_fitx,right_fitx,m_inv,ploty)
#     mpimg.imsave(OutputPath+"frame"+str(x),results)
####################################################Below code to convert video into images###########################################################
# vidcap = cv2.VideoCapture('project_video.mp4')
# success,image = vidcap.read()
# count = 0
# success = True
# while success:
#   success,image = vidcap.read()
#   print ('Read a new frame: ', success)
#   cv2.imwrite("output_images/frame%d.jpg" % count, image)     # save frame as JPEG file
#   count += 1
###########################################################################################################################################
mtx,dist,newcameramtx = Utilities.CalibrateCamera()

def Process (image):
    # img = mpimg.imread(image)
    undistortionImg = Utilities.Undistortion(image, mtx, dist, newcameramtx)
    Binaryimg = Utilities.ColorandGradient(undistortionImg)
    binary_warped, unwarped, m, m_inv = Utilities.corners_unwarp(9, 6, Binaryimg)
    #################################################Find lines#########################################################
    left_fit, right_fit, left_lane_inds, right_lane_inds, nonzerox, nonzeroy, out_img, left_fitx, right_fitx, margin, ploty = Utilities.Findlines(
        binary_warped)
    left_fit_new, right_fit_new, left_lane_inds, right_lane_inds = Utilities.polyfit_using_prev_fit(binary_warped,
                                                                                                    left_fit, right_fit)
    ##################################################Measuring Curveature##############################################
    left_curverad, right_curverad, center_dist = Utilities.calc_curv_rad_and_center_dist(binary_warped, left_fit_new,
                                                                                         right_fit_new, left_lane_inds,
                                                                                         right_lane_inds)
    #################################################Draw lane##########################################################
    results = Utilities.draw_lane(image, binary_warped, left_fitx, right_fitx, m_inv, ploty)
    return results

#################Detect lane in video##################################
Video_Output_path = "output_video"
if not os.path.isdir(Video_Output_path):
    os.mkdir(Video_Output_path)

input_path = "project_video.mp4"
print(input_path)
Video_Output_path = "output_video/" + "output_video.mp4"
clip1 = VideoFileClip(input_path)
white_clip = clip1.fl_image(Process)
white_clip.write_videofile(Video_Output_path, audio=False)
