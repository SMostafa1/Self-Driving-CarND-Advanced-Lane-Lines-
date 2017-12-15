import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from scipy.misc import toimage
#======================================================================================================================================================#
def CalibrateCamera():
    ##REF:http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    images = glob.glob('camera_cal/calibration*.jpg')
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (9, 6), corners2, ret)
            # cv2.imshow('img',img)
            # cv2.waitKey(500)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
    #rework:To undistort the image after camera calibration
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    # f.subplots_adjust(hspace=.2, wspace=.05)
    # ax1.imshow(img)
    # ax1.set_title('Original Image', fontsize=30)
    # ax2.imshow(dst)
    # ax2.set_title('Undistorted Image', fontsize=30)
    # plt.show()
    # cv2.destroyAllWindows()
    # print(mtx)
    return mtx,dist,newcameramtx
#======================================================================================================================================================#

#======================================================================================================================================================#
def Undistortion(img,mtx,dist,newcameramtx):
    print(mtx,dist)
    undstImg = np.matrix([])
    undstImg = cv2.undistort(img, mtx, dist, undstImg, newcameramtx)
    # Visualize undistortion
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    # f.subplots_adjust(hspace=.2, wspace=.05)
    # ax1.imshow(img)
    # ax1.set_title('Original Image', fontsize=30)
    # ax2.imshow(undstImg)
    # ax2.set_title('Undistorted Image', fontsize=30)
    # print('...................')
    # plt.show()
    return undstImg
#======================================================================================================================================================#


#======================================================================================================================================================#
#===================================================UnUsed Functions=================================================================#
#======================================================================================================================================================#

# # Define a function to return the magnitude of the gradient
# # for a given sobel kernel size and threshold values
# def mag_thresh(img, sobel_kernel=25, mag_thresh=(25, 255)):
#     # Convert to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     # Take both Sobel x and y gradients
#     sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
#     sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
#     # Calculate the gradient magnitude
#     gradmag = np.sqrt(sobelx**2 + sobely**2)
#     # Rescale to 8 bit
#     scale_factor = np.max(gradmag)/255
#     gradmag = (gradmag/scale_factor).astype(np.uint8)
#     # Create a binary image of ones where threshold is met, zeros otherwise
#     binary_output = np.zeros_like(gradmag)
#     binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
#
#     # Return the binary image
#     return binary_output
#
# def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
#     # Grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     # Calculate the x and y gradients
#     sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
#     sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
#     # Take the absolute value of the gradient direction,
#     # apply a threshold, and create a binary image result
#     absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
#     binary_output =  np.zeros_like(absgraddir)
#     binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
#
#     # Return the binary image
#     return binary_output
#
# # Define a function that takes an image, gradient orientation,
# # and threshold min / max values.
# def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
#     # Convert to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     # Apply x or y gradient with the OpenCV Sobel() function
#     # and take the absolute value
#     if orient == 'x':
#         abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
#     if orient == 'y':
#         abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
#     # Rescale back to 8 bit integer
#     scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
#     # Create a copy and apply the threshold
#     binary_output = np.zeros_like(scaled_sobel)
#     # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
#     binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
#
#     # Return the result
#     return binary_output
#
#
# def combinethresholds(image):
#     grad_binary_x = abs_sobel_thresh(image, orient='x', thresh_min=20, thresh_max=100)
#     grad_binary_y = abs_sobel_thresh(image, orient='y', thresh_min=20, thresh_max=100)
#     mag_binary = mag_thresh(image, sobel_kernel=3, mag_thresh=(30, 100))
#     dir_binary = dir_threshold(image,sobel_kernel=3, thresh=(0, np.pi/2)) #sobel_kernel=15, thresh=(0.7, 1.3))
#     combined = np.zeros_like(mag_binary)
#     combined[((grad_binary_x == 1) & (grad_binary_y == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
#     combined[((grad_binary_x == 1) & (grad_binary_y == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
#     return combined
#
# # Define a function that thresholds the B-channel of LAB
# # Use exclusive lower bound (>) and inclusive upper (<=), OR the results of the thresholds (B channel should capture
# # yellows)
# def lab_bthresh(img, thresh=(190,255)):
#     # 1) Convert to LAB color space
#     lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
#     lab_b = lab[:,:,2]
#     # don't normalize if there are no yellows in the image
#     if np.max(lab_b) > 175:
#         lab_b = lab_b*(255/np.max(lab_b))
#     # 2) Apply a threshold to the L channel
#     binary_output = np.zeros_like(lab_b)
#     binary_output[((lab_b > thresh[0]) & (lab_b <= thresh[1]))] = 1
#     # 3) Return a binary image of threshold result
#     return binary_output
#
# # Define a function that thresholds the S-channel of HLS
# def hls_select(img, thresh=(125, 255)):
#     hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
#     s_channel = hls[:,:,2]
#     binary_output = np.zeros_like(s_channel)
#     binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
#     return binary_output
#
# # hls_binary = hls_select(image, thresh=(90, 255))

#======================================================================================================================================================#
def ColorandGradient(img,HLSthreshold=(0,255),threshold=(0,255)):
    ###############################################commmented due to rework############################################
    # hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    # l_channel = hls[:, :, 1]
    # s_channel = hls[:, :, 2]
    # # Sobel x
    # sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    # abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    # scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    #
    # sx_thresh = (20, 100)
    # s_thresh = (90, 255)
    # # Threshold x gradient
    # sxbinary = np.zeros_like(scaled_sobel)
    # sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    #
    # # Threshold color channel
    # s_binary = np.zeros_like(s_channel)
    # s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # # Stack each channel
    # # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # # be beneficial to replace this channel with something else.
    #
    # color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    #
    # # Combine the two binary thresholds
    # color_binary = np.zeros_like(sxbinary)
    # color_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    ###############################################commmented due to rework############################################
    ########################Rework#################
    s_channel = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)[:, :, 2]

    l_channel = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)[:, :, 0]

    b_channel = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)[:, :, 2]

    # Threshold color channel
    s_thresh_min = 180
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

    b_thresh_min = 155
    b_thresh_max = 200
    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel >= b_thresh_min) & (b_channel <= b_thresh_max)] = 1

    l_thresh_min = 225
    l_thresh_max = 255
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh_min) & (l_channel <= l_thresh_max)] = 1

    # color_binary = np.dstack((u_binary, s_binary, l_binary))

    combined_binary = np.zeros_like(s_binary)
    combined_binary[(l_binary == 1) | (b_binary == 1)] = 1

    ##############################################
    return combined_binary
#======================================================================================================================================================#

#======================================================================================================================================================#
def corners_unwarp(nx, ny,binaryimg):
    h, w = binaryimg.shape[:2]
    # src = np.float32(
    #     [[200, 720],
    #      [1100, 720],
    #      [595, 450],
    #      [685, 450]])
    # dst = np.float32(
    #     [[300, 720],
    #      [980, 720],
    #      [300, 0],
    #      [980, 0]])
    #Rework:
    src = np.float32([[545, 460],
                      [735, 460],
                      [1280, 700],
                      [0, 700]])

    dst = np.float32([[0, 0],
                      [1280, 0],
                      [1280, 720],
                      [0, 720]])
    # src = np.float32([[w // 2 - 76, h * .625], [w // 2 + 76, h * .625], [-100, h], [w + 100, h]])
    # dst = np.float32([[100, 0], [w - 100, 0], [100, h], [w - 100, h]])
    img_size = (binaryimg.shape[1], binaryimg.shape[0])
    m = cv2.getPerspectiveTransform(src, dst)
    m_inv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(binaryimg, m, img_size, flags=cv2.INTER_LINEAR)
    unwarped = cv2.warpPerspective(warped, m_inv, (warped.shape[1], warped.shape[0]), flags=cv2.INTER_LINEAR)
    # Visualize unwarp
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    f.subplots_adjust(hspace=.2, wspace=.05)
    ax1.imshow(binaryimg)
    x = [src[0][0], src[2][0], src[3][0], src[1][0], src[0][0]]
    y = [src[0][1], src[2][1], src[3][1], src[1][1], src[0][1]]
    ax1.plot(x, y, color='#33cc99', alpha=0.4, linewidth=3, solid_capstyle='round', zorder=2)
    ax1.set_ylim([h, 0])
    ax1.set_xlim([0, w])
    ax1.set_title('Undistorted Image', fontsize=30)
    ax2.imshow(unwarped)
    ax2.set_title('Unwarped Image', fontsize=30)
    # plt.show()
    return warped, unwarped, m, m_inv

    # Return the resulting image and matrix
#======================================================================================================================================================#

#======================================================================================================================================================#
def Findlines(binary_warped):
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 10
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 80
    # Set minimum number of pixels found to recenter window
    minpix = 40
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),(0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),(0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    #=============================================================
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                   left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                         left_fit[1] * nonzeroy + left_fit[
                                                                             2] + margin)))

    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                    right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                           right_fit[1] * nonzeroy + right_fit[
                                                                               2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    return left_fit,right_fit,left_lane_inds,right_lane_inds,nonzerox,nonzeroy,out_img,left_fitx,right_fitx,margin,ploty
#======================================================================================================================================================#

#======================================================================================================================================================#
# Method to determine radius of curvature and distance from lane center
# based on binary image, polynomial fit, and L and R lane pixel indices
#======================================================================================================================================================#
def calc_curv_rad_and_center_dist(bin_img, l_fit, r_fit, l_lane_inds, r_lane_inds):
    # Define conversions in x and y from pixels space to meters
    # ym_per_pix = 3.048 / 100  # meters per pixel in y dimension, lane line is 10 ft = 3.048 meters
    # xm_per_pix = 3.7 / 378  # meters per pixel in x dimension, lane width is 12 ft = 3.7 meters
    ym_per_pix = 26 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.3 / 700  # meters per pixel in x dimension
    left_curverad, right_curverad, center_dist = (0, 0, 0)
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    h = bin_img.shape[0]
    ploty = np.linspace(0, h - 1, h)
    y_eval = np.max(ploty)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = bin_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Again, extract left and right line pixel positions
    leftx = nonzerox[l_lane_inds]
    lefty = nonzeroy[l_lane_inds]
    rightx = nonzerox[r_lane_inds]
    righty = nonzeroy[r_lane_inds]

    if len(leftx) != 0 and len(rightx) != 0:
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curverad = (
                         (1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])
        # Now our radius of curvature is in meters

    # Distance from center is image x midpoint - mean of l_fit and r_fit intercepts
    if r_fit is not None and l_fit is not None:
        car_position = bin_img.shape[1] / 2
        l_fit_x_int = l_fit[0] * h ** 2 + l_fit[1] * h + l_fit[2]
        r_fit_x_int = r_fit[0] * h ** 2 + r_fit[1] * h + r_fit[2]
        lane_center_position = (r_fit_x_int + l_fit_x_int) / 2
        center_dist = (car_position - lane_center_position) * xm_per_pix
    return left_curverad, right_curverad, center_dist
#======================================================================================================================================================#

#======================================================================================================================================================#
def draw_lane(original_img, warped, left_fitx, right_fitx, Minv,ploty):
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (original_img.shape[1], original_img.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(original_img, 1, newwarp, 0.3, 0)
    plt.imshow(result)
    # plt.show()
    return result
#======================================================================================================================================================#

#======================================================================================================================================================#
def polyfit_using_prev_fit(binary_warped, left_fit_prev, right_fit_prev):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 80
    left_lane_inds = (
    (nonzerox > (left_fit_prev[0] * (nonzeroy ** 2) + left_fit_prev[1] * nonzeroy + left_fit_prev[2] - margin)) &
    (nonzerox < (left_fit_prev[0] * (nonzeroy ** 2) + left_fit_prev[1] * nonzeroy + left_fit_prev[2] + margin)))
    right_lane_inds = (
    (nonzerox > (right_fit_prev[0] * (nonzeroy ** 2) + right_fit_prev[1] * nonzeroy + right_fit_prev[2] - margin)) &
    (nonzerox < (right_fit_prev[0] * (nonzeroy ** 2) + right_fit_prev[1] * nonzeroy + right_fit_prev[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit_new, right_fit_new = (None, None)
    if len(leftx) != 0:
        # Fit a second order polynomial to each
        left_fit_new = np.polyfit(lefty, leftx, 2)
    if len(rightx) != 0:
        right_fit_new = np.polyfit(righty, rightx, 2)
    return left_fit_new, right_fit_new, left_lane_inds, right_lane_inds
#======================================================================================================================================================#

