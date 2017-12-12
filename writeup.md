## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./report/Cam_Calib.png "Undistorted"
[image2]: ./report/Figure_distort.png "Road Transformed"
[image3]: ./report/Figure_1-1.png "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./report/Figure_1-2.png "Output"
[image7]:./report/calib_undistortion.png
[video1]: ./output_video/output_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  
'Main.py' is the mai file and used functions are found in 'Utilities.py'
---



### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

I used the following reference for camera calibration:

 * http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
 
 Important input data needed for camera calibration is a set of 3D real world points and its corresponding 2D image points. 2D image points are OK which we can easily find from the image.
 3D points are called object points and 2D image points are called image points.
I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. 
Then, I refine the camera matrix based on a free scaling parameter using cv2.getOptimalNewCameraMatrix().
![alt text][image1]

![alt text][image7]
### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
I used `Undistortion()` which call `cv2.undistort()`
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps found in function `ColorandGradient()` in `Utilities.py`). 
I used HLS combined with sobel.
 Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `corners_unwarp()`, which appears in the file `Utilities.py`.  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.I tried several values for src and destination points (manually selected) and I chose the source and destination points in the following manner:

```python
 src = np.float32(
        [[200, 720],
         [1100, 720],
         [595, 450],
         [685, 450]])
    dst = np.float32(
        [[300, 720],
         [980, 720],
         [300, 0],
         [980, 0]])
```


I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image3]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I used `Findlines()` function.I first take a histogram along all the columns in the lower half of the image.With this histogram I am adding up the pixel values along each column in the image. In my thresholded binary image, pixels are either 0 or 1, so the two most prominent peaks in this histogram will be good indicators of the x-position of the base of the lane lines.
I used also `polyfit_using_prev_fit()` after `Findlines()`to fit 2nd order polynomial.
![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I used function `calc_curv_rad_and_center_dist()` to determine radius of curvature and distance from lane center based on binary image, polynomial fit, and L and R lane pixel indices.
Assuming that lane width is 3.3m.


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
I used the following steps in my project:
* Camera calibration:

  at the begining I didn't use termination criteria so I found issue that distortion coefficient and matrix were null.
* Distortion correction

   I didn't find distortion give much better results.
   
* Color/gradient threshold

    Maybe if I combined different color thresholds it will give better results.
* Perspective transform
    I couldn't find an automatic way to get source and distenation points, this maybe more accurate and easy.
