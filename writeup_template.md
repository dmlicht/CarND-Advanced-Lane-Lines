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

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

[undistort]: ./writeup_images/distortion_correction_example.png "Distortion Correction Example"
[thresholding]: ./writeup_images/thresholding.png "Thresholding tuning images"
[large_transform_example]: ./writeup_images/large_transformation_example "Large Perspective Transformation Example"
[with_dots_all]: ./writeup_images/with_dots_all.png "Dots showing transformation points"
[transformed_all]: ./writeup_images/transformed_all.png "All example images transformed"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

I calibrate the camera matrix and distortion coefficients in `distortion_correction.py` in the method `DistortionCorrection.fit_to_chessboards`
This function takes a generator of chessboard images, finds the corners and uses them as "object points"

I'm treating the object points as the (x, y, z) coordinates of the chessboard corners in the world. 
Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  
Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  
`imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  
I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result.
 
I wrap the distortion correction setup and functionality in the utility class `DistortionCorrection`.
DistortionCorrection can be used as shown below `##Test Distortion Correction On All Images` in `distortion_correction.ipynb`.

![Example of distortion correction][undistort]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![Example of distortion correction][undistort]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I explored color transformations and threshold parameters in `thresholding.ipynb`
![Images from tuning gradient and color channel thresholding][thresholding]

In my actual pipeline, I ended up using only saturation without combining the results with any gradients or other color channels, because
saturation on its own appearing to give the clearest signals. The code that is used in my final pipeline is in `threshold.py`.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is in `perspective_transform.py`. 
I tuned and tested my perspective transform points in `perspective_transform.ipynb`

I handle transformation using a `PerspectiveTransform` class that takes the shape of an image and uses hardcoded values
that I visually selected and verified my values for `src` and `dst` in `perspective_transform.ipynb`. 
The hardcoded values would have to be changed for images of different sizes and for images with different camera angles.

```python
PERSPECTIVE_BOTTOM_LEFT = (250, 680)
PERSPECTIVE_BOTTOM_RIGHT = (1090, 680)
PERSPECTIVE_TOP_LEFT = (595, 450)
PERSPECTIVE_TOP_RIGHT = (715, 450)
```

These values will be mapped to the outside corners of the top down perspective image with some degree of padding.

![Large Perspective Transformation Example][large_transform_example]
![Dots showing transformation points][with_dots_all]
![All example images transformed][transformed_all]


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

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

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
