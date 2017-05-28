## Writeup

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
[distortion_correction]: ./writeup_images/distortion_correction.png
[distortion_correction_many]: ./writeup_images/distortion_correction_many.png
[thresholding]: ./writeup_images/thresholding.png "Thresholding tuning images"
[large_transform_example]: ./writeup_images/large_transform_example.png "Large Perspective Transformation Example"
[with_dots_all]: ./writeup_images/with_dots_all.png "Dots showing transformation points"
[transformed_all]: ./writeup_images/transformed_all.png "All example images transformed"
[fit_lines]: ./writeup_images/fit_lines.png "fit lines to our points"
[curve_radius]: ./writeup_images/radius_of_curve.png "radius of curve"
[test_image_inputs]: ./writeup_images/test_images.png

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
![Example of distortion correction in pipeline][distortion_correction]

and on all the test images before:

![before][test_image_inputs]
![after][distortion_correction]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I explored color transformations and threshold parameters in `thresholding.ipynb`
![Images from tuning gradient and color channel thresholding][thresholding]

In the end I ended up using a combination of thresholded gradientxy magnitude and the color saturation channel. 
The code that is used in my final pipeline is in `threshold.py`.

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

I identify lane-line pixels and fit them with polynomials in `find_lines.py`.
We can do this in two way, by sliding a window across the image and by using previous lines.
 
###### Sliding window (`find_lines_with_sliding_window`)
To identify lines with a sliding window, we first need to decide where to initially place the window and choose parameters for shape.
We can estimate roughly where the line will be in the image by taking the bottom half and creating a histogram of where on the x-axis
We split the image in half horizontally as well, because we want to find two lines, one for each lane. 
Then within each half we will choose our centers based on where most of the active pixels occur. 
This will show us a lot of where the line is and provide a good starting point.
Then, once we have a window to start with, we can use all of the active pixels that occur within the window to create a
new x-value center for the next window up. We continue this process until we slide the window up across the entire image.
Then once we have a collection of all of the images that have occured within our windows. 
We can fit a polynomial to all of the pixels we've identified. 

###### Using prior Lines (`find_lines.find_line_from_prior`) 
If we have already fit a line in a previous frame, we can use it to identify candidate points for our new lines.
Because the two images occured in close time proximity, we can assume that the lines should be very close to each other.
Using this assumption, we can take all the points that occur within a certain distance from our previous line and use them
as the points for our new line. Once we have the points, we can fit our line in the same way we did before with the sliding window.


![fit lines to binary images][fit_lines]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I calculate the radius of the curve with respect to the center in `find_lines.curve_radius`.
It seems to be working as the radius gets very large when the road is relatively straight and shrinks during curves in the road.

![curve radius][curve_radius]


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implement a the full pipeline in `pipeline.py` in the function `AveragingPipeline.highlight_lane`.
This function can be passed into `VideoFileClip.fl_image` to annotate the lanes of a whole video.
It averages the results of previous frames to smooth changes between frames.

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's my [result video](./output_videos/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One of the biggest challenges I faced in the image was getting good top down binary versions of the images. 
I had to spend several hours tweaking different thresholds for gradients and saturation and combining them in various ways. 
In the end I got significantly better results by running the perspective transform **before** I converted the images to binary pixels.
Doing it in the reverse order produced a lot of noise.
To continue improving this project I would likely create a more thoughtful management system for tracking and combining the last few lane lines. 
