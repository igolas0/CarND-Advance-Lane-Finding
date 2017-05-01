## Advanced Lane Finding

The goal of this project is to write a software pipeline to identify the road lane boundaries in a video. The steps followed to achieve this goal are described below:


The Project
---

The steps of this project are the following:

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
[image7]: ./output_images/test1_undist.jpg "Test1 image undistorted"
[image8]: ./output_images/result_test3.jpg "Test3 binary result"
[image9]: ./test_images/test3.jpg "Test3"
[image10]: ./output_images/result_straight_lines2.jpg "binary straight lines"
[image11]: ./test_images/straight_lines2.jpg "Test straight lines"
[image12]: ./output_images/conv1.jpg "convolutions example1"
[image13]: ./output_images/conv2.jpg "convolutions example2"
[video1]: ./project_video.mp4 "Video"

---


#### 1. Camera Calibration


The code for this step is contained in the python script file camera_calibration.py.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Applying the distortion correction parameters of the previous step on a raw image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

Loading the parameters which were stored in the pickle file during the camera calibration process described above and applying them to the raw image above in the same way as described in the file camera_calibration.py produces this image:

![alt text][image7]

This is an important step preceding the detection of the lanes so that we get rid of the distortion introduced by the camera and get better information of real distances and shapes of the objects found in the images.

#### 2. Using color transforms, Open CV's sobel gradients and thresholds to create a thresholded binary image.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines #11 through #118 in `process_image.py`).  Here's an example of my output for this step.  

![alt text][image8]

And this is the original test image:

![alt text][image9]

#### 3. Perspective transform.

To be able to better identify the lanes on the road and better judging the curvature of the lanes we need to get a "bird-eyes" view of the road. For this we will use a perspective transform using OpenCV's Library.

The code for my perspective transform includes a function called `warper()`, which appears in lines 120 through 142 in the file `process_image.py` The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
        img_size = (image.shape[1], image.shape[0])
        bot_width = .76 #percent of bottom trapizoid height
        mid_width = .08 #percent of middle trapizoid height
        height_pct = .62 #percent for trapizoid height
        bottom_trim = .935 # percent from top (bottom of img) to avoid hood)

        src = np.float32([[image.shape[1]*(.5-mid_width/2),image.shape[0]*height_pct],
                          [image.shape[1]*(.5+mid_width/2),image.shape[0]*height_pct],
                          [image.shape[1]*(.5+bot_width/2),image.shape[0]*bottom_trim],
                          [image.shape[1]*(.5-bot_width/2),image.shape[0]*bottom_trim]])

        offset = img_size[0]*0.25

        dst = np.float32([[offset, 0], [img_size[0]-offset, 0],
                          [img_size[0]-offset, img_size[1]], 
                          [offset, img_size[1]]])  
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 589, 446      | 320, 0        | 
| 691, 446      | 960, 0      |
| 1126, 673     | 960, 720      |
| 154, 673      | 320, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

Here is the result of a warped binary image:

![alt text][image10]

And here is the corresponding original image:

![alt text][image11]

#### 4.  Identifying lane-line pixels and fitting their positions with a polynomial

In order to identify the lane-line pixels I followed a sliding window method search using convolutions. These convolutions will maximize the number of "hot" pixels in each window. A convolution is the summation of the product of two separate signals, in our case the window template and the vertical slice of the pixel image.

The window template slides across the image from left to right and any overlapping values are summed together, creating the convolved signal. The peak of the convolved signal is where there was the highest overlap of pixels and the most likely position for the lane marker.

The code for this is define also in the file 'process_image.py' from line 146 to 179. Among these lines of code the function find_window_centroids() is called and an object 'curve_centers' of the Class Line is created. Both the function find_window_centroids() and the class Line are defined in the file 'lane_finding.py'.  

Here are two examples of the images which result after the sliding window search using convolutions. Here is example no. 1:

![alt text][image12]

And here is another example:

![alt text][image13]

Then I fitted the identified window centroids on top of the left and right lanes with a 2nd order polynomial. The code is again found on the file 'process_image.py' from line 181 to 197. Below you can find a descriptive image of the followed principle, though it is not an actual image generated by the described code:   

![alt text][image5]

Finally, the identified lines (polynomials) and the area in between was drawed on to the image (code lines 199 to 211). 

#### 5. Calculating the radius of curvature of the lane and the position of the vehicle with respect to center.

The calculation of the radius of lane curvature and vehicle position relative to the center takes place from lines 213 to 235. To achieve this we first need to convert pixel space to real physical space. Actually we already defined this conversions back in the code line no. 150 ("process_image.py"), when we created the object of the Line class. The parameters My_ym and My_xm represent the conversions in vertical and horizontal dimensions respectively. In this case 720 pixels correspond to 30 meters of physical space in vertical direction and 700 pixels represent 3.7 meters of real space.

 The way we come up with these numbers is by comparing the warped images above to the U.S. regulation that requires a minimum lane width of 12 feet or 3.7 meters, and the dashed lane lines are 10 feet or 3 meters long each. Using this conversions we can compute lane curvature using this [formula](http://www.intmath.com/applications-differentiation/8-radius-curvature.php).

 The displacement from the center of the lane is easily computed by using the horizontal conversion above and applying it to the divergence in pixels of the center of the image from the center between the identified lanes. 

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
