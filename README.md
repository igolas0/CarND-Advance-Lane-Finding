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

[image1]: ./camera_cal/test_undist.jpg "Undistorted"
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
[image14]: ./output_images/result.jpg "Result image of pipeline"
[image15]: ./camera_cal/calibration1.jpg "Undistorted"
[video1]: ./project_video.mp4 "Video"

---


#### 1. Camera Calibration


The code for this step is contained in the python script file camera_calibration.py.

The camera was calibrated using the chessboard images in 'camera_cal/*.jpg'. The following steps were performed for each calibration image:

1.    Convert to grayscale
2.    Find chessboard corners with OpenCV's findChessboardCorners() function, assuming a 9x6 board

Then I used OpenCV's calibrateCamera() function to calculate the distortion correction factors. Using the distortion matrices we can undistort images using OpenCV's undistort() function. This is the result applied to the first image in the directory camera_cal:

![alt text][image1]

An this is the original image (calibration1.jpg):

![alt text][image15]

Finally the distortion correction factors were saved to a pickle file.

### Pipeline (single images)

#### 1. Applying the distortion correction parameters of the previous step on a raw image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

Loading the parameters which were stored in the pickle file during the camera calibration process described above and applying them to the raw image above in the same way as described in the file camera_calibration.py produces this image:

![alt text][image7]

This is an important step preceding the detection of the lanes so that we get rid of the distortion introduced by the camera and get better information of real distances and shapes of the objects found in the images.

#### 2. Using color transforms, sobel gradients and thresholds to create a thresholded binary image.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines #11 through #118 in `process_image.py`).  Here's an example of my output for this step.  

![alt text][image8]

And this is the original test image:

![alt text][image9]

#### 3. Perspective transform.

To be able to better identify the lanes on the road and better judging the curvature of the lanes we need to get a "bird-eyes" view of the road. For this we will use a perspective transform using OpenCV's Library.

The code for my perspective transform appears in lines 120 through 142 in the file `process_image.py`. I make use of OpenCV's function getPerspectiveTransform(), which takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

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

Here is an example result of a warped binary image:

![alt text][image10]

And here is the corresponding original image:

![alt text][image11]

#### 4.  Identifying lane-line pixels and fitting their positions with a polynomial

In order to identify the lane-line pixels I followed a sliding window method search. First of all we compute the histogram of the bottom half of the binary image. This way we can track where most pixels are located, which will be the origins for our window search for the left and right lane lines.

The code for this is defined also in the file 'process_image.py' from line 147 to 221. Here are two examples of the images which result after using the sliding window search on the binary images. Here is example no. 1:

![alt text][image12]

And here is another example:

![alt text][image13]

Then I fitted the identified pixels corresponding to the left and right lanes with a 2nd order polynomial. The code is again found on the file 'process_image.py' from line 211 to 213. Below you can find a descriptive image of the followed principle, though it is not an actual image generated by the described code:   

![alt text][image5]


#### 5. Calculating the radius of curvature of the lane and the position of the vehicle with respect to center.

The calculation of the radius of lane curvature and vehicle position relative to the center takes place from lines 246 to 266. To achieve this we first need to convert pixel space to real physical space. The parameters xm_per_pix and ym_per_pix represent the conversions in vertical and horizontal dimensions respectively. In this case 100 pixels correspond to 3 meters of physical space in vertical direction and 700 pixels represent 3.7 meters of real space.

 The way we come up with these numbers is by comparing the warped images above to the U.S. regulation that requires a minimum lane width of 12 feet or 3.7 meters, and the dashed lane lines are 10 feet or 3 meters long each. Using this conversions we can compute lane curvature using this [formula](http://www.intmath.com/applications-differentiation/8-radius-curvature.php).

 The displacement from the center of the lane is easily computed by using the horizontal conversion above and applying it to the divergence in pixels of the center of the image from the center between the identified lanes. 

#### 6. Drawing the lane boundaries on the road.

I implemented this step in lines #233 through #243 in my code in `process_image.py`.  Here is an example of my result on a test image:

![alt text][image14]

---

### Pipeline (video)

Here's the [link to my video result](./project_solution.mp4)

---

### Discussion

The bigger challenges of this project were as in the first project to detect the lanes in different lighting conditions as well as lanes of different colors. Changes of the road pavement color or texture impose a difficult task identifying the lanes.

My pipeline suffices to meet the requirements of the project video, but does not suffice to handle the more difficult conditions met in the challenge videos. Here is further work needed.

In the challenge videos the reflections of the sun through the windshield on to the camera made the pipeline detect a lot of clutter. The big turns were also a problem to the conshardcoded source and destination points of the perspective transform. Ideally one would define a more dinamically defined perspective transform based on the measured curvature.

There is also room for further optimizing the thresholding of the binary image to improve the results. Further experimentation is needed (maybe including dynamic thresholding). Also a more refined way of smoothing the results over several frames and logic to discard difficult frames would be needed for good results on the challenger videos. 

