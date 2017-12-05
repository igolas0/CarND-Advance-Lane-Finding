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
[image8]: ./output_images/bin1.jpg "Test3 binary result"
[image9]: ./test_images/straight_lines1.jpg "Test straight 1"
[image10]: ./output_images/result_straight_lines2.jpg "binary straight lines"
[image11]: ./test_images/straight_lines2.jpg "Test straight lines"
[image12]: ./output_images/conv1.jpg "convolutions example1"
[image13]: ./output_images/conv2.jpg "convolutions example2"
[image14]: ./output_images/result.jpg "Result image of pipeline"
[image15]: ./camera_cal/calibration1.jpg "Undistorted"
[image16]: ./output_images/bin3.jpg "Test3 binary result"
[image17]: ./test_images/test6.jpg "Test straight 1"
[video1]: ./project_video.mp4 "Video"

---


#### 1. Camera Calibration


The code for this step is contained in the python script file camera_calibration.py.

The camera was calibrated using the chessboard images in `camera_cal/*.jpg`. The following steps were performed for each calibration image:

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

#### 2. Using color transforms and thresholds to create a thresholded binary image.

I used a combination of color thresholds to generate a binary image. Before this combination I was also using sobel gradients, but I decided to ditch them, since they create a lot of noise in for example images with a lot of shadows.

To create the color transforms and thresholds I defined the function `color_threshold()`  (lines #70 through #93 in `process_image.py`) and call this function later in lines 123 to 126. I got the best results thresholding the colors `b` from Lab colorspace and `B` from RGB colorspace. In conjunction they did a good job detecting both yellow and white lanes while minimizing noise.


 Here's an example of my output for this step.  

![alt text][image8]

And this is the original test image:

![alt text][image9]

And here is another example:

![alt text][image16]

And its original counterpart:

![alt text][image17]

#### 3. Perspective transform.

To be able to better identify the lanes on the road and better judging the curvature of the lanes we need to get a "bird-eyes" view of the road. For this we will use a perspective transform using OpenCV's Library.

The code for my perspective transform appears in lines 131 through 154 in the file `process_image.py`. I make use of OpenCV's function getPerspectiveTransform(), which takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
        img_size = (image.shape[1], image.shape[0])
        bot_width = .76 #percent of bottom trapizoid height
        mid_width = .17 #percent of middle trapizoid height
        height_pct = .66 #percent for trapizoid height
        bottom_trim = .935 # percent from top (bottom of img) to avoid hood)

        src = np.float32([[image.shape[1]*(.5-mid_width/2),image.shape[0]*height_pct],
                          [image.shape[1]*(.5+mid_width/2),image.shape[0]*height_pct],
                          [image.shape[1]*(.5+bot_width/2),image.shape[0]*bottom_trim],
                          [image.shape[1]*(.5-bot_width/2),image.shape[0]*bottom_trim]])

        offset = img_size[0]*0.15

        dst = np.float32([[offset, 0], [img_size[0]-offset, 0],
                          [img_size[0]-offset, img_size[1]], 
                          [offset, img_size[1]]])  
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 531, 475      | 192, 0        | 
| 749, 475      | 1088, 0      |
| 1126, 673     | 1088, 720      |
| 154, 673      | 192, 720      |

Here is an example result of a warped binary image:

![alt text][image10]

And here is the corresponding original image:

![alt text][image11]

#### 4.  Identifying lane-line pixels and fitting their positions with a polynomial

In order to identify the lane-line pixels I followed a sliding window method search. First of all we compute the histogram of the bottom half of the binary image. This way we can track where most pixels are located, which will be the origins for our window search for the left and right lane lines.

The code for this is defined also in the file 'process_image.py' from line 159 to 235. Here are two examples of the images which result after using the sliding window search on the binary images. Here is example no. 1:

![alt text][image12]

And here is another example:

![alt text][image13]

Then I fitted the identified pixels corresponding to the left and right lanes with a 2nd order polynomial. The code is again found on the file 'process_image.py' from line 224 to 230. Below you can find a descriptive image of the followed principle, though it is not an actual image generated by the described code:   

![alt text][image5]


#### 5. Calculating the radius of curvature of the lane and the position of the vehicle with respect to center.

The calculation of the radius of lane curvature and vehicle position relative to the center takes place from lines 265 to 279. To achieve this we first need to convert pixel space to real physical space. The parameters xm_per_pix and ym_per_pix represent the conversions in vertical and horizontal dimensions respectively (in warped image space). In this case 720 pixels correspond to 25 meters of physical space in vertical direction (as reference was taken the length of one lane, which is about 3 meters) and 700 pixels represent 4 meters of real space in horizonzal direction (in warped space this was the average distance in pixels between the lanes. Four meters is a good approximation for the real width of a lane). Also a good measure to judge if the chosen parameters were good was benchmarking the resulting curvature of the first left turn in the project video (which measured from a map has a radius of about 1km).

 Using this conversions we can compute lane curvature using this [formula](http://www.intmath.com/applications-differentiation/8-radius-curvature.php). The displacement from the center of the lane is easily computed by using the horizontal conversion above and applying it to the divergence in pixels of the center of the image from the center between the identified lanes. 

#### 6. Drawing the lane boundaries on the road.

I implemented this step in lines #237 through #255 in my code in `process_image.py`.  Here is an example of my result on a test image:

![alt text][image14]

---

### Pipeline (video)

In the pipeline for the video a very similar approach was used. The main difference is that a sanity check is introduced for every frame where the lanes are tracked to check if the identified lines are valid and then averaging over the ten last valid detections takes place. 

 The pipeline is implemented in the file `process_video.py`. Most of the functionality described above plus the added sanity checks and averaging now take place using a defined class for this purpose. The class `Line` and its methods are defined in the file `lane_finding.py`.

The main method of the class Line in `lane_finding.py` is main(), which is called from `process_video.py` in line of code 186 (first of all we must create/initialize an object of the class Line - see line 136 of process_video.py). 

This main method (starting in line of code 40 of lane_finding.py) calls the class methods find_lanes(), sanity_check(), calc_radius() and car_pos(). The find_lanes() function is identical to the one described/used for the image pipeline. The sanity check function checks if the intercept of the identified lanes are inside of an expected x-dimension interval and also that the curvatures of the left and right lanes are not too different. If the sanity check succeeds the detection is added to the history and the average of the last ten valid frames is taken into account to compute the lanes. If not, the current frame detection is ditched and the last ten valid detections are taken into account (the video is shot at 25 frames per second, so even with some failed detections the time offset should not be to big).

Here's the [link to my video result](./project_solution.mp4)

And here is  [link to my video result for the challenge video](./solution_challenge.mp4) (without changing the src and dst points for the perspective transform which actually is rather a must for this video in my view).

---

### Discussion

The bigger challenges of this project were as in the first project to detect the lanes in different lighting conditions as well as lanes of different colors. Changes of the road pavement color or texture impose a difficult task identifying the lanes.

My pipeline works pretty well in the project video and fairly well in the harder challenge videos. Still, further improvements are possible.

The perspective transform needed for the challenge videos is actually different. The key to achieve a general working pipeline is a good dynamically adjusting perspective transform (which adjusts depending on measured curvature, slope or maybe other parameters).

Also the thresholds to filter the lane features into a binary image were pretty much hardcoded and it is very difficult to find a static hard-coded solution that works under all lighting/shadow conditions. A more dynamical and adapting function is needed, especially for standard difficult situations like for example too much glare from the sun.

One easy thing to implement would be to perform a weighted numpy averaging of the last detections (with higher weights for the more recent frames).

Another thing would be to skip searching from scratch the lanes for the frames where the previus frame lanes were successfully detected and then only fully resetting the detection process after a couple of failed frame detections in a row.

The difficulty of using classic computer vision approaches makes me appreciate me even more the beauty of end-to-end machine learning with deep neural nets. Still it is very enriching and insightful to learn more about the classical approach. Even for pure DNN approaches the classical knowledge helps us judge better the usefulness of the learned features.

