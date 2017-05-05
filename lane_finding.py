global int counter

 

# New = gamma * New + (1-gamma) * Old. 0 < gamma < 1.0

 

 

class Line():

    def __init__(self):

        # was the line detected in the last iteration?

        self.detected = False 

        # x values of the last n fits of the line

        self.recent_xfitted = []

        #average x values of the fitted line over the last n iterations

        self.bestx = None    

        #polynomial coefficients averaged over the last n iterations

        self.best_fit = None 

        #polynomial coefficients for the most recent fit

        self.current_fit = [np.array([False])] 

        #radius of curvature of the line in some units

        self.radius_of_curvature = None

        #distance in meters of vehicle center from the line

        self.line_base_pos = None

        #difference in fit coefficients between last and new fits

        self.diffs = np.array([0,0,0], dtype='float')

        #x values for detected line pixels

        self.allx = None 

        #y values for detected line pixels

        self.ally = None

 

def sanity_check(self):
    self.detected = False

    if (200 < self.left_fitx[-1] < 400) & (900 < self.right_fitx[-1] < 1100):

       self.detected = True

       if self.left_radius < 2000 or right_radius < 2000:

          check_radius = self.right_radius/self.left_radius

          if check_radius<0.5 or check_radius>1.5:
             self.detected = False


 

def find_lanes(self, warped):

               

        # Take a histogram of the bottom half of the image

        histogram = np.sum(warped[int(warped.shape[0]/2):,:], axis=0)

        # Create an output image to draw on and  visualize the result

        out_img = np.dstack((warped, warped, warped))*255

 

        # Find the peak of the left and right halves of the histogram

        # These will be the starting point for the left and right lines

        midpoint = np.int(histogram.shape[0]/2)

        leftx_base = np.argmax(histogram[:midpoint])

        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

 

        # Choose the number of sliding windows

        nwindows = 9

        # Set height of windows

        window_height = np.int(warped.shape[0]/nwindows)

        # Identify the x and y positions of all nonzero pixels in the image

        nonzero = warped.nonzero()

        nonzeroy = np.array(nonzero[0])

        nonzerox = np.array(nonzero[1])

        # Current positions to be updated for each window

        leftx_current = leftx_base

        rightx_current = rightx_base

        # Set the width of the windows +/- margin

        margin = 100

        # Set minimum number of pixels found to recenter window

        minpix = 50

        # Create empty lists to receive left and right lane pixel indices

        left_lane_inds = []

        right_lane_inds = []

 

        # Step through the windows one by one

        for window in range(nwindows):

           # Identify window boundaries in x and y (and right and left)

           win_y_low = warped.shape[0] - (window+1)*window_height

           win_y_high = warped.shape[0] - window*window_height

           win_xleft_low = leftx_current - margin

           win_xleft_high = leftx_current + margin

           win_xright_low = rightx_current - margin

           win_xright_high = rightx_current + margin

           # Draw the windows on the visualization image

           cv2.rectangle(out_img,(int(win_xleft_low),int(win_y_low)),(int(win_xleft_high),int(win_y_high)),(0,255,0), 2)

           cv2.rectangle(out_img,(int(win_xright_low),int(win_y_low)),(int(win_xright_high),int(win_y_high)),(0,255,0), 2)

           # Identify the nonzero pixels in x and y within the window

           good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]

           good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

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

        return left_fit, right_fit

                                                              

#if counter false detections >10:

#--- > search from scratch:

#else: search using last windows:

#             return leftx, lefty, rightx, righty

                              

Do Sanity check:

 

if False take Last Best fit, last center pos and last best Curvature,

                else WEIGHTED AVERAGE OVER CURRENT AND LAST FITS, CURVATURES (take middle of left and right) and center_pos

 

def sanity_check:

 

                self.detected = Faulse

 

  1.) curvatures similar

  2.) intercepts in range +- 100 px

  3.) (if detected/last_detected = True: xvalues from polyfit at y=0 and ymax similar (+-25px max) 

 

                if 1.)2.)3.)= TRUE: then self.detected = TRUE and SET COUNTER TO ZERO!

                ELSE counter += 1

               

                return self.detected
