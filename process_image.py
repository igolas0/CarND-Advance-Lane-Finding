import os
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#from lane_finding_v2 import Line
#%matplotlib inline


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(10, 255)):
    
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
           
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return binary_output


def mag_thresh(img, sobel_kernel=3, mag_thresh=(10, 255)):
    
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0,ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1,ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    # Return the binary image
    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)        
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0,ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1,ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    grad_dir = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(grad_dir)
    binary_output[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output 

def color_threshold(img, s_thresh=(255, 255), b_lab_thresh=(250, 255), r_thresh=(150, 255), b_thresh=(200, 255)):
    image = np.copy(img)
    #extrag bgr channels
    b_channel = image[:,:,0]
    g_channel = image[:,:,1]
    r_channel = image[:,:,2]
    # Convert to Lab color space and separate the L channel
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab).astype(np.float)
    b_lab_channel = lab[:,:,2]
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    v_channel = hsv[:,:,2]
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
    h_channel = hls[:,:,0]
    #l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
      
    combined = np.zeros_like(s_channel)
    combined[(b_lab_channel >= b_lab_thresh[0]) & (b_lab_channel <= b_lab_thresh[1]) 
                                         | (r_channel >= r_thresh[0]) & (r_channel <= r_thresh[1])
                                         | (b_channel >= b_thresh[0]) & (b_channel <= b_thresh[1])
                                         | (s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    return combined

def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output


# Read in the saved camera matrix and distortion coefficients saved in pickle file
dist_pickle = pickle.load( open("./camera_cal/dist_pickle.p", "rb" ))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

dirs = os.listdir("test_images/")
str1 = './test_images/'
str2 = 'result_'

for file in dirs:
    if file[0:7] != str2:

        image = cv2.imread(str1+file)
        image = cv2.undistort(image, mtx, dist, None, mtx)

        #ksize = 3
        #mag_tr = (20,255)

        #gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize,thresh= (12, 255))
        #grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize,thresh= (25, 255))
        #mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=mag_tr)
        #dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.3))
        color_binary = color_threshold(image, b_lab_thresh=(150,255), s_thresh=(255,255), b_thresh=(190,255), r_thresh=(255,255))
        preprocess_img = np.zeros_like(color_binary)
        #preprocess_img[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (color_binary == 1)] = 255 
        preprocess_img[(color_binary == 1)] = 255 

        result1 = preprocess_img

        #define are for pespective transform
        img_size = (image.shape[1], image.shape[0])
        bot_width = .76 #percent of bottom trapizoid width
        #mid_width = .08 #percent of middle trapizoid width
        mid_width = .17 #percent of middle trapizoid width
        #height_pct = .62 #percent for trapizoid height
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
        #print('src=',src)
        #print('dst=',dst)

        #use cv2.getPerspectiveTransform() to get M, the transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        #use cv2.getPerspectiveTransform() to get inverse of M, the inverse of the transform matrix
        Minv = cv2.getPerspectiveTransform(dst, src)
        #use cv2.warpPerspective() to warp your image to a top-down view
        warped = cv2.warpPerspective(preprocess_img, M, img_size,flags=cv2.INTER_LINEAR)

        result1 = warped
        

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


        ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        result1 = out_img

        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        # Color in left and right line pixels
        #color_warp[lefty, leftx] = [255, 0, 0]
        #color_warp[righty, rightx] = [0, 0, 255]


        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
        
        xm_per_pix = 4/700
        ym_per_pix = 25/720

        #y_eval = 600
        y_eval = np.max(ploty)

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
        # Calculate the new radius of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        curverad = (left_curverad + right_curverad)/2
        # Now our radius of curvature is in meters        


        #calculate offset of car from center of the road
        lane_center = (left_fitx[-1] + right_fitx[-1])/2
        car_center = warped.shape[1]/2
        center_diff = (car_center-lane_center)*xm_per_pix
        side_pos = 'right'
        if center_diff <= 0:
           side_pos = 'left'

        #print('leftfitx[-1]',left_fitx[-1])
        #print('rightfitx[-1]',right_fitx[-1])
        #print('center diff',center_diff)

        #draw the text showing curvature, offset and speed
        cv2.putText(result,'Radius of Curvature ='+str(round(curverad,3))+'(m)',(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
        #cv2.putText(result,'Radius of Right Curvature ='+str(round(right_curverad,3))+'(m)',(50,100), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
        cv2.putText(result,'Vehicle is '+str(abs(round(center_diff,3)))+'m '+side_pos+' of center',(50,100), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)

        write_name = './test_images/result_'+file
        cv2.imwrite(write_name, result) 
