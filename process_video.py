from 	moviepy.editor import VideoFileClip
import os
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from lane_finding import Line
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

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

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

#global lane_tracker
lane_tracker = Line(My_ym = 25/720, My_xm = 4/500, Mysmooth_factor = 10)

def process_img(image):

        #image = cv2.imread(str1+file)
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

        #result1 = preprocess_img

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

        #use cv2.getPerspectiveTransform() to get M, the transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        #use cv2.getPerspectiveTransform() to get inverse of M, the inverse of the transform matrix
        Minv = cv2.getPerspectiveTransform(dst, src)
        #use cv2.warpPerspective() to warp your image to a top-down view
        warped = cv2.warpPerspective(preprocess_img, M, img_size,flags=cv2.INTER_LINEAR)


        #result1 = warped
        
        #set up overall class for lane finding
        #lane_tracker = Line(My_ym = 25/720, My_xm = 4/500, Mysmooth_factor = 20)
        left_fit, right_fit, center_diff, curvature = lane_tracker.main(warped) 
        #left_fit, right_fit, center_diff, curvature = lane_tracker.main(warped) 
        #curve_centers = Line(Mywindow_width = window_width, Mywindow_height = window_height, Mymargin = 25, My_ym = 20/720, My_xm = 3.7/500, Mysmooth_factor = 15)
        #window_centroids = curve_centers.find_window_centroids(warped)


        ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        #out_img[lefty, leftx] = [255, 0, 0]
        #out_img[righty, rightx] = [0, 0, 255]

        #result = out_img

        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        # Color in left and right line pixels
        #color_warp[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        #color_warp[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        #color_warp[lefty, leftx] = [255, 0, 0]
        #color_warp[righty, rightx] = [0, 0, 255]

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

        side_pos = 'right'
        if center_diff <= 0:
           side_pos = 'left'

        #draw the text showing curvature, offset and speed
        cv2.putText(result,'Radius of Curvature ='+str(round(curvature,3))+'(m)',(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
        #cv2.putText(result,'Counter ='+str(counter)+'(m)',(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
        cv2.putText(result,'Vehicle is '+str(abs(round(center_diff,3)))+'m '+side_pos+' of center',(50,100), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)

        return result


#project_output = 'solution_harder.mp4'
#input_video = 'harder_challenge_video.mp4'
project_output = 'project_solution.mp4'
input_video = 'project_video.mp4'
clip1 = VideoFileClip(input_video)
#new_clip = clip1.subclip(1,10)
#project_clip = new_clip.fl_image(process_img) #NOTE: this function expects color images!!
project_clip = clip1.fl_image(process_img) #NOTE: this function expects color images!!
project_clip.write_videofile(project_output, audio=False)
