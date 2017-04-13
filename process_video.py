from moviepy.editor import VideoFileClip
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

def color_threshold(img, s_thresh=(100, 255), v_thresh=(100, 255)):
    image = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    v_channel = hsv[:,:,2]
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
      
    combined = np.zeros_like(s_channel)
    combined[(v_channel >= v_thresh[0]) & (v_channel <= v_thresh[1]) 
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

def process_img(image):

        #image = cv2.imread(str1+file)
        image = cv2.undistort(image, mtx, dist, None, mtx)

        ksize = 3
        mag_tr = (20,255)
        #preprocess_img = np.zeros_like(image[:,:,0])
        gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize,thresh= (12, 255))
        grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize,thresh= (25, 255))
        mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=mag_tr)
        dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.3))
        color_binary = color_threshold(image, v_thresh=(150,255), s_thresh=(100,255))
        preprocess_img = np.zeros_like(gradx)
        preprocess_img[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (color_binary == 1)] = 255 

        #define are for prespective transform
        img_size = (image.shape[1], image.shape[0])
        bot_width = .76 #percent of bottom trapizoid height
        mid_width = .08 #percent of middle trapizoid height
        height_pct = .62 #percent for trapizoid height
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
        window_width = 80
        window_height = 80

        #set up overall class for lane finding
        curve_centers = Line(Mywindow_width = window_width, Mywindow_height = window_height, Mymargin = 25, My_ym = 30/720, My_xm = 3.7/700, Mysmooth_factor = 15)
        window_centroids = curve_centers.find_window_centroids(warped)

        #points to draw left and right windows
        l_points = np.zeros_like(warped)
        r_points = np.zeros_like(warped)

        #points used to find the left and right lanes
        rightx = []
        leftx = []

        #iterate over each level and draw windows
        for level in range(0,len(window_centroids)):
           #window_mask is a function to draw window areas
           l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
           r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
           #add center value in frame to list of lane points per left,right
           leftx.append(window_centroids[level][0])
           rightx.append(window_centroids[level][1])
	   # Add graphic points from window mask here to total pixels found 
           l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
           r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

        # Draw the results
        template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
        zero_channel = np.zeros_like(template) # create a zero color channle 
        template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
        warpage = np.array(cv2.merge((warped,warped,warped)),np.uint8) # making the original road pixels 3 color channels
        #result = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results
        
        #fit the lane boundaries to the left,right center positions found
        yvals = range(0,warped.shape[0])

        res_yvals = np.arange(warped.shape[0]-(window_height/2),0,-window_height)

        left_fit = np.polyfit(res_yvals, leftx, 2)
        left_fitx = left_fit[0]*yvals*yvals + left_fit[1]*yvals + left_fit[2]
        left_fitx = np.array(left_fitx,np.int32)


        right_fit = np.polyfit(res_yvals, rightx, 2)
        right_fitx = left_fit[0]*yvals*yvals + right_fit[1]*yvals + right_fit[2]
        right_fitx = np.array(right_fitx,np.int32)

        left_lane = np.array(list(zip(np.concatenate((left_fitx-window_width/2,left_fitx[::-1]+window_width/2),axis=0),np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)
        right_lane = np.array(list(zip(np.concatenate((right_fitx-window_width/2,right_fitx[::-1]+window_width/2),axis=0),np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)
        middle_marker = np.array(list(zip(np.concatenate((right_fitx-window_width/2,right_fitx[::-1]+window_width/2),axis=0),np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)
        inner_lane = np.array(list(zip(np.concatenate((left_fitx+window_width/2,right_fitx[::-1]-window_width/2),axis=0),np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)


        road = np.zeros_like(image)
        road_bkg = np.zeros_like(image)
        cv2.fillPoly(road,[left_lane],color=[255,0,0]) 
        cv2.fillPoly(road,[right_lane],color=[0,0,255]) 
        cv2.fillPoly(road,[inner_lane],color=[0,255,0]) 
        cv2.fillPoly(road_bkg,[left_lane],color=[255,255,255]) 
        cv2.fillPoly(road_bkg,[right_lane],color=[255,255,255]) 

        #result = road

        road_warped = cv2.warpPerspective(road,Minv, img_size, flags=cv2.INTER_LINEAR)
        road_warped_bkg = cv2.warpPerspective(road_bkg,Minv, img_size, flags=cv2.INTER_LINEAR)
        base = cv2.addWeighted(image, 1.0, road_warped_bkg, -1.0, 0.0)
        result = cv2.addWeighted(base, 1.0, road_warped, 0.7, 0.0)


        xm_per_pix = curve_centers.xm_per_pix
        ym_per_pix = curve_centers.ym_per_pix


        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(np.array(res_yvals,np.float32)*ym_per_pix, np.array(leftx,np.float32)*xm_per_pix, 2)
        right_fit_cr = np.polyfit(np.array(res_yvals,np.float32)*ym_per_pix, np.array(rightx,np.float32)*xm_per_pix, 2)
        # Calculate the new radius of curvature  
        left_curverad = ((1 + (2*left_fit_cr[0]*yvals[-1]*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*yvals[-1]*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        # Now our radius of curvature is in meters        

        #calculate offset of car from center of the road
        camera_center = (left_fitx[-1] + right_fitx[-1])/2
        center_diff = (camera_center-warped.shape[1]/2)*xm_per_pix
        side_pos = 'left'
        if center_diff <= 0:
           side_pos = 'right'

        #draw the text showing curvature, offset and speed
        cv2.putText(result,'Radius of Left Curvature ='+str(round(left_curverad,3))+'(m)',(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
        cv2.putText(result,'Vehicle is '+str(abs(round(center_diff,3)))+'m '+side_pos+' of center',(50,100), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)

        return result


project_output = 'project_solution.mp4'
input_video = 'project_video.mp4'
clip1 = VideoFileClip(input_video)
project_clip = clip1.fl_image(process_img) #NOTE: this function expects color images!!
project_clip.write_videofile(project_output, audio=False)
