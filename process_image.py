import os
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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

def pipeline_test(img, s_thresh=(100, 255), v_thresh=(50, 255), mag_tr=(50, 255), ksize=3):
    image = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    red_channel = image[:,:,0]
    
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=thresh)
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize,thresh=thresh)
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=mag_tr)
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.3))
      
    combined = np.zeros_like(gradx)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))
                                           | (s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
      
    return combined


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

        result = warped

        write_name = './test_images/result_'+file
        cv2.imwrite(write_name, result) 
