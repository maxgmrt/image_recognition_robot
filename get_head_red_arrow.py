import utils_max
from utils_max import *
import cv2 as cv
import numpy as np


# returns highest (in terms of y axis) point in red arrow
def get_head_red_arrow(frame_hist_eq):

    # ############
    # # RED MASK #
    # ############

    # Converts images from BGR to HSV 
    hsv = cv.cvtColor(frame_hist_eq, cv.COLOR_BGR2HSV) 
    lower_red = np.array([0,220,10]) 
    upper_red = np.array([255,255,255]) 

    # Here we are defining range of bluecolor in HSV 
    # This creates a mask of blue coloured  
    # objects found in the frame. 
    mask_red = cv.inRange(hsv, lower_red, upper_red) 

    # The bitwise and of the frame and mask is done so  
    # that only the blue coloured objects are highlighted  
    # and stored in res 
    res_red = cv.bitwise_and(frame_hist_eq,frame_hist_eq, mask = mask_red) 

    #print("RED OBJECTS")
#     display_image(mask_red)
#     plt.show()
#     display_image(res_red)
#     plt.show()

    points_done_app, reg_size_app = region_growing_objects(mask_red)

            
    # POINTS FILTERING
    all_points_filtered = []
    reg_size_filtered = []
    #print("%s objects have been detected" % len(points_done_app))

    # we filter objects with less than six points
    for j in range(len(points_done_app)):
        if len(points_done_app[j]) > 500:
            all_points_filtered.append(points_done_app[j])
            reg_size_filtered.append(reg_size_app[j])
            #print(reg_size_filtered)


    #print("After filtering, %s objects remain" % len(all_points_filtered))
    #print("\nSizes after filtering: %s" % reg_size_filtered)

    centroids = []
    for k in range(len(all_points_filtered)):   
        array = np.asarray(all_points_filtered[k])
        np.reshape(array, (2*len(all_points_filtered[k]), 1)) 
        centroids.append(centeroidnp(array))

    centroids_array = np.asarray(centroids)
    #print("Red centroids are %s" % centroids)
    
    head_position = min(min(all_points_filtered))
    
    return centroids, head_position
