import utils_max
import cv2
import numpy as np
import utils_max
from utils_max import *
import csv


def get_centroids_black(frame):
    ret,thresh1 = cv2.threshold(frame,80,255,cv2.THRESH_BINARY)

    #display_image(thresh1)
    #plt.show('gray')

    # ##############
    # # BLACK MASK #
    # ##############

    # Converts images from BGR to HSV 
    hsv = cv2.cvtColor(thresh1, cv2.COLOR_BGR2HSV) 
    lower_black = np.array([0,0,0]) 
    upper_black = np.array([0,0,10]) 

    # Here we are defining range of bluecolor in HSV 
    # This creates a mask of blue coloured  
    # objects found in the frame. 
    mask_black = cv2.inRange(hsv, lower_black, upper_black) 

    # The bitwise and of the frame and mask is done so  
    # that only the blue coloured objects are highlighted  
    # and stored in res 
    res_black = cv2.bitwise_and(thresh1, thresh1, mask = mask_black) 

    #print("\nBLACK OBJECTS")
    #display_image(mask_black)
    #plt.show()
    #display_image(res_black)
    #plt.show()


    # REGION GROWING
    points_done_app = []
    reg_size_app = []

    points_done_app, reg_size_app = region_growing_objects(mask_black)


    # POINTS FILTERING
    all_black_points_filtered = []
    reg_size_black_filtered = []
    #print("%s objects have been detected" % len(points_done_app))

    # we filter objects with less than six points
    for j in range(len(points_done_app)):
        if (len(points_done_app[j]) > 30) & (len(points_done_app[j]) < 400):
            all_black_points_filtered.append(points_done_app[j])
            reg_size_black_filtered.append(reg_size_app[j])


        with open('black_points.csv', 'w', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerows(all_black_points_filtered)        

    #print("After filtering, %s objects remain" % len(all_black_points_filtered))
    #print("\nSizes after filtering: %s" % reg_size_black_filtered)

    centroids = []
    for k in range(len(all_black_points_filtered)):   
        array = np.asarray(all_black_points_filtered[k])
        np.reshape(array, (2*len(all_black_points_filtered[k]), 1)) 
        centroids.append(centeroidnp(array))

    return centroids, all_black_points_filtered