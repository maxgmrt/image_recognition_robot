import cv2
import numpy as np

def histogram_equalization(frame):
    frame_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    frame_yuv[:,:,0] = cv2.equalizeHist(frame_yuv[:,:,0])

    # convert the YUV image back to RGB format
    frame_hist_eq = cv2.cvtColor(frame_yuv, cv2.COLOR_YUV2BGR)
    
    return frame_hist_eq


def mask_color(frame_hist_eq):
    # Converts images from BGR to HSV 
    hsv = cv2.cvtColor(frame_hist_eq, cv2.COLOR_BGR2HSV) 
    lower_blue = np.array([110,50,50]) 
    upper_blue = np.array([130,255,255]) 

    # Here we are defining range of bluecolor in HSV 
    # This creates a mask of blue coloured  
    # objects found in the frame. 
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue) 

    # The bitwise and of the frame and mask is done so  
    # that only the blue coloured objects are highlighted  
    # and stored in res 
    res_blue = cv2.bitwise_and(frame_hist_eq,frame_hist_eq, mask = mask_blue) 
    return mask_blue, res_blue

def mask_black_borders(frame):
    h, w = frame.shape[:2]
    mask_color = (255,255,255) # isolate a repeating constant

    left_mask = cv2.rectangle(frame, (0, 0), (int(0.15 * w), h), mask_color, -1)
    middle_mask = cv2.rectangle(frame, (int(0.45*w), 0), (int(0.48 * w), h), mask_color, -1)
    right_mask = cv2.rectangle(frame, (w, 0), (int(0.8 * w), h), mask_color, -1)
                               
def mask_arrow(frame, centroid):
    mask_color = (255,255,255)
    arrow_mask = cv2.rectangle(frame, (int(centroid[0][1] - 70), int(centroid[0][0] - 70)), \
                               (int(centroid[0][1] + 70), int(centroid[0][0] + 70)), mask_color, -1)
        
# calculate centroid from array of points coordinates
def centeroidnp(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length

def get_8_neigh(y, x, vert_lim, hor_lim):
    x_lim = hor_lim-1
    y_lim = vert_lim-1
    neigh = []
    
    #Get center_left neighborhhod
    coord_x = max(x-1,0)
    coord_y = y
    neigh.append((coord_y, coord_x))
    
    #Get top_left neighborhood
    coord_x = max(x-1,0)
    coord_y = max(y-1,0)
    neigh.append((coord_y, coord_x))
    
    #Get top_center neighborhood
    coord_x = x
    coord_y = max(y-1,0)
    neigh.append((coord_y, coord_x))

    #Get top_right neighborhood
    coord_x = min(x+1,x_lim)
    coord_y = max(y-1,0)
    neigh.append((coord_y, coord_x))
    
    #Get center_right neighborhood
    coord_x = min(x+1,x_lim)
    coord_y = y
    neigh.append((coord_y, coord_x))
    
    #Get bottom_right neighborhood
    coord_x = min(x+1,x_lim)
    coord_y = min(y+1,y_lim)
    neigh.append((coord_y, coord_x))  
    
    #Get bottom_center neighborhood
    coord_x = x
    coord_y = min(y+1,y_lim)
    neigh.append((coord_y, coord_x))
    
    #Get bottom_left neighborhood
    coord_x = max(x-1,0)
    coord_y = min(y+1,y_lim)
    neigh.append((coord_y, coord_x))
    
    return neigh

def region_growing_2(image, seed, threshold):
    region_size = 0
    neigh_list = []
    points_done = []
    new_image = np.zeros_like(image)
    neigh_list.append((seed[0],seed[1]))
    while(len(neigh_list)>0):
        current_pix = neigh_list[0]
        current_pix_value = image[current_pix]
        points_done.append(current_pix)
        for i in get_8_neigh(current_pix[0], current_pix[1], image.shape[0], image.shape[1]):
            if (abs(int(image[i[0], i[1]]) - int(current_pix_value)) < threshold and not i in points_done):
                new_image[i[0], i[1]] = 255
                neigh_list.append(i)
                points_done.append(i)
                region_size = region_size + 1
        neigh_list.pop(0)
    return new_image, region_size, points_done

def region_growing_objects(mask_blue):
    list_point_to_map = []
    reg_size_app = []
    points_done_app = []
    points = np.argwhere(mask_blue >250)

    list_point_to_map = points.tolist()

    while(len(list_point_to_map)>0):

            # Do the region growing for every point that belongs to an object
            SEED = list_point_to_map[0]
            grow_im, reg_size, list_points_done = region_growing_2(mask_blue, SEED, 0.1)

            # Convert list of tuple to list of list    
            list_points_done = [list(i) for i in list_points_done]

            # store size and points coordinates of objects
            reg_size_app.append(reg_size)
            points_done_app.append(list_points_done)

            # list_point_to_map - list_points_done : points that are not already processed
            list_difference = [item for item in list_point_to_map if item not in list_points_done]

            #We clear the list of point and we add the resting ones
            list_point_to_map.clear()
            for i in range(len(list_difference)):
                list_point_to_map.append(list_difference[i])
                
    return points_done_app, reg_size_app

def get_index_unwanted_objects(final_centroids):
    index = []
    for k in range(0, len(final_centroids)-1):
        if np.sqrt((final_centroids[k][0] - final_centroids[k+1][0])**2 + \
                   (final_centroids[k][1] - final_centroids[k+1][1])**2) < 50:

            index.append(k)
    return index