import cv2 as cv
import numpy as np
import os
import gzip
import math
import itertools
import tarfile
import time as _time
import utils
from utils import *
import utils_max
from utils_max import *
import csv
import optparse
import get_centroids_black
import get_head_red_arrow
import train_model
from get_head_red_arrow import get_head_red_arrow
from get_centroids_blue import get_centroids_blue_operators
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from joblib import dump, load
import matplotlib.pyplot as plt


############################ Instantiate usefull fonctions ####################################

class ParticleFilterStepTwo(ParticleFilterStepOne):
    
    def __init__(self,
                 model,
                 search_space,
                 n_particles,
                 state_dims,
                 sigma_perturbation=10.0,
                 sigma_similarity=15.0,
                 alpha=0.0,
                 update_rate=10):
        """
        Constructor
        :param model:         Template image to be tracked (numpy.ndarray)
        :param search_space:  Possible size of the search space (i.e. image size)
        :param n_particles:   Number of particle used to perform tracking
        :param state_dims:    State space dimensions (i.e. number of parameters
                              being tracked)
        :param sigma_perturbation: How much each particle will be perturbated, float
        :param sigma_similarity:  Similarity residual distribution
        :param alpha:         Blending factor for model adaptation, float
        :param update_rate:   Frequency at which the model will be updated, int
        """
        super(ParticleFilterStepTwo, self).__init__(model,
                                                    search_space,
                                                    n_particles,
                                                    state_dims,
                                                    sigma_perturbation,
                                                    sigma_similarity,
                                                    alpha)
        
    def update(self, frame):
        """
        Update particle filter for a new frame
        :param frame: New frame in the tracking sequence
        :return:  Best state (i.e. estimated position
        """
        self.perturbate()
        self.reweight(frame)
        # Other steps goes here
        # ...
        
        self.state = self.current_state()
        return self.state
    
    def reweight_impl(self, frame):
        """
        Update particle's weight for the current frame
        :param frame: New frame in the tracking sequence
        """
        
        # YOUR CODE HERE
        sim = np.ones(self.n_particles)
    
        for i in self.indexes:
            x_lim = (int(self.particles[i][0]-model.shape[0]/2),int(self.particles[i][0]+model.shape[0]/2))
            y_lim = (int(self.particles[i][1]-model.shape[1]/2),int(self.particles[i][1]+model.shape[1]/2))
            patch = frame[y_lim[0]:y_lim[1],x_lim[0]:x_lim[1]]
            sim[i] = similarity_fn(patch, model, sigma=self.sigma_similarity)
            
        self.weights = sim/sim.sum()

class ParticleFilterStepThree(ParticleFilterStepTwo):
    
    def __init__(self,
                 model,
                 search_space,
                 n_particles,
                 state_dims,
                 sigma_perturbation=15.0,
                 sigma_similarity=15.0,
                 alpha=0.0,
                 update_rate=10):
        """
        Constructor
        :param model:         Template image to be tracked (numpy.ndarray)
        :param search_space:  Possible size of the search space (i.e. image size)
        :param n_particles:   Number of particle used to perform tracking
        :param state_dims:    State space dimensions (i.e. number of parameters
                              being tracked)
        :param sigma_perturbation: How much each particle will be perturbated, float
        :param sigma_similarity:  Similarity residual distribution
        :param alpha:         Blending factor for model adaptation, float
        :param update_rate:   Frequency at which the model will be updated, int
        """
        super(ParticleFilterStepTwo, self).__init__(model,
                                                    search_space,
                                                    n_particles,
                                                    state_dims,
                                                    sigma_perturbation,
                                                    sigma_similarity,
                                                    alpha)
        
    def update(self, frame):
        """
        Update particle filter for a new frame
        :param frame: New frame in the tracking sequence
        :return:  Best state (i.e. estimated position
        """
        self.perturbate()
        self.reweight(frame)
        self.resample()
        # Other steps goes here
        # ...
        
        self.state = self.current_state()
        return self.state
    
    def resample_impl(self):
        """ 
        Resample a new set of particle based on update weight distribution and previously 
        perturbated particles
        Hint: You may want to use `np.random.choice` here
        """
        new_index = np.random.choice(self.indexes,self.n_particles, p=self.weights)
        self.particles = self.particles[new_index]
        
class ParticleFilterStepFour(ParticleFilterStepThree):
    
    def __init__(self,
                 model,
                 search_space,
                 n_particles,
                 state_dims,
                 sigma_perturbation=10.0,
                 sigma_similarity=20.0,
                 alpha=0.0,
                 update_rate=10):
        """
        Constructor
        :param model:         Template image to be tracked (numpy.ndarray)
        :param search_space:  Possible size of the search space (i.e. image size)
        :param n_particles:   Number of particle used to perform tracking
        :param state_dims:    State space dimensions (i.e. number of parameters
                              being tracked)
        :param sigma_perturbation: How much each particle will be perturbated, float
        :param sigma_similarity:  Similarity residual distribution
        :param alpha:         Blending factor for model adaptation, float
        :param update_rate:   Frequency at which the model will be updated, int
        """
        super(ParticleFilterStepTwo, self).__init__(model,
                                                    search_space,
                                                    n_particles,
                                                    state_dims,
                                                    sigma_perturbation,
                                                    sigma_similarity,
                                                    alpha)
        
    def update(self, frame):
        """
        Update particle filter for a new frame
        :param frame: New frame in the tracking sequence
        :return:  Best state (i.e. estimated position
        """
        self.perturbate()
        self.reweight(frame)
        self.resample()
        self.state = self.current_state()
        self.frame_counter += 1
        if self.alpha > 0.0 and (self.frame_counter % self.update_rate) == 0:
            self.update_model(frame)
        return self.state
    
    def update_model(self, frame):
        """
        This function perform a `model update using the current best estimation of
        the object position (i.e. state) and linearly blend the previous model and
        the patch at the best state:
          model_new = alpha * best_patch + (1 - alpha) * model_old
        It also perform so sanity check to ensure the dimensions of the model are
        valid.
        :param frame: Current frame
        """
        # YOUR CODE HERE
        x_lim = (int(self.state[0]-model.shape[0]/2),int(self.state[0]+model.shape[0]/2))
        y_lim = (int(self.state[1]-model.shape[1]/2),int(self.state[1]+model.shape[1]/2))
        patch = frame[x_lim[0]:x_lim[1],y_lim[0]:y_lim[1]]
        self.model = self.alpha*patch+(1-self.alpha)*self.model

        
################################# START OF THE MAIN ###########################

# Define parameters
parser = optparse.OptionParser()
parser.add_option('--input', help='arguments', dest='infile', action='store')
parser.add_option('--output', help='arguments', dest='outfile', action='store')
(opts, args) = parser.parse_args()
#print(opts.infile) 
file_name = opts.infile
out_video_path = opts.outfile
#out_video_path = '../data/videos/outpy.avi'
#file_name = '../data/videos/robot_parcours_1.avi'








############################ IMAGE PRE-PROCESSING ########################

#Get the first frame:
cap = cv.VideoCapture(file_name)
if cap.isOpened():
    im_width = cap.get(3)
    im_height = cap.get(4)
    nb_frames =cap.get(7)
    # Read the first frame
    ret, frame = cap.read()

else:
    raise ValueError('Can not open file: {}'.format(file_path))
    
# Close reader
cap.release()

# HISTOGRAM EQUALIZATION
frame_hist_eq = frame
frame_hist_eq = histogram_equalization(frame)



height,width,depth = frame.shape
#print("First frame has dimensions: %d x %d" % (width,height))

    
#print("\nMasking black borders !")
mask_black_borders(frame)
#display_image(frame)
#plt.show()

############################ IMAGE SEGMENTATION ##################################

# FIND THE HEAD POSITION OF THE ARROW
red_centroid, red_head = get_head_red_arrow(frame_hist_eq)
red_centroid_array = np.asarray(red_centroid)

mask_arrow(frame, red_centroid_array)



############################ FIND THE CENTER OF EACH DIGITS -> IMAGE CLASSIFICATION ##################################### 
if (os.path.exists('../code/model_mlp.joblib')):
    mlp = load('model_mlp.joblib')
else:
    train_and_save_model()
    
rescaled_digits, black_centroids_array = get_digit_images(frame)
reshaped_digits = np.zeros([len(rescaled_digits),1,784])
resu = np.zeros([len(rescaled_digits),1])
#print(reshaped_digits.shape)
for f in range(len(rescaled_digits)):
    #plt.imshow(rescaled_digits[f])
    #plt.show()

    reshaped_digits[f] = np.reshape(rescaled_digits[f], (1,784), order='C')

    reshaped_digits[f] = reshaped_digits[f] * 255
    
    resu[f] = mlp.predict(reshaped_digits[f])
    


############################ FIND THE CENTER OF EACH OPERATORS ################################### 
centroid_multiplication, centroid_addition, centroid_division, centroid_equality, centroid_substraction = get_centroids_blue_operators(frame_hist_eq)


#################### INSTANTIATE THE TRACKER AND TRACK THE ARROW #########################

#Get the first frame:
cap = cv.VideoCapture(file_name)
if cap.isOpened():
    im_width = cap.get(3)
    im_height = cap.get(4)
    nb_frames =cap.get(7)
    # Read the first frame
    ret, first_frame = cap.read()

else:
    raise ValueError('Can not open file: {}'.format(file_path))
    
# Close reader
cap.release()

model = first_frame[red_head[0]-2:red_head[0]+18,red_head[1]-6:red_head[1]+14]
#display_image(model)
#plt.show()

# # Create particle filter
pfhand = ParticleFilterStepFour(model=model,
                              search_space=(int(im_height), int(im_width)),
                              n_particles=1000,
                              sigma_perturbation=10.0,
                              sigma_similarity=10.0,
                              state_dims=2,
                              alpha=0.07,
                              update_rate=20)


# Call tracking function on video
params = {'tracker': pfhand}
results, pt1, pt2 = transform_video_file(file_name, particles_filter_fn, params=params)




# Display results of the particules filter:
#for idx,out in enumerate(results):
#    if idx % 1 == 0:
#        print('Frame number: {}'.format(idx))
#        display_image(out)
#        plt.show()




############################ Background of the final video ###################################
# Without the arrow, the borders and the vertical line

im = frame
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 110, 255, 0)

background = np.dstack([thresh, thresh, thresh]).astype(np.uint8)
#display_image(background)
#plt.show()

############## Digit and operands handling ###############

nb_oper = 0

if(centroid_multiplication.size != 0):
    nb_oper += 1
if(centroid_addition.size != 0):
    nb_oper += 1
if(len(centroid_division) != 0):
    nb_oper += 1
if(len(centroid_equality) != 0):
    nb_oper += 1
if(len(centroid_substraction) != 0):
    nb_oper += 1

oper_pos = np.zeros([nb_oper,2])
oper_label = np.zeros([nb_oper,5])
digit_pos = np.zeros([len(black_centroids_array),2])
digit_label = np.zeros([len(black_centroids_array),10])

index = 0

if(centroid_multiplication.size != 0):
    oper_pos[index,:] = int(centroid_multiplication[0]), int(centroid_multiplication[1])
    oper_label[index,:] = 0, 0, 1, 0, 0
    index += 1
if(centroid_addition.size != 0):
    oper_pos[index,:] = int(centroid_addition[0]), int(centroid_addition[1])
    oper_label[index,:] = 1, 0, 0, 0, 0
    index += 1
if(len(centroid_division) != 0):
    oper_pos[index,:] = int(centroid_division[0]), int(centroid_division[1])
    oper_label[index,:] = 0, 0, 0, 1, 0
    index += 1
if(len(centroid_equality) != 0):
    oper_pos[index,:] = int(centroid_equality[0]), int(centroid_equality[1])
    oper_label[index,:] = 0, 0, 0, 0, 1
    index += 1
if(len(centroid_substraction) != 0):
    oper_pos[index,:] = int(centroid_substraction[0]), int(centroid_substraction[1])
    oper_label[index,:] = 0, 1, 0, 0, 0
    index += 1

for i in range(len(black_centroids_array)):
    digit_pos[i,:] = int(black_centroids_array[i][0]), int(black_centroids_array[i][1])
    if(resu[i] == 0):
        digit_label[i,:] = 1, 0, 0, 0, 0, 0, 0, 0, 0, 0
    elif(resu[i] == 1):
        digit_label[i,:] = 0, 1, 0, 0, 0, 0, 0, 0, 0, 0
    elif(resu[i] == 2):
        digit_label[i,:] = 0, 0, 1, 0, 0, 0, 0, 0, 0, 0
    elif(resu[i] == 3):
        digit_label[i,:] = 0, 0, 0, 1, 0, 0, 0, 0, 0, 0
    elif(resu[i] == 4):
        digit_label[i,:] = 0, 0, 0, 0, 1, 0, 0, 0, 0, 0
    elif(resu[i] == 5):
        digit_label[i,:] = 0, 0, 0, 0, 0, 1, 0, 0, 0, 0
    elif(resu[i] == 6):
        digit_label[i,:] = 0, 0, 0, 0, 0, 0, 1, 0, 0, 0
    elif(resu[i] == 7):
        digit_label[i,:] = 0, 0, 0, 0, 0, 0, 0, 1, 0, 0
    elif(resu[i] == 8):
        digit_label[i,:] = 0, 0, 0, 0, 0, 0, 0, 0, 1, 0  
    else:
        digit_label[i,:] = 0, 0, 0, 0, 0, 0, 0, 0, 0, 1

        
        
        
        
############################ Write video: #################################### 


# Parameters of the text of the corresponding equation:

# font 
font = cv2.FONT_HERSHEY_SIMPLEX  
# org 
org = (10, 430)   
# fontScale 
fontScale = 1 
# White color in BGR 
color = (70, 70, 255) 
# Line thickness of 2 px 
thickness = 1
# Beginning of the sentence:
text_init = 'Frame num '

        
# Points of the trajectory of the robot:
# Convert points to np.array (opencv)
points = np.zeros([int(nb_frames),2])
for i in range(int(nb_frames)):
    points[i,0] = pt1[i][1]
    points[i,1] = pt1[i][0]

threshold = 30
# Generate the equation and resolve it -> returns it as a text
eq_list_ex = eq_maker(points, oper_pos, oper_label, digit_pos, digit_label, threshold)

# Instantiate the video writer
#out = cv.VideoWriter('outpy.avi',cv.VideoWriter_fourcc('D','I','V','X'), 2, (int(im_width),int(im_height)))
out = cv.VideoWriter(out_video_path,cv.VideoWriter_fourcc('D','I','V','X'), 2, (int(im_width),int(im_height)))

# For each new frame, plot the trajectory from the start:
for i in range(int(nb_frames)-1):
    inter = np.empty_like(background)
    inter = background.copy()
    cv.circle(inter,(int(points[0][1]),int(points[0][0])),10,(0, 0, 255))
    cv.polylines(inter, np.int32([pt1[:i+1]]), False, (0, 0, 255), 2 )
    if eq_list_ex[i]:
        cv.putText(inter, text_init + str(i+1) + ' ' + eq_list_ex[i], org, font, fontScale, color, thickness, cv2.LINE_AA)
    # Write the frame into the file 'output.avi'
    out.write(inter)


# When everything done, release the video capture and video write objects
out.release()
# Closes all the frames
cv2.destroyAllWindows()

        
        
        