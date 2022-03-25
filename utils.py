#coding=utf-8
import cv2
import numpy as np
import math
import itertools
import matplotlib.pyplot as plt
from get_centroids_black import get_centroids_black

def display_image(mat, axes=None, cmap=None, hide_axis=True):
    """
    Display a given matrix into Jupyter's notebook
    
    :param mat: Matrix to display
    :param axes: Subplot on which to display the image
    :param cmap: Color scheme to use
    :param hide_axis: If `True` axis ticks will be hidden
    :return: Matplotlib handle
    """
    img = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB) if mat.ndim == 3 else mat
    cmap= cmap if mat.ndim != 2 or cmap is not None else 'gray'
    if axes is None:
        if hide_axis:
            plt.xticks([])
            plt.yticks([])
        return plt.imshow(img, cmap=cmap)
    else:
        if hide_axis:
            axes.set_xticks([])
            axes.set_yticks([])
        return axes.imshow(img, cmap=cmap)
    
    
def plot_confusion_matrix(cm, 
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          axes=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    
    :param cm: Confusion matrix
    :param classes: Classes name
    :param normalize: Indicate if the confusion matrix need to be normalized
    :param title: Plot's title
    :param axes: Subplot on which to display the image
    :param cmap: Colormap to use
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    # Show cm
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    #fmt = '.2f' if normalize else 'd'
    #thresh = cm.max() / 2.
    #for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #    plt.text(j, i, format(cm[i, j], fmt),
    #             horizontalalignment="center",
    #             color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
class ParticleFilterInterface:

    def __init__(self,
                 model,
                 search_space,
                 n_particles,
                 state_dims,
                 sigma_perturbation,
                 sigma_similarity,
                 alpha,
                 **kwargs):
        """
        Constrcutor
        :param model:         Template image to be tracked (numpy.ndarray)
        :param search_space:  Possible size of the search space (i.e. image size)
        :param n_particles:   Number of particle used to perform tracking
        :param state_dims:    State space dimensions (i.e. number of parameters
                              being tracked)
        :param sigma_perturbation: How much each particle will be perturbate (i.e. noise)
        :param alpha: Model adaptation, blending factor
        """
        # `Model` of the object being tracked (i.e. template)
        self.model = np.copy(model)
        # Range of search space (i.e. image dimensions)
        self.search_space = search_space
        # Number of particles used to estimate object position
        self.n_particles = n_particles
        # Number of state being estimated (will be equal to 2 => x, y)
        self.state_dims = state_dims
        # Magnitude of the perturbation added to the particles
        self.sigma_perturbation = sigma_perturbation
        # Similarity residual variance (i.e. eps = N(0, sigma_similarity ** 2.0))
        self.sigma_similarity = sigma_similarity
        # Blending coefficient for model adaptation
        self.alpha = alpha
        # Number of frames processed so far
        self.frame_counter = 0
        # 2D array particles stored as [[x0, y0], ..., [xN, yN]]
        self.particles = None
        # 1D array weights, probability of particles being at the real object
        # location
        self.weights = None
        # Current best estimation of the position, 1D array [x, y]
        self.state = None
        # Index of each particles
        self.indexes = np.arange(n_particles)
        # Toggle on/off sanity check, default is on
        self.verbose = kwargs.get('verbose', True)

    def current_state(self):
        """
        Select the current particles with the highest probability of being at the
        correct location of the object being tracked
        :return:  1D array, best state in form of [x, y] position
        """
        state_idx = np.random.choice(self.indexes, p=self.weights)
        return self.particles[state_idx, :]

    def perturbate(self):
        """ Perturbate particles by adding random normal noise """
        self.perturbate_impl()
        # Sanity check goes here
        if self.verbose:
          self._check_particle_dims()

    def perturbate_impl(self):
        """ Implementation of `Perturbate` function """
        raise NotImplementedError('Must be implemented by subclass')


    def reweight(self, frame):
        """
        Update particle's weight for the current frame. Check the similarity between
        every particles and the model of the tracked object.
        :param frame: New frame in the tracking sequence
        """
        self.reweight_impl(frame)
        if self.verbose:
            self._check_weights()

    def reweight_impl(self, frame):
        raise NotImplementedError('Must be implemented by sublass')

    def resample(self):
        """
        Draw a new set of particles using the probability distribution of the
        new weights.
        """
        self.resample_impl()
        if self.verbose:
            self._check_particle_dims(init=False)
            # Check particles is not outside image
            sh, sw = self.search_space[:2]
            msg = 'Particle larger than image width'
            assert np.any(self.particles[:, 0] < sw), msg
            msg = 'Particle larger than image height'
            assert np.any(self.particles[:, 1] < sh), msg

    def resample_impl(self):
        raise NotImplementedError('Must be implemented by sublclass')

    def update_model(self, frame):
        """
        Update tracking object model using current estimation and frame.
        :param frame: Current frame
        """
        shp = self.model.shape
        self.update_model_impl(frame)
        if self.verbose:
            msg = 'Update model must have same shape as before'
            assert self.model.shape == shp, msg

    def update_model_impl(self, frame):
        raise NotImplementedError('Must be implemented by sublclass')


    def draw_particles(self, image, color=(180, 255, 0)):
        """
        Draw current estimation of the tracked object by each individual particles
        :param image: Image to draw on
        :param color: Tuple, color to draw with
        :return: Updated image with particles draw on it
        """
        for p in self.particles:
            cv2.circle(image,
                       tuple(p.astype(int)),
                       radius=4,
                       color=color,
                       thickness=-1,
                       lineType=cv2.LINE_AA)
        return image

    def draw_window(self, image):
        """
        Draw current estimation of the tracked object using the best particles
        (i.e. The one with the highest probability)
        :param image: Image to draw on
        :return: Image with object position
        """
        best_idx = self.weights.argmax()
        best_state = self.state #self.particles[best_idx, :]
        pt1 = (best_state - np.asarray(self.model.shape[1::-1]) / 2).astype(np.int)
        #  pt1 = (self.state - np.array(self.model.shape[1::-1])/2).astype(np.int)
        pt2 = pt1 + np.asarray(self.model.shape[1::-1])
        cv2.rectangle(image,
                      tuple(pt1),
                      tuple(pt2),
                      color=(0, 255, 0), thickness=2,
                      lineType=cv2.LINE_AA)
        return image, pt1, pt2

    def draw_std(self, image):
        """
        Draw standard deviation between current state and all particles
        :param image: Canvas on which to draw
        :return:  Updated canvas
        """
        dist = np.linalg.norm(self.particles - self.state)
        weighted_sum = np.sum(dist * self.weights.reshape((-1, 1)))
        cv2.circle(image,
                   tuple(self.state.astype(np.int)),
                   int(weighted_sum),
                   (255, 255, 255),
                   1,
                   lineType=cv2.LINE_AA)
        return image

    def visualize_filter(self, image):
        """
        Visualize internal parts of the filter such as:
          - particles state
          - Current estimation of the object position (i.e. state)
          - Standard deviation between particles and estimation
          - Object's model
        :param image: Image to draw on
        :return:  Updated canvas
        """
        canvas = self.draw_particles(np.copy(image))
        canvas, pt1, pt2 = self.draw_window(canvas)
        canvas = self.draw_std(canvas)
        # Add model in the top left corner
        canvas[:self.model.shape[0], :self.model.shape[1]] = self.model
        return canvas, pt1, pt2

    def _check_particle_dims(self, init=False):
        msg = 'particles dimensions must be (n_particle, state_dims)'
        assert self.particles.shape == (self.n_particles, self.state_dims), msg
        if init:
            p_max = self.particles.max(axis=0)
            msg = 'particle state 0 must be smaller than ' \
                  'search space: {}'.format(self.search_space[1])
            assert p_max[0] < self.search_space[1], msg
            msg = 'particle state 0 must be smaller than ' \
                  'search space: {}'.format(self.search_space[0])
            assert p_max[1] < self.search_space[0], msg

    def _check_weights(self):
          msg = 'weights dimensions must be equal to the number of particles, (n,)'
          assert self.weights.shape == (self.n_particles,), msg
          # Must be valid probability distribution
          assert np.abs(self.weights.sum() - 1.0) < 1e-6, 'Weight must sum to 1.0'
    

_tracker_ctor = {'mil': cv2.TrackerMIL_create,
                 'kcf': cv2.TrackerKCF_create,
                 'tld': cv2.TrackerTLD_create,
                 'medianflow': cv2.TrackerMedianFlow_create,
                 'mosse': cv2.TrackerMOSSE_create,
                 'goturn': cv2.TrackerGOTURN_create}
    
def create_face_tracker(name='KCF'):
    """
    Create an instance of a face tracker from a given `name`. The list of available tracker is :
    
    ['MIL', 'KCF', 'TLD', 'MedianFlow', 'Mosse', 'GoTurn']
    
    :param name: Name of the tracker instance to create
    :raise: ValueError exception if the name of the tracker do not match anything known.
    """
    n = name.lower()
    ctor = _tracker_ctor.get(n, None)
    if ctor is None:
        raise ValueError('Unknown type of tracker')
    return ctor()

def transform_video_file(file_path, function, params=None, n_frame=-1):
    """
    Given the path of a video file (file_path) the function reads every frame of the input video and applies 
    a given transformation (function) using the parameters (params)

    :input_image:       Input video file path 
    :function:          Function be applied to each frame of the image. Signature `function(np.array, Any) -> Any`
    :params:            Any parameter needed for the function above.
    :n_frame:           Maxiumum number of frame to read. Default `-1`, read all the content
    :return:            output_handler this can be anything you may need to save your results.
    """
    output_handler = []
    pt_1 = []
    pt_2 = []
    # Open video
    cap = cv2.VideoCapture(file_path)
    if cap.isOpened():
        # YOUR CODE HERE
        #################################
        
        # Solution
        n_comp_frames = 0
        while(True):
            # Check if we computed the desired number of frame (we skip this part if we need all the frames)
            if(n_frame != -1):
                if(n_comp_frames >= n_frame):
                    break
            # Read the frame
            ret, frame = cap.read()
            # If end of the video or no frame -> quit
            if(ret==False):
                break
            # Apply the function
            #output_handler.append(function(frame,params))
            img, pt1, pt2 = function(frame,params)
            output_handler.append(img)
            pt_1.append(pt1)
            pt_2.append(pt2)
            
            #display_image(img)
            #plt.show()
            #print(pt1)
            #print(pt2)
            #test = cv.cvtColor(output_handler[n_comp_frames].copy(), cv.COLOR_BGR2RGB)
            #plt.imshow(frame)
            #plt.show()
            #plt.imshow(test)
            #plt.show()
            # Increment number of computed frames
            n_comp_frames += 1
        #################################     
        
    else:
        raise ValueError('Can not open file: {}'.format(file_path))
    # Close reader
    cap.release()
    # Return custom structure
    return output_handler, pt_1, pt_2

class ParticleFilterStepOne(ParticleFilterInterface):
    
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
        :param sigma_similarity:  Similarity residual %distribution
        :param alpha:         Blending factor for model adaptation, float
        :param update_rate:   Frequency at which the model will be updated, int
        """
        super(ParticleFilterStepOne, self).__init__(model,
                                                    search_space,
                                                    n_particles,
                                                    state_dims,
                                                    sigma_perturbation,
                                                    sigma_similarity,
                                                    alpha)
        self.state = None
        self.update_rate = update_rate
        self._initialize_particles()
        self._check_particle_dims(init=True)
        self._check_weights()
        
    def _initialize_particles(self):
        """
        Initialize each particle state by sampling uniformly the whole search space. Plus the each weight
        associated to a particles is initialized with a uniform probability:
        """
        
        # YOUR CODE HERE
        
        self.particles = np.zeros((self.n_particles,self.state_dims))
        self.particles[:,0] = np.random.uniform(0, self.search_space[1], self.n_particles)
        self.particles[:,1] = np.random.uniform(0, self.search_space[0], self.n_particles)
        self.weights = np.ones(self.n_particles)/ self.n_particles
    
        # Solution
        ########################

        
    def update(self, frame):
        """
        Update particle filter for a new frame
        :param frame: New frame in the tracking sequence
        :return:  Best state (i.e. estimated position
        """
        self.perturbate()
        # Other steps goes here
        # ...
        
        self.state = self.current_state()
        return self.state
    
    def perturbate_impl(self):
        """ Perturbate particles with random noise """
        
        # YOUR CODE HERE
        mu, sigma, size = 0, self.sigma_perturbation,self.n_particles
        #perturbation = np.random.normal(mu, sigma, size)
        #print(perturbation)
        #for i in self.indexes :
            #print(self.particles[i],self.particles[i]+np.random.normal(mu, sigma, size) )
        self.particles[:,0] += np.random.normal(mu, sigma, size)
        self.particles[:,1] += np.random.normal(mu, sigma, size)
        


def similarity_fn(x, y, sigma):
    """
    Compute similarity between `x` and `y` assuming the residual difference
    follow a gaussian distribution N(0, sigma).
    Special case, when the shape of `x`/`y` are not the same/empty it must return a similarity of 0.0
    :param x: First image
    :param y: Second image to compare
    :param sigma: Standard deviation of the residual differences
    :return: Scalar value, similarity between `x` and `y`
    """

    # YOUR CODE HERE
    if (x.size != y.size)  or x.size == 0 or y.size == 0:
        return 0.0
    
    error = x.flatten().astype(np.float64)-y.flatten().astype(np.float64)
    r = np.square(error).mean()
    s = float(np.exp(-r/(2*sigma**2)))

    return s



def particles_filter_fn(frame, params):
    """
    Run one step of particles filter and draw tracking results on top of the image
    :param params: dict instance with the following entries: 
                    'tracker': Tracker instance
    """
    # YOUR CODE HERE
    params['tracker'].update(frame)
    img, pt1, pt2 = params['tracker'].visualize_filter(frame)
    return img, pt1, pt2
   
# Function that receives a string
# Compute the equation
# Return the result as a chr
def eq_solver(eq_chr):
    eq = eq_chr[10:]
    x = eval(eq)
    res = str(x)
    
    return res

# Function that generate the equation to resolve depending on the 
# position of each digit/operator and the trajectory of the tracker (list of points):
def eq_maker(points, oper_pos, oper_label, digit_pos, digit_label, threshold):
    dist_oper = np.zeros([np.shape(oper_pos)[0],1])
    dist_digit = np.zeros([np.shape(digit_pos)[0],1])
    eq_list = []
    eq = 'equation: '

    # For each frame:
    for i in range(np.shape(points)[0]):
        
        # Compute the distance between the actual point and the operands and digits:
        for j in range(np.shape(oper_pos)[0]):
            dist_oper[j] = dist_points(oper_pos[j][0], oper_pos[j][1], points[i][0], points[i][1])
        for k in range(np.shape(digit_pos)[0]):
            dist_digit[k] = dist_points(digit_pos[k][0], digit_pos[k][1], points[i][0], points[i][1])

        # Take the minimum value and their corresponding index:
        min_dist_oper = min(dist_oper)
        idx_oper = np.argmin(dist_oper)
        min_dist_digit = min(dist_digit)
        idx_digit = np.argmin(dist_digit)
        
        # Check if the distance is below the threshold:
        if (min_dist_oper < threshold) or (min_dist_digit < threshold):
            
            # Check which is the minimum:
            # If operand is the nearest:
            if(min_dist_oper < min_dist_digit):
                # Determine the associated operand:
                if(oper_label[idx_oper][0] == 1):
                    label = chr(43)
                elif(oper_label[idx_oper][1] == 1):
                    label = chr(45)
                elif(oper_label[idx_oper][2] == 1):
                    label = chr(42)
                elif(oper_label[idx_oper][3] == 1):
                    label = chr(47)
                # Equal sign -> do the equation solving
                else:
                    label = chr(61)
                    #Find the result
                    res_eq = eq_solver(eq)
                    #Add the result to the eq
                    eq = eq + label
                    eq = eq + res_eq
                    #Fill the current and resting frame-eq with the current state (final state)
                    for t in range(i,np.shape(points)[0]):
                        eq_list.append(eq)
                    #Return the eq
                    return eq_list
                    
                # Check if we already computed the associated operand (last element of the string):
                if(label != eq[-1]):
                    # New element to add to the equation
                    eq = eq + label
            
            # If digits is the nearest:
            else:
                # Determine the associated digits:
                if(digit_label[idx_digit][0] == 1):
                    label = chr(48)
                elif(digit_label[idx_digit][1] == 1):
                    label = chr(49)
                elif(digit_label[idx_digit][2] == 1):
                    label = chr(50)
                elif(digit_label[idx_digit][3] == 1):
                    label = chr(51)
                elif(digit_label[idx_digit][4] == 1):
                    label = chr(52)
                elif(digit_label[idx_digit][5] == 1):
                    label = chr(53)
                elif(digit_label[idx_digit][6] == 1):
                    label = chr(54)
                elif(digit_label[idx_digit][7] == 1):
                    label = chr(55)
                elif(digit_label[idx_digit][8] == 1):
                    label = chr(56)
                else:
                    label = chr(57)
                    
                # Check if we already computed the associated operand (last element of the string):
                if(label != eq[-1]):
                    # New element to add to the equation
                    eq = eq + label
            
            
            
            
        # Add a the current state of the equation:
        eq_list.append(eq)
    return eq_list

# Function that compute the euclidean distance between 2 points
def dist_points(x1, y1, x2, y2):
    return math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )

def get_digit_images(frame):
    
    black_centroids, all_black_points = get_centroids_black(frame)

    black_centroids_array = np.asarray(black_centroids)

    #print("\nObjects centroids:")
    #print(black_centroids_array)
    
    rescaled_digits = np.zeros([len(all_black_points), 28, 28])
    rescaled_digits_border = np.zeros([len(all_black_points), 38, 38])
    objects = []
    for f in range(len(all_black_points)):
        objects.append(all_black_points[f])

        x_values = []
        y_values = []

        for l in range(len(all_black_points[f])):
            y_values.append(objects[f][l][0])
            x_values.append(objects[f][l][1])

        shift_x = min(x_values)
        shift_y = min(y_values)

        for l in range(len(all_black_points[f])):
            objects[f][l][0] = objects[f][l][0] - shift_y
            objects[f][l][1] = objects[f][l][1] - shift_x

        x_values_array = np.asarray(x_values)
        y_values_array = np.asarray(y_values)

        height = max(x_values_array - shift_x+1)
        width = max(y_values_array - shift_y+1)

        #print("Width is %s" % width)
        #print("Height is %s" % height)
        longer_side = max(width, height)

        digit_images = np.zeros([len(all_black_points), width, height], dtype=np.uint8)

        for l in range(len(all_black_points[f])):
            x = objects[f][l][0]
            y = objects[f][l][1]
            digit_images[f][x][y] = 1

        np.savetxt("image%s.csv" % f, digit_images[f])
        from matplotlib import pyplot as plt
        

#         rescaled_digits[f] = cv2.resize(digit_images[f], dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
#         plt.imshow(rescaled_digits[f], interpolation='nearest')
#         plt.show()
        
        color = [0, 0, 0]
        top, bottom, left, right = [5]*4
        rescaled_digits_border = cv2.copyMakeBorder(digit_images[f], \
                                             top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        rescaled_digits[f] = cv2.resize(rescaled_digits_border, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
#         rescaled_digits[f] = cv2.cvtColor(rescaled_digits[f], cv2.COLOR_BGR2GRAY)


        #plt.imshow(rescaled_digits[f], interpolation='nearest')
        #plt.show()
    return rescaled_digits, black_centroids_array