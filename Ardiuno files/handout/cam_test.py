from pickle import TRUE
from re import I, search
from turtle import right
#import robot
import sys
import numpy as np
import cv2
from time import sleep 
import camera




# Flags
showGUI = True  # Whether or not to open GUI windows
onRobot = True # Whether or not we are running on the Arlo robot

#======== Udregninger fra opgave teksten ===========
def el(lx,ly,x,y): # e_l = (l_x - x, l_y - y)
    d = distance(lx,ly,x,y)
    return np.transpose([lx-x,ly-y])/d

def distance(lx,ly,x,y): # Distance from particle to landmark
    result = np.sqrt(((lx-x)**2)+((ly-y)**2))
    return result#Distance from particle (x,y) to landmark (lx,ly)

def e_theta(theta): 
    return np.transpose((np.cos(theta),np.sin(theta)))

def e_theta_hat(theta):
    return np.transpose((-np.sin(theta),-np.cos(theta)))

def particle_angle(lx,ly,x,y,theta): # Angle from orientation angle (For particle) 
    return np.sign(np.dot(el(lx,ly,x,y),e_theta_hat(theta))*np.arccos(np.dot(el(lx,ly,x,y),e_theta(theta))))

def gaussian_pdf_distance(d,dm,stdd): # Se opgave tekst
                return ((1.0/math.sqrt(2.0*math.pi*stdd**2.0)))*(np.exp(-(((dm-d)**2)/(2.0*stdd**2.0))))

def gaussian_pdf_angle(m_angle,lx,ly,x,y,theta,stdd): # Se opgave tekst
                return (1.0/math.sqrt(2.0*math.pi*stdd**2.0))*np.exp(-(((m_angle-particle_angle(lx,ly,x,y,theta))**2.0)/(2.0*stdd**2.0)))


def compute_weights(landmarkIDs,landmark_d, landmark_a ,old_particles): # Computes weights for particles
    for op in old_particles:
        weight = 1
        for i in range(len(landmarkIDs)):
            d = distance(landmarks[landmarkIDs[i]][0],landmarks[landmarkIDs[i]][1],op.getX(),op.getY()) #hypo distance
            dm = landmark_d[i]
            gpdfd = gaussian_pdf_distance(d,dm,5) 
            gpdfa = gaussian_pdf_angle(landmark_a[i],landmarks[landmarkIDs[i]][0],landmarks[landmarkIDs[i]][1],op.getX(),op.getY(),op.getTheta(),0.1)
            weight = weight * gpdfd  * gpdfa 
        op.setWeight(weight) 
      

def normalize_weights(particles): # Normalizes the computed weights
    sum = 0
    for p in particles:
        sum += p.getWeight()
    for p2 in particles:
        p2.setWeight(p2.getWeight()/sum)

def resample_gaussian(particles): # Resample new particles (NORMAL)
    weights = []
    for p in particles:
        weights.append(p.getWeight())
    print("sum:",sum(weights))
    #resamples = np.random.choice(particles,10000,p=weights,replace=True)
    temp = random.choices(particles, weights, k = len(particles))
    for i in range(len(temp)):
        particles[i] = copy.copy(temp[i])





def sample_motion_model_velocity_withT(particle,v,w,delta_t): # See page 124 in the book
    x = particle.getX()
    y = particle.getY()
    theta = particle.getTheta()
    v_hat = v + randn(0,(1.2*(v**2))+(0.05*(w**2))) #Velocity with noise
   # print("v_hat:",v_hat)
    w_hat = w + randn(0,(1.2*(v**2))+(0.05*(w**2))) # angular velocity with noise
   #print("w_hat:",w_hat)
    epsilon = randn(0,1.2*(v**2)+0.05*(w**2)) # Random term
    new_x = x - (v_hat/w_hat)*np.sin(theta) + (v_hat/w_hat)*np.sin(theta + (w_hat*delta_t)) 
    new_y = y + (v_hat/w_hat)*np.cos(theta) - (v_hat/w_hat)*np.cos(theta + (w_hat*delta_t))
    new_theta = theta + w_hat*delta_t + epsilon*delta_t
    particle.setX(new_x)
    particle.setY(new_y)
    particle.setTheta(new_theta)


def Turn(angle): #Turns the robot depending on given angle
    if angle < 0:
        arlo.go_diff(30,30,0,1)
        sleep(0.0153*abs(angle))
        arlo.stop()
    else:
        arlo.go_diff(30,30,1,0)
        sleep(0.0153*abs(angle))
        arlo.stop()

def isRunningOnArlo():
    """Return True if we are running on Arlo, otherwise False.
      You can use this flag to switch the code from running on you laptop to Arlo - you need to do the programming here!
    """
    return onRobot


if isRunningOnArlo():
    # XXX: You need to change this path to point to where your robot.py file is located
    sys.path.append("../robot.py ")



try:
    import robot
    onRobot = True
except ImportError:
    print("selflocalize.py: robot module not present - forcing not running on Arlo!")
    onRobot = False




# Some color constants in BGR format
CRED = (0, 0, 255)
CGREEN = (0, 255, 0)
CBLUE = (255, 0, 0)
CCYAN = (255, 255, 0)
CYELLOW = (0, 255, 255)
CMAGENTA = (255, 0, 255)
CWHITE = (255, 255, 255)
CBLACK = (0, 0, 0)

# Landmarks.
# The robot knows the position of 2 landmarks. Their coordinates are in the unit centimeters [cm].
landmarkIDs = [2, 4]
landmarks = {
    2: (0.0, 0.0),  # Coordinates for landmark 1
    4: (300.0, 0.0)  # Coordinates for landmark 2
}
landmark_colors = [CRED, CGREEN] # Colors used when drawing the landmarks



def jet(x):
    """Colour map for drawing particles. This function determines the colour of 
    a particle from its weight."""
    r = (x >= 3.0/8.0 and x < 5.0/8.0) * (4.0 * x - 3.0/2.0) + (x >= 5.0/8.0 and x < 7.0/8.0) + (x >= 7.0/8.0) * (-4.0 * x + 9.0/2.0)
    g = (x >= 1.0/8.0 and x < 3.0/8.0) * (4.0 * x - 1.0/2.0) + (x >= 3.0/8.0 and x < 5.0/8.0) + (x >= 5.0/8.0 and x < 7.0/8.0) * (-4.0 * x + 7.0/2.0)
    b = (x < 1.0/8.0) * (4.0 * x + 1.0/2.0) + (x >= 1.0/8.0 and x < 3.0/8.0) + (x >= 3.0/8.0 and x < 5.0/8.0) * (-4.0 * x + 5.0/2.0)

    return (255.0*r, 255.0*g, 255.0*b)

def draw_world(est_pose, particles, world):
    """Visualization.
    This functions draws robots position in the world coordinate system."""

    # Fix the origin of the coordinate system
    offsetX = 100
    offsetY = 250

    # Constant needed for transforming from world coordinates to screen coordinates (flip the y-axis)
    ymax = world.shape[0]

    world[:] = CWHITE # Clear background to white

    # Find largest weight
    max_weight = 0
    for particle in particles:
        max_weight = max(max_weight, particle.getWeight())

    # Draw particles
    for particle in particles:
        x = int(particle.getX() + offsetX)
        y = ymax - (int(particle.getY() + offsetY))
        colour = jet(particle.getWeight() / max_weight)
        cv2.circle(world, (x,y), 2, colour, 2)
        b = (int(particle.getX() + 15.0*np.cos(particle.getTheta()))+offsetX, 
                                     ymax - (int(particle.getY() + 15.0*np.sin(particle.getTheta()))+offsetY))
        cv2.line(world, (x,y), b, colour, 2)

    # Draw landmarks
    for i in range(len(landmarkIDs)):
        ID = landmarkIDs[i]
        lm = (int(landmarks[ID][0] + offsetX), int(ymax - (landmarks[ID][1] + offsetY)))
        cv2.circle(world, lm, 5, landmark_colors[i], 2)

    # Draw estimated robot pose
    a = (int(est_pose.getX())+offsetX, ymax-(int(est_pose.getY())+offsetY))
    b = (int(est_pose.getX() + 15.0*np.cos(est_pose.getTheta()))+offsetX, 
                                 ymax-(int(est_pose.getY() + 15.0*np.sin(est_pose.getTheta()))+offsetY))
    cv2.circle(world, a, 5, CMAGENTA, 2)
    cv2.line(world, a, b, CMAGENTA, 2)

def rotate_vector(x,y,angle): #Rotates vector (x,y) by given angle
    new_x = x * np.cos(angle) - y * np.sin(angle)
    new_y = x * np.sin(angle) + y * np.cos(angle)
    return (new_x,new_y)


def initialize_particles(num_particles):
    particles = []
    for i in range(num_particles):
        # Random starting points. 
        p = particle.Particle(600.0*np.random.ranf() - 100.0, 600.0*np.random.ranf() - 250.0, np.mod(2.0*np.pi*np.random.ranf(), 2.0*np.pi), 1.0/num_particles)
        particles.append(p)
    return particles


unit_vector = [1,0]
count = 0
test = 0
rot_count = 0
found_id = []
found_dists = []

try:
    velocity = 0.0 # cm/sec
    angular_velocity = 0.0 # radians/sec

    # Initialize the robot (XXX: You do this)
    arlo = robot.Robot()

    cam = camera.Camera(0, 'arlo', useCaptureThread = False)
    cam.terminateCaptureThread()
    while True:

        


        colour = cam.get_next_frame()
        
        # Detect objects
        objectIDs, dists, angles = cam.detect_aruco_objects(colour)
        if not isinstance(objectIDs, type(None)):
            for i in range(3):
                # List detected objects
                accepted_ids = []
                accepted_dists = []
                accepted_angles = []
                for i in range(len(objectIDs)):
                    print("Object ID = ", objectIDs[i], ", Distance = ", dists[i], ", angle = ", angles[i])
            exit(0)

        arlo.go_diff(30,30,1,0) #spins the robots
        sleep(0.5)
        arlo.stop()

finally:
    cam.terminateCaptureThread()