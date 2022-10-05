from cmath import sin
from statistics import median

from random_numbers import randn
from particle import add_uncertainty
import cv2
from cv2 import sqrt
import particle
import camera
import numpy as np
import time
from timeit import default_timer as timer
import sys
from time import sleep, time  


# Flags
showGUI = True  # Whether or not to open GUI windows
onRobot = True # Whether or not we are running on the Arlo robot


def el(lx,ly,x,y):
    d = distance(lx,ly,x,y)
    print("d:",d,"\n")
    return np.transpose([lx-x,ly-y])/d

def distance(lx,ly,x,y):
    result = sqrt(((lx-x)**2)+((ly-y)**2))
    print("result:",result,"\n")
    return result#Distance from particle (x,y) to landmark (lx,ly)

def e_theta(theta):
    return np.transpose(np.cos(theta),np.sin(theta))

def e_theta_hat(theta):
    return np.transpose(-(np.sin(theta),np.cos(theta)))

def particle_angle(lx,ly,x,y,theta):
    return np.sign(np.dot(el(lx,ly,x,y),e_theta_hat(theta))*np.arccos(np.dot(el(lx,ly,x,y),e_theta(theta))))

def gaussian_pdf_distance(d,dm,stdd):
                return (1.0/np.sqrt(2.0*np.pi*stdd**2))*np.exp(-(((dm-d)**2)/(2.0*stdd**2)))

def gaussian_pdf_angle(m_angle,lx,ly,x,y,theta,stdd):
                print("lx:",lx,"ly:",ly,"\n")
                print("x:",x,"y:",y,"\n")
                return (1.0/np.sqrt(2.0*np.pi*stdd**2))*np.exp(-(((m_angle-particle_angle(lx,ly,x,y,theta))**2)/(2.0*stdd**2)))


def compute_weights(landmarkIDs,landmark_d, landmark_a ,old_particles):
    pweights = []
    for op in old_particles:
        weight = 1
        for i in range(len(landmarkIDs)):
            d = distance(landmarks[landmarkIDs[i]][0],landmarks[landmarkIDs[i]][1],op.getX(),op.getY()) #hypo distance
            dm = landmark_d[i]
            weight = weight * gaussian_pdf_distance(d,dm,0.2)*gaussian_pdf_angle(landmark_a[i],landmarks[landmarkIDs[i]][0],landmarks[landmarkIDs[i]][1],op.getX(),op.getY(),op.getTheta(),0.2)
        pweights.append((old_particles,weight))
    return pweights

def normalize_weights(pweights):
    nweights = []
    for p in pweights:
        norm_weight = p[1]/(sum(p[:,len(p)]))
        nweights.append(pweights[0],norm_weight)
    return nweights


def resample_gaussian(sw_list):
    resamples = np.random.choice(sw_list[0],1000,p=sw_list[1],replace=True)
    return resamples

def simple_sample(b):
    b = sqrt(b)
    print(b)
    return (1.0/2.0)*np.sum(np.random.uniform(low=-b,high=b,size=12))

def sample_motion_model_velocity(particle,v,w):
    x = particle.getX()
    y = particle.getY()
    theta = particle.getTheta()
    v_hat = v + simple_sample((0.1*v**2+0.2*w**2))
    w_hat = w + simple_sample((0.1*v**2+0.2*w**2))
    epsilon = simple_sample((0.1*v**2+0.2*w**2))
    new_x = x - (v_hat/w_hat)*np.sin(theta) + (v_hat/w_hat)*np.sin(theta + w_hat)
    new_y = y + (v_hat/w_hat)*np.cos(theta) - (v_hat/w_hat)*np.cos(theta + w_hat)
    new_theta = theta + w_hat + epsilon
    particle.setX(new_x)
    particle.setY(new_y)
    particle.setTheta(new_theta)
    return particle


def sample_motion_model_velocity_withT(particle,v,w,delta_t):
    x = particle.getX()
    y = particle.getY()
    theta = particle.getTheta()
    #v_hat = v + simple_sample(0.1*v**2+0.2*w**2)
    #w_hat = w + simple_sample(0.1*v**2+0.2*w**2)
    #epsilon = simple_sample(0.1*v**2+0.2*w**2)
    v_hat = v + randn(0,0.1*v**2+0.2*w**2)
    w_hat = w + randn(0,0.1*v**2+0.2*w**2)
    epsilon = randn(0,0.1*v**2+0.2*w**2)
    new_x = x - (v_hat/w_hat)*np.sin(theta) + (v_hat/w_hat)*np.sin(theta + w_hat*delta_t)
    new_y = y + (v_hat/w_hat)*np.cos(theta) - (v_hat/w_hat)*np.cos(theta + w_hat*delta_t)
    new_theta = theta + w_hat*delta_t + epsilon*delta_t
    particle.setX(new_x)
    particle.setY(new_y)
    particle.setTheta(new_theta)

    # particles er defineret ved:
    # num_particles = 1000 
    # particles = initialize_particles(num_particles)
    # For ex2.2; 0.3*gaussian_pdf(x,2,1) + 0.4*gaussian_pdf(x,5,2) + 0.3*gaussian_pdf(x,9,1)
            
#def gaussian(x,n,k):
#    return p(x)/gaussian_pdf_distance(x,n,k)
#
#def sample_gaussian(n,k,værdi):
#    samples = np.random.normal(n,k,værdi)
#    #værdi kunne være 1000
#    return samples
#            
#def weight_gaussian(samples):
#    weighted_samples = np.array(list(map(gaussian,samples)))
#    return weighted_samples
#
#def gaussian_normalize_weight():
#    def normalize_weights(weights):
#        normalized = []
#        for w in weights:
#            normalized.append(w/(sum(weights)))
#        return normalized






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
landmarkIDs = [1, 9]
landmarks = {
    1: (0.0, 0.0),  # Coordinates for landmark 1
    9: (300.0, 0.0)  # Coordinates for landmark 2
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



def initialize_particles(num_particles):
    particles = []
    for i in range(num_particles):
        # Random starting points. 
        p = particle.Particle(600.0*np.random.ranf() - 100.0, 600.0*np.random.ranf() - 250.0, np.mod(2.0*np.pi*np.random.ranf(), 2.0*np.pi), 1.0/num_particles)
        particles.append(p)

    return particles


# Main program #
try:
    if showGUI:
        # Open windows
        WIN_RF1 = "Robot view"
        cv2.namedWindow(WIN_RF1)
        cv2.moveWindow(WIN_RF1, 50, 50)

        WIN_World = "World view"
        cv2.namedWindow(WIN_World)
        cv2.moveWindow(WIN_World, 500, 50)


    # Initialize particles
    num_particles = 1000
    particles = initialize_particles(num_particles)

    found_objects = [] #use this to save found objects (if needed)

    est_pose = particle.estimate_pose(particles) # The estimate of the robots current pose


    # Driving parameters
    velocity = 0.0 # cm/sec
    angular_velocity = 0.0 # radians/sec

    # Initialize the robot (XXX: You do this)
    arlo = robot.Robot()

    # Allocate space for world map
    world = np.zeros((500,500,3), dtype=np.uint8)

    # Draw map
    draw_world(est_pose, particles, world)

    print("Opening and initializing camera")
    if camera.isRunningOnArlo():
        cam = camera.Camera(0, 'arlo', useCaptureThread = True)
    else:
        cam = camera.Camera(0, 'macbookpro', useCaptureThread = True)

    while True:

        # Move the robot according to user input (only for testing)
        action = cv2.waitKey(10)
        if action == ord('q'): # Quit
            break
    
        if not isRunningOnArlo():
            if action == ord('w'): # Forward
                velocity += 4.0
            elif action == ord('x'): # Backwards
                velocity -= 4.0
            elif action == ord('s'): # Stop
                velocity = 0.0
                angular_velocity = 0.0
            elif action == ord('a'): # Left
                angular_velocity += 0.2
            elif action == ord('d'): # Right
                angular_velocity -= 0.2


        #THIS MIGHT BE SHIT
        #if len(found_objects) < 2:
        #    arlo.go_diff(30,30,1,0)
        #    sleep(0.5)
        #    arlo.stop()
        #if len(found_objects) == 2:
        #    obj1 = found_objects[0]
        #    obj2 = found_objects[1]
        #    angle = (obj1[1]**2+obj2[1]**2-300**2)/(2*obj1[1]*obj2[1]) #Compute angle between landmarks
        #    mid_angle = angle/2
        #    median_line = (1/2)*(sqrt(2*obj1[1]**2 + 2*obj2[1]*2 - 300**2)) #Compute median of triangle
        #    arlo.go_diff(30,30,1,0)
        #    sleep(0.019*abs(angle))
        #    arlo.stop
        #    colour = cam.get_next_frame()
        #    objectIDs, dists, angles = cam.detect_aruco_objects()
        #    if not isinstance(objectIDs, type(None)):
        #        arlo.go_diff(30,30,0,1)
        #        sleep(0.019*abs(mid_angle))
        #        arlo.stop()
        #        arlo.go_diff(52,50,1,1)
        #        sleep(0.028*(median_line))
        #        arlo.stop()
        #    else:
        #        arlo.go_diff(30,30,0,1)
        #        sleep(0.019*abs(angle)*2)
        #        arlo.stop()
        #        arlo.go_diff(52,50,1,1)
        #        sleep(0.028*(median_line))
        #        arlo.stop()

        #VERY  simple test for our robot:
        arlo.go_diff(30,30,1,0)
        sleep(0.5)
        arlo.stop()
        velocity = 0
        angular_velocity = np.deg2rad(26.3)
        for p in particles:
            sample_motion_model_velocity_withT(p,velocity,angular_velocity,0.5)


        
        # Use motor controls to update particles
        # XXX: Make the robot drive
        # XXX: You do this


        # Fetch next frame
        colour = cam.get_next_frame()
        
        # Detect objects
        objectIDs, dists, angles = cam.detect_aruco_objects(colour)
        if not isinstance(objectIDs, type(None)):
            # List detected objects
            for i in range(len(objectIDs)):
                print("Object ID = ", objectIDs[i], ", Distance = ", dists[i], ", angle = ", angles[i])
                #if objectIDs[i] in landmarkIDs:
                #    if not isinstance(found_objects, type(None)):
                #        for ob in found_objects:
                #            if ob[0] == objectIDs[i]:
                #                ob = (objectIDs[i],dists[i],angles[i])
                #        else:
                #            found_objects.append(np.array(objectIDs[i],dists[i],angles[i]),axis=0)
                #    found_objects.append(np.array(objectIDs[i],dists[i],angles[i]),axis=0)
                    # XXX: Do something for each detected object - remember, the same ID may appear several times

            # Compute particle weights
            # XXX: You do this
            
            particles_w_weights = compute_weights(objectIDs,dists,angles,particles)
            particles_w_normweights = normalize_weights(particles_w_weights)
            particles = resample_gaussian(particles_w_normweights)
            particles = add_uncertainty(particles,0.2,0.2)
            # Draw detected objects
            cam.draw_aruco_objects(colour)
        else:
            # No observation - reset weights to uniform distribution
            for p in particles:
                p.setWeight(1.0/num_particles)

    
        est_pose = particle.estimate_pose(particles) # The estimate of the robots current pose

        if showGUI:
            # Draw map
            draw_world(est_pose, particles, world)
    
            # Show frame
            cv2.imshow(WIN_RF1, colour)

            # Show world
            cv2.imshow(WIN_World, world)
    
  
finally: 
    # Make sure to clean up even if an exception occurred
    
    # Close all windows
    cv2.destroyAllWindows()

    # Clean-up capture thread
    cam.terminateCaptureThread()

