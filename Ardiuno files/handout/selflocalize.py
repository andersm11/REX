from cmath import sin
from statistics import median
import cv2
from cv2 import sqrt
import particle
import camera
import numpy as np
import time
from timeit import default_timer as timer
import sys
from time import sleep, time  
from sklearn import preprocessing


# Flags
showGUI = True  # Whether or not to open GUI windows
onRobot = True # Whether or not we are running on the Arlo robot
theta = "Theta"

def el(lx,ly,x,y):
    d = distance(lx,ly,x,y)
    return np.transpose([lx-x,ly-y])/d

def distance(lx,ly,x,y):
    return sqrt(((lx-x)**2)+((ly-y)**2)) #Distance from particle (x,y) to landmark (lx,ly)

def e_theta(theta):
    return np.transpose(np.cos(theta),np.sin(theta))

def e_theta_hat(theta):
    return np.transpose(-(np.sin(theta),np.cos(theta)))

def particle_angle(lx,ly,x,y,theta):
    return np.sign(np.dot(el(lx,ly,x,y),e_theta_hat(theta))*np.arccos(np.dot(el(lx,ly,x,y),e_theta(theta))))

def isRunningOnArlo():
    """Return True if we are running on Arlo, otherwise False.
      You can use this flag to switch the code from running on you laptop to Arlo - you need to do the programming here!
    """
    return onRobot


if isRunningOnArlo():
    # XXX: You need to change this path to point to where your robot.py file is located
    sys.path.append("REX/Ardiuno files/robot.py ")



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
    2: (300.0, 0.0)  # Coordinates for landmark 2
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
        if len(found_objects) < 2:
            arlo.go_diff(30,30,1,0)
            sleep(0.5)
            arlo.stop()
        if len(found_objects) == 2:
            obj1 = found_objects[0]
            obj2 = found_objects[1]
            angle = (obj1[1]**2+obj2[1]**2-300**2)/(2*obj1[1]*obj2[1]) #Compute angle between landmarks
            mid_angle = angle/2
            median_line = (1/2)*(sqrt(2*obj1[1]**2 + 2*obj2[1]*2 - 300**2)) #Compute median of triangle
            arlo.go_diff(30,30,1,0)
            sleep(0.019*abs(angle))
            arlo.stop
            colour = cam.get_next_frame()
            objectIDs, dists, angles = cam.detect_aruco_objects()
            if not isinstance(objectIDs, type(None)):
                arlo.go_diff(30,30,0,1)
                sleep(0.019*abs(mid_angle))
                arlo.stop()
                arlo.go_diff(52,50,1,1)
                sleep(0.028*(median_line))
                arlo.stop()
            else:
                arlo.go_diff(30,30,0,1)
                sleep(0.019*abs(angle)*2)
                arlo.stop()
                arlo.go_diff(52,50,1,1)
                sleep(0.028*(median_line))
                arlo.stop()


        
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
                if objectIDs[i] in landmarkIDs:
                    if not isinstance(found_objects, type(None)):
                        for ob in found_objects:
                            if ob[0] == objectIDs[i]:
                                ob = (objectIDs[i],dists[i],angles[i])
                        else:
                            found_objects.append(np.array(objectIDs[i],dists[i],angles[i]),axis=0)
                    found_objects.append(np.array(objectIDs[i],dists[i],angles[i]),axis=0)
                    # XXX: Do something for each detected object - remember, the same ID may appear several times

            # Compute particle weights
            # XXX: You do this
            def gaussian_pdf(d,dm,stdd):
                return (1/np.sqrt(2*np.pi*stdd**2))*np.exp(-(((dm-d)**2)/(2*stdd**2)))
            
            def p(x):
                return 0.3*gaussian_pdf(x,2,1) + 0.4*gaussian_pdf(x,5,2) + 0.3*gaussian_pdf(x,9,1)
            
            def gaussian(x,n,k):
                return p(x)/gaussian_pdf(x,n,k)

            def sample_gaussian(n,k,værdi):
                samples = np.random.normal(n,k,værdi)
                #værdi kunne være 1000
                return samples
            
            def weight_gaussian(samples):
                weighted_samples = np.array(list(map(gaussian,samples)))
                return weighted_samples

            def gaussian_normalize_weight():
                 def normalize_weights(weights):
                    normalized = []
                    for w in weights:
                        normalized.append(w/(sum(weights)))
                    return normalized

            
            # Resampling
            # XXX: You do this
            def resample_gaussian(samples,n_weights):
                resamples = np.random.choice(samples,1000,p=n_weights,replace=True)
                return resamples
            
            samples_norm = sample_gaussian()
            weights_norm = weight_gaussian(samples_norm)
            n_weights_norm = gaussian_normalize_weight(weights_norm)
            resamples_norm = resample_gaussian(samples_norm, n_weights_norm)


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

