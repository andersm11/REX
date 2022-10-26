from cmath import sin
import copy
from statistics import median
import math

from particle import move_particle
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
import random


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


        ## Use motor controls to update particles
        # XXX: Make the robot drive
        # XXX: You do this
        #VERY  simple test for our robot:
        arlo.go_diff(30,30,1,0) #spins the robots
        sleep(0.5)
        arlo.stop()
        velocity = 0
        angular_velocity = -np.deg2rad(32) # Gives the angular velocity in radians
        for p in particles:
            move_particle(p,0,0,angular_velocity)# Adds rotation to particles
        add_uncertainty(particles,5,0.1)
        angular_velocity = 0

        x_diff = 150 - est_pose.getX() #Difference of robot location to center point
        y_diff = 0 - est_pose.getY() #Differnce of robot location to center point
        dest_vector = [x_diff,y_diff] # The vector from robot to destination

        pose_angle = np.rad2deg(est_pose.getTheta()) # Gives orientation angle in degrees
        new_vector = rotate_vector(unit_vector[0],unit_vector[1],pose_angle) #Rotate unit vector to fit with robot orientation angle
        vec_distance = np.linalg.norm(dest_vector)
        
        norm_dest_vector = dest_vector/np.linalg.norm(dest_vector) #Normalize destination-vector
        angle_between = np.rad2deg(np.arccos(np.dot(new_vector,norm_dest_vector))) #Compute angle between robot-orientation-vector and destination-vector
        sign = np.sign(np.dot(new_vector,norm_dest_vector)) #Gives the sign of the angle
        angle_between *= sign


        count += 1
        if count > 20 or (rot_count == 1 and count > 15):
            
            rot_count += 1
            print("x:",est_pose.getX(),"y:",est_pose.getY())
            print("x diff", x_diff, "y_diff:", y_diff)
            print("pose angle:",pose_angle, "new vector:",new_vector)
            print("TURNING NOW. ANGLE:",angle_between)
            for k in range(5):
                est_pose = particle.estimate_pose(particles)
                draw_world(est_pose,particles,world)
                sleep(1)
            Turn(angle_between)
            angular_velocity = -np.deg2rad(32)
            for p in particles:
                move_particle(p,0,0,angular_velocity)
            add_uncertainty(particles,5,0.1)
            angular_velocity = 0
            print("TURN ENDED")
            arlo.go_diff(52,50,1,1)
            sleep(0.028 * vec_distance)
            for p in particles:
                move_particle(p,x_diff,y_diff,0)
            add_uncertainty(particles,10,0.05)
            velocity = 0
            count = 0
            if rot_count == 2:
                exit()

        # Fetch next frame
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
                    #if objectIDs[i] in landmarkIDs:
                    #    if not isinstance(found_objects, type(None)):
                    #        for ob in found_objects:
                    #            if ob[0] == objectIDs[i]:
                    #                ob = (objectIDs[i],dists[i],angles[i])
                    #        else:
                    #            found_objects.append(np.array(objectIDs[i],dists[i],angles[i]),axis=0)
                    #    found_objects.append(np.array(objectIDs[i],dists[i],angles[i]),axis=0)
                        # XXX: Do something for each detected object - remember, the same ID may appear several times
                    if objectIDs[i] in landmarkIDs and objectIDs[i] not in accepted_ids:
                        accepted_ids.append(objectIDs[i])
                        accepted_dists.append(dists[i]+22.0)
                        accepted_angles.append(angles[i])
                        if objectIDs[i] not in found_id:
                            found_id.append(objectIDs[i])
                            found_dists.append(dists[i]+22.0)

                objectIDs = accepted_ids
                dists = accepted_dists
                angles = accepted_angles
                if len(objectIDs) > 0:
                    # Compute particle weights
                    # XXX: You do this
                    compute_weights(objectIDs,dists,angles,particles)
                    normalize_weights(particles)
                    resample_gaussian(particles)
                    # Draw detected objects
                    add_uncertainty(particles,3,0.1)
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
            #cv2.imshow(WIN_RF1, colour)

            # Show world
            cv2.imshow(WIN_World, world)
    
  
finally: 
    # Make sure to clean up even if an exception occurred
    
    # Close all windows
    cv2.destroyAllWindows()

    # Clean-up capture thread
    cam.terminateCaptureThread()

