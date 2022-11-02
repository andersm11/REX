from pickle import TRUE
from re import I, search
from turtle import right
import robot
import sys
import numpy as np
from time import sleep 
import camera


sys.path.append("../robot.py ")
version = "v0.0.2"
landmarks = [4,2,11]
landmark_numbers = {
    2 : 0,
    4 : 1,
    11 : 2
}
states = [0,1]

########### Setup ############


#Camera info:
#cam_imageSize = (1280, 720)
##self.intrinsic_matrix = np.asarray([ 7.1305391967046853e+02, 0., 3.1172820723774367e+02, 0.,
##       7.0564929862291285e+02, 2.5634470978315028e+02, 0., 0., 1. ], dtype = np.float64)
##self.intrinsic_matrix = np.asarray([ 6.0727040957659040e+02, 0., 3.0757300398967601e+02, 0.,
##       6.0768864690145904e+02, 2.8935674612358201e+02, 0., 0., 1. ], dtype = np.float64)
#cam_intrinsic_matrix = np.asarray([500, 0., cam_imageSize[0] / 2.0, 0.,
#       500, cam_imageSize[1] / 2.0, 0., 0., 1.], dtype = np.float64)
#cam_intrinsic_matrix.shape = (3, 3)
##self.distortion_coeffs = np.asarray([ 1.1911006165076067e-01, -1.0003366233413549e+00,
##       1.9287903277399834e-02, -2.3728201444308114e-03, -2.8137265581326476e-01 ], dtype = np.float64)
#cam_distortion_coeffs = np.asarray([0., 0., 2.0546093607192093e-02, -3.5538453075048249e-03, 0.], dtype = np.float64)


#try:
#    import robot
#    onRobot = True
#except ImportError:
#    print("rally_%s.py: robot module not present - forcing not running on Arlo!",version)
#    onRobot = False
#
class object:
    def __init__(self, id, dist, angle):
        self.id = id
        self.dist = dist
        self.angle = angle #IS DEGREES

    def getID(self):
        return self.id

    def getDist(self):
        return self.dist

    def getAngle(self):
        return self.angle #IS DEGREES
    

#def searchtarget(landmark):
#    while arlo.read_front_ping_sensor(self) > 200 and arlo.read_left_ping_sensor(self) > 100 and arlo.read_right_ping_sensor(self) > 100:
#        if Ex3.CheckID(landmark.landmarks) == True:
#            Ex3.search_and_find
#        else:
#            changeposition()
#            searchtarget(landmark)
#    obstacleavoid()

#def changeposition():
#    while arlo.read_front_ping_sensor() > 200 and arlo.read_left_ping_sensor() > 100 and arlo.read_right_ping_sensor() > 100:
#        arlo.go_diff(40,40,1,1)
#        sleep(1)
#        search_and_find()
#    obstacleavoid()

#def drivetotarget(landmark):
#    while arlo.read_front_ping_sensor(self) > 200 and arlo.read_left_ping_sensor(self) > 100 and arlo.read_right_ping_sensor(self) > 100:
#        #vend mod targetbox
#        #kør mod targetbox
#        if landmark.lastLandmark == True:
#            arlo.stop
#        elif dist(landmark) <= 400:
#            landmark.nextLandmark
#            searchtarget(landmark)
#    obstacleavoid
#    searchtarget(targetbox)

#def obstacleavoid():
#    if arlo.read_front_ping_sensor() > 200:
#        search_and_find()
#    else:
#        while arlo.read_front_ping_sensor() < 200:
#            if arlo.read_left_ping_sensor() > 200:
#                arlo.go_diff(30,30,0,1)
#                sleep(0.3)
#            elif arlo.read_right_ping_sensor() > 200:
#                arlo.go_diff(30,30,1,0)
#                sleep(0.3)
#            else:
#                arlo.go_diff(30,30,0,0)
#                sleep(0.5)
#                arlo.go_diff(30,30,0,1)
#                sleep(0.3)
#                arlo.stop
#        obstacleavoid()

#search_and_find()

############# LANDMARK CLASS ##############

############# LANDMARK CLASS ##############
##################  END  ##################
#
#

############ Particle filter #############

############ Particle filter #############
#################  END  ##################



############  Localization  ##############


############  Localization  ##############
#################  END  ##################


###########  ROBOT FUNCTIONS  ############
def robot_drive(direction = 1):
    arlo.go_diff(50,50,direction,direction)


def turn(angle):
    if angle <= 0:
        arlo.go_diff(30,30,0,1)
        sleep(0.0153*abs(angle))
        arlo.stop()
    else:
        arlo.go_diff(30,30,1,0)
        sleep(0.0153*abs(angle))
        arlo.stop()    

def avoidance():
    right = arlo.read_right_ping_sensor()
    mid = arlo.read_front_ping_sensor()
    left = arlo.read_right_ping_sensor()
    if right < 200:
        print("right")
        arlo.stop()
        arlo.go_diff(30,30,0,1)
        sleep(0.0153*45)
        robot_drive(1)
        sleep(0.5)
        arlo.stop()
        arlo.go_diff(30,30,1,0)
        sleep(0.0153*45)
        arlo.stop()
        sleep(5)
    if left < 200:
        print("left")
        arlo.stop()
        arlo.go_diff(30,30,1,0)
        sleep(0.0153*45)
        robot_drive(1)
        sleep(0.5)
        arlo.stop()
        arlo.go_diff(30,30,0,1)
        sleep(0.0153*45)
        arlo.stop()
        sleep(5)
    if mid < 200 and right < 300:
        print("mid-right")
        arlo.stop()
        turn(-90)
        robot_drive(1)
        sleep(0.5)
        arlo.stop()
        turn(90)
        sleep(5)
    if mid < 200:
        print("mid")
        arlo.stop()
        turn(90)
        robot_drive(1)
        sleep(0.5)
        arlo.stop()
        turn(-90)
        sleep(5)



###########  ROBOT FUNCTIONS  ############
#################  END  ##################


############   RALLY CODE   ##############
try:
    
    arlo = robot.Robot()
    print("Opening and initializing camera")
    cam = camera.Camera(0, 'arlo', useCaptureThread = True)
    current_target = 0
    target_object = None
    found_target = False
    state = 0

    while True: #Main loop
        #print("State:",state)           

        frame = cam.get_next_frame()
        objectIDs, dists, angles = cam.detect_aruco_objects(frame)
        sleep(0.2)
        if not isinstance(objectIDs, type(None)):
            for i in range(len(objectIDs)):
                print("Object ID = ", objectIDs[i], ", Distance = ", dists[i], ", angle = ", angles[i])
                if objectIDs[i] in landmarks and landmark_numbers[objectIDs[i]] == current_target: #Check if object is our current target
                    target_object = object(objectIDs[i],dists[i],np.rad2deg(angles[i]))
                    state = 1
#                    target_object = found_obj
#                    state = 1
#                    print(" got em")
                    #if (target_object is None) or target_object.getDist() > found_obj.getDist(): #Set our target to the object found if it is closer
                    #    print("found")
                    #    target_object = found_obj
                    #    state = 1
        sleep(0.3)
        if state == 0:
            arlo.go_diff(30,30,1,0)
            sleep(0.5)
            arlo.stop()

        if target_object is not None:
            if abs(target_object.getAngle()) > 5:
                print("angle:",target_object.getAngle())
                print("TURNING")
                sleep(3)
                turn(target_object.getAngle())
                target_object = None
                sleep(3)
            else:
                state = 2
                print("DRIVING")
                arlo.go_diff(0.028*target_object.dist)
                #robot_drive(1)
        else:
            state = 0
        avoidance()

finally: 
    cam.terminateCaptureThread()


############   RALLY CODE   ##############   
#################  END  ##################

    


