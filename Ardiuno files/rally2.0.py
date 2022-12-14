from pickle import TRUE
from re import I, search
from turtle import right
import time
#import robot
import sys
import numpy as np
import cv2
from time import sleep 
import camera
from aux_file import Take_pic


sys.path.append("../robot.py ")
version = "v0.0.2"
landmarks = [1,2,3,4]
landmark_numbers = {
    1 : 0,
    2 : 1,
    3 : 2,
    4 : 3,
}
states = [0,1]

########### Setup ############


#Camera info:
cam_imageSize = (1280, 720)
##self.intrinsic_matrix = np.asarray([ 7.1305391967046853e+02, 0., 3.1172820723774367e+02, 0.,
##       7.0564929862291285e+02, 2.5634470978315028e+02, 0., 0., 1. ], dtype = np.float64)
##self.intrinsic_matrix = np.asarray([ 6.0727040957659040e+02, 0., 3.0757300398967601e+02, 0.,
##       6.0768864690145904e+02, 2.8935674612358201e+02, 0., 0., 1. ], dtype = np.float64)
#cam_intrinsic_matrix = np.asarray([[1672,0,540],[0,1672,400],[0,0,1]])
cam_intrinsic_matrix = np.asarray([1672, 0., cam_imageSize[0] / 2.0, 0.,
       1672, cam_imageSize[1] / 2.0, 0., 0., 1.], dtype = np.float64)
cam_intrinsic_matrix.shape = (3, 3)
##self.distortion_coeffs = np.asarray([ 1.1911006165076067e-01, -1.0003366233413549e+00,
##       1.9287903277399834e-02, -2.3728201444308114e-03, -2.8137265581326476e-01 ], dtype = np.float64)
cam_distortion_coeffs = np.asarray([0., 0., 2.0546093607192093e-02, -3.5538453075048249e-03, 0.], dtype = np.float64)

arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
arucoParams = cv2.aruco.DetectorParameters_create()
dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
z = np.array([0,0,1],dtype=float)
x = np.array([1,0,0],dtype=float)

try:
    import robot
    onRobot = True
except ImportError:
    print("rally_%s.py: robot module not present - forcing not running on Arlo!",version)
    onRobot = False
arlo = robot.Robot()
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
#        #k??r mod targetbox
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
    arlo.go_diff(54,50,direction,direction)


def turn(angle):
    if angle <= 0:
        arlo.go_diff(30,30,0,1)
        sleep(0.0153*abs(angle))
        arlo.stop()
    else:
        arlo.go_diff(30,30,1,0)
        sleep(0.0153*abs(angle))
        arlo.stop()    

def check_id(corners, ids, current_target):
    for i in range(len(ids)):
        print(ids[i][0])
        if ids[i][0] == 1 and current_target == 4:
            return (corners[i],ids[i][0])
        if ids[i][0] in landmark_numbers and landmark_numbers[ids[i][0]] == current_target:
            return (corners[i],ids[i][0])
    return None, None

def check_id_mod(corners, ids, last_box):
    for i in range(len(ids)):
        if ids[i][0] not in landmark_numbers and ids[i][0] != last_box:
            return (corners[i],ids[i][0])
    return None,None


def compute_angle_and_distance(vector):
    #tvec2 = np.reshape(tvec[0,:],(3,))
    vector = vector[0].reshape((3,))
    vector_norm = vector/np.linalg.norm(vector)
    print(vector.shape)
    beta = np.rad2deg(np.arccos(np.dot(vector_norm,z)))
    print("dot:",np.dot(vector_norm, z) )
    sign = np.sign(np.dot(np.transpose(x),vector))
    angle = beta*sign
    print("beta", angle, "sign:", sign)
    dist = np.linalg.norm(vector)
    return angle, dist

def avoidance():
    right = arlo.read_right_ping_sensor()
    mid = arlo.read_front_ping_sensor()
    left = arlo.read_left_ping_sensor()
    if right < 300:
        print("right")
        arlo.stop()
        #sleep(1)
        arlo.go_diff(30,30,0,1)
        sleep(0.0153*80)
        robot_drive(1)
        sleep(0.4)
        if avoidance() != "free":
            return "s_right"
        sleep(0.25)
        arlo.stop()
        arlo.go_diff(30,30,1,0)
        sleep(0.0153*60)
        #arlo.stop()
        #if avoidance() != "free":
        #    return "s_right"
        #robot_drive(1)
        #sleep(0.4)
        #if avoidance() != "free":
        #    return "s_right"
        #sleep(0.25)
        #arlo.stop()
        #arlo.go_diff(30,30,1,0)
        #sleep(0.0153*50)
        #arlo.stop()
        #if avoidance() != "free":
        #    return "s_right"
        #robot_drive(1)
        #sleep(0.4)
        #if avoidance() != "free":
        #    return "s_left"
        #sleep(0.25)
        arlo.stop()

        #sleep(2)
        return "s_right"
    if left < 300:
        print("left")
        arlo.stop()
        #sleep(1)
        arlo.go_diff(30,30,1,0)
        sleep(0.0153*70)
        robot_drive(1)
        sleep(0.4)
        if avoidance() != "free":
            return "s_left"
        sleep(0.25)
        arlo.stop()
        arlo.go_diff(30,30,0,1)
        sleep(0.0153*60)
        #arlo.stop()
        #if avoidance() != "free":
        #    return "s_left"
        #robot_drive(1)
        #sleep(0.4)
        #if avoidance() != "free":
        #    return "s_left"
        #sleep(0.25)
        arlo.stop()
        #arlo.go_diff(30,30,0,1)
        #sleep(0.0153*50)
        #arlo.stop()
        #if avoidance() != "free":
        #    return "s_right"
        #robot_drive(1)
        #sleep(0.4)
        #if avoidance() != "free":
        #    return "s_right"
        #sleep(0.25)
        #arlo.stop()
        #sleep(2)
        return "s_left"
    if mid < 200 and right < 300:
        print("mid-right")
        arlo.stop()
        #sleep(1)
        turn(-100)
        robot_drive(1)
        if avoidance() != "free":
            return "s_right"
        sleep(1)
        arlo.stop()
        turn(90)
        #if avoidance() != "free":
        #    return "s_right"
        #robot_drive(1)
        #sleep(0.4)
        #if avoidance() != "free":
        #    return "s_right"
        #sleep(0.30)
        arlo.stop()
        #turn(45)
        #if avoidance() != "free":
        #    return "s_right"
        #robot_drive(1)
        #sleep(0.4)
        #if avoidance() != "free":
        #    return "s_right"
        #sleep(0.25)
        #arlo.stop()

        #sleep(2)
        return "s_right"
    if mid < 200:
        print("mid")
        arlo.stop()
        #sleep(1)
        turn(100)
        if avoidance() != "free":
            return "s_left"
        robot_drive(1)
        if avoidance() != "free":
            return "s_left"
        sleep(1)
        arlo.stop()
        turn(-90)
        #if avoidance() != "free":
        #    return "s_left"
        #robot_drive(1)
        #sleep(0.5)
        #if avoidance() != "free":
        #    return "s_left"
        #sleep(0.35)
        arlo.stop()
        #turn(45)
        #if avoidance() != "free":
        #    return "s_right"
        #robot_drive(1)
        #sleep(0.40)
        #if avoidance() != "free":
        #    return "s_left"
        #sleep(0.30)
        #arlo.stop()

        #sleep(2)
        return "s_left"
    else:
        return "free"



###########  ROBOT FUNCTIONS  ############
#################  END  ##################


############   RALLY CODE   ##############

def main():

    
    #print("Opening and initializing camera")
    #cam = camera.Camera(0, 'arlo', useCaptureThread = True)
    current_target = 0
    last_orientation_box = 0
    search_side = "s_right"
    state = 0
    counter = 0



    while True: #Main loop
        if state != 3:
            retval, frameReference = Take_pic()
            print("state:",state)
            print("TARGET", current_target)

            if not retval: # Error
                print(" < < <  Game over!  > > > ")
                exit(-1)
            (corners, ids, rejected) = cv2.aruco.detectMarkers(frameReference, arucoDict,parameters=arucoParams)


            if ids is not None and current_target == 4:
                t_corners, t_id = check_id(corners,ids,current_target)
                rvec, tvec, objPoints = cv2.aruco.estimatePoseSingleMarkers(t_corners,15,cam_intrinsic_matrix,cam_distortion_coeffs)
                print("t_id",t_id)
            elif ids is not None:
                t_corners, t_id = check_id(corners,ids,current_target)
                rvec, tvec, objPoints = cv2.aruco.estimatePoseSingleMarkers(t_corners,15,cam_intrinsic_matrix,cam_distortion_coeffs)
                print("t_id",t_id)
            else:
                tvec = None



        if tvec is not None and state == 0:
            angle, distance = compute_angle_and_distance(tvec)
            print("angle",angle)
            print("dist:", distance)
            turn(angle)
            sleep(0.5)
            start_time = time.time()
            time_to_drive = 0.028*(abs(distance-15))
            state = 1
            
        if state == 1:
            counter = 0
            robot_drive(1)
            if time_to_drive < 1.6:
                robot_drive(1)
                start_time = time.time()
                end_time = time.time()
                while (end_time - start_time) < time_to_drive:
                    avoidance()
                    end_time = time.time()
                arlo.stop()
                
                state = 0
                if current_target == 4:
                    exit(0)
                current_target += 1
            else:
                while state == 1: 
                    end_time = time.time()
                    time_diff = end_time - start_time
                    check = avoidance()
                    if check != "free":
                        search_side = check
                        state = 0
                    elif time_diff >= time_to_drive:
                        arlo.stop()
                        if current_target == 4:
                            exit(0)
                        current_target += 1
                        last_orientation_box = 0
                        state = 0

        if state == 0 and search_side == "s_right" and counter < 14:
            counter = counter+1
            print(counter)
            print(search_side)
            arlo.go_diff(30,30,1,0)
            sleep(0.5)
            arlo.stop()
        if state == 0 and search_side =="s_left" and counter < 14:
            counter = counter+1
            print(counter)
            print(search_side)
            arlo.go_diff(30,30,0,1)
            sleep(0.5)
            arlo.stop()
        elif (state == 0 or state == 3) and counter >= 14:
            state = 3
            if search_side == "s_left":
                arlo.go_diff(30,30,0,1)
                sleep(0.5)
            else:
                arlo.go_diff(30,30,1,0)
                sleep(0.5)
            arlo.stop()
            retval, frameReference = Take_pic()
            (corners, ids, rejected) = cv2.aruco.detectMarkers(frameReference, arucoDict,parameters=arucoParams)
            if ids is not None:
                t_corners, t_id = check_id_mod(corners,ids,last_orientation_box)
                rvec, tvec, objPoints = cv2.aruco.estimatePoseSingleMarkers(t_corners,15,cam_intrinsic_matrix,cam_distortion_coeffs)
                if tvec is not None:
                    angle, distance = compute_angle_and_distance(tvec)
                    if distance >= 100:
    
                        print("angle",angle)
                        print("dist:", distance)
                        turn(angle)
                        sleep(0.2)
                        if distance > 250:
                            time_to_drive = 0.028*(abs(distance-10)/2)
                        else:
                            time_to_drive = 0.028*(abs(distance-10))
                        print("Going forward")
                        last_orientation_box = t_id
                        robot_drive(1)    
                        start_time = time.time()
                        sleep(0.4)
                        check = "free"
                        while check == "free":
                            print("i am moving")
                            end_time = time.time()
                            check = avoidance()
                            if end_time-start_time >= time_to_drive:
                                break
                        counter = 0
                        state = 0
                        arlo.stop()


    
    


        #if CheckID(ids) is True:
        #    if 3 < beta:
        #        Turn(angle)
        #    else:
        #        length = np.linalg.norm(tvec)
        #        if length < 30:
        #            exit(1)
        #        

        #print("State:",state)           

        #frame = cam.get_next_frame()
        #objectIDs, dists, angles = cam.detect_aruco_objects(frame)
        #sleep(0.2)
        #if not isinstance(objectIDs, type(None)):
        #    for i in range(len(objectIDs)):
        #        print("Object ID = ", objectIDs[i], ", Distance = ", dists[i], ", angle = ", angles[i])
        #        if objectIDs[i] in landmarks and landmark_numbers[objectIDs[i]] == current_target: #Check if object is our current target
        #            target_object = object(objectIDs[i],dists[i],np.rad2deg(angles[i]))
        #            state = 1
        #            sleep(1)
        #            print(target_object.angle)
        #            if target_object.angle > 11:
        #                arlo.go_diff(30,30,0,1)
        #                sleep(0.0153*abs(target_object.angle-10))
        #                arlo.stop()
        #            elif target_object.angle < -11:
        #                arlo.go_diff(30,30,1,0)
        #                sleep(0.0153*abs(target_object.angle+10))
        #                arlo.stop()
        #else:
        #    target_object = None
#       #             target_object = found_obj
#       #             state = 1
#                    print(" got em")
                    #if (target_object is None) or target_object.getDist() > found_obj.getDist(): #Set our target to the object found if it is closer
                    #    print("found")
                    #    target_object = found_obj
                    #    state = 1
        #search_and_find()
        #avoidance()

main()

############   RALLY CODE   ##############   
#################  END  ##################

    


