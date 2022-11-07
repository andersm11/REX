from pickle import TRUE
from re import I, search
from turtle import right
import time
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

cam_intrinsic_matrix = np.asarray([1672, 0., cam_imageSize[0] / 2.0, 0.,
       1672, cam_imageSize[1] / 2.0, 0., 0., 1.], dtype = np.float64)
cam_intrinsic_matrix.shape = (3, 3)

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
    vector = vector[0].reshape((3,))
    vector_norm = vector/np.linalg.norm(vector)
    beta = np.rad2deg(np.arccos(np.dot(vector_norm,z)))
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
        arlo.go_diff(30,30,0,1)
        sleep(0.0153*80)
        robot_drive(1)
        sleep(0.4)
        if avoidance() != "free":
            return "s_right"
        sleep(0.25)
        arlo.stop()
        arlo.go_diff(30,30,1,0)
        sleep(0.0153*70)
        arlo.stop()
        if avoidance() != "free":
            return "s_right"
        robot_drive(1)
        sleep(0.4)
        if avoidance() != "free":
            return "s_right"
        sleep(0.25)
        arlo.stop()
        arlo.go_diff(30,30,1,0)
        sleep(0.0153*70)
        arlo.stop()
        if avoidance() != "free":
            return "s_right"
        robot_drive(1)
        sleep(0.4)
        if avoidance() != "free":
            return "s_left"
        sleep(0.25)
        arlo.stop()
        return "s_left"
    if left < 300:
        print("left")
        arlo.stop()
      
        arlo.go_diff(30,30,1,0)
        sleep(0.0153*70)
        robot_drive(1)
        sleep(0.4)
        if avoidance() != "free":
            return "s_left"
        sleep(0.25)
        arlo.stop()
        arlo.go_diff(30,30,0,1)
        sleep(0.0153*70)
        arlo.stop()
        if avoidance() != "free":
            return "s_left"
        robot_drive(1)
        sleep(0.4)
        if avoidance() != "free":
            return "s_left"
        sleep(0.25)
        arlo.stop()
        arlo.go_diff(30,30,0,1)
        sleep(0.0153*70)
        arlo.stop()
        if avoidance() != "free":
            return "s_right"
        robot_drive(1)
        sleep(0.4)
        if avoidance() != "free":
            return "s_right"
        sleep(0.25)
        arlo.stop()
       
        return "s_right"
    if mid < 250 and right < 350:
        print("mid-right")
        arlo.stop()
       
        turn(-100)
        robot_drive(1)
        if avoidance() != "free":
            return "s_right"
        sleep(1)
        arlo.stop()
        turn(90)
        if avoidance() != "free":
            return "s_right"
        robot_drive(1)
        sleep(0.4)
        if avoidance() != "free":
            return "s_right"
        sleep(0.30)
        arlo.stop()
        turn(50)
        if avoidance() != "free":
            return "s_right"
        robot_drive(1)
        sleep(0.4)
        if avoidance() != "free":
            return "s_right"
        sleep(0.25)
        arlo.stop()

     
        return "s_left"
    if mid < 200:
        print("mid")
        arlo.stop()
     
        turn(100)
        if avoidance() != "free":
            return "s_left"
        robot_drive(1)
        if avoidance() != "free":
            return "s_left"
        sleep(1)
        arlo.stop()
        turn(-90)
        if avoidance() != "free":
            return "s_left"
        robot_drive(1)
        sleep(0.5)
        if avoidance() != "free":
            return "s_left"
        sleep(0.35)
        arlo.stop()
        turn(45)
        if avoidance() != "free":
            return "s_right"
        robot_drive(1)
        sleep(0.40)
        if avoidance() != "free":
            return "s_left"
        sleep(0.30)
        arlo.stop()


        return "s_right"
    else:
        return "free"



###########  ROBOT FUNCTIONS  ############
#################  END  ##################


############   RALLY CODE   ##############

def main():

    current_target = 0
    last_orientation_box = 0
    search_side = "s_right"
    state = 0
    counter = 0



    while True: #Main loop
        if state != 3:
            retval, frameReference = Take_pic()
            print("state:",state)
            print("TARGET:",current_target)
    
            if not retval: # Error
                print(" < < <  Game over!  > > > ")
                exit(-1)
            (corners, ids, rejected) = cv2.aruco.detectMarkers(frameReference, arucoDict,parameters=arucoParams)
            
    
            if ids is not None and current_target == 4:
                t_corners, t_id = check_id(corners,ids,current_target)
                rvec, tvec, objPoints = cv2.aruco.estimatePoseSingleMarkers(t_corners,15,cam_intrinsic_matrix,cam_distortion_coeffs)
            elif ids is not None:
                t_corners, t_id = check_id(corners,ids,current_target)
                rvec, tvec, objPoints = cv2.aruco.estimatePoseSingleMarkers(t_corners,15,cam_intrinsic_matrix,cam_distortion_coeffs)
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
            if time_to_drive < 2:
                print("I am too close")
                robot_drive(1)
                sleep(time_to_drive)
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
            arlo.go_diff(45,45,1,0)
            sleep(0.3)
            arlo.stop()
        if state == 0 and search_side =="s_left" and counter < 14:
            counter = counter+1
            print(counter)
            print(search_side)
            arlo.go_diff(45,45,0,1)
            sleep(0.3)
            arlo.stop()
        elif (state == 0 or state == 3) and counter >= 14:
            state = 3
            if search_side == "s_left":
                arlo.go_diff(45,45,0,1)
                sleep(0.3)
            else:
                arlo.go_diff(45,45,1,0)
                sleep(0.3)
            arlo.stop()
            retval, frameReference = Take_pic()
            (corners, ids, rejected) = cv2.aruco.detectMarkers(frameReference, arucoDict,parameters=arucoParams)
            if ids is not None:
                t_corners, t_id = check_id_mod(corners,ids,last_orientation_box)
                rvec, tvec, objPoints = cv2.aruco.estimatePoseSingleMarkers(t_corners,15,cam_intrinsic_matrix,cam_distortion_coeffs)
                if tvec is not None:
                    angle, distance = compute_angle_and_distance(tvec)
                    if distance >= 70:
                        print("angle",angle)
                        print("dist:", distance)
                        turn(angle)
                        sleep(0.2)
                        if distance > 250:
                            time_to_drive = 0.028*(abs(distance)*0.7)
                        else:
                            time_to_drive = 0.028*(abs(distance-15))
                        print("Going forward")
                        last_orientation_box = t_id
                        robot_drive(1)
                        start_time = time.time()
                        sleep(0.4)
                        check = "free"
                        while check == "free":
                            end_time = time.time()
                            check = avoidance()
                            if end_time-start_time >= time_to_drive:
                                break
                        arlo.stop()
                        if end_time-start_time >= time_to_drive and (distance < 200):
                            arlo.go_diff(45,30,1,0)
                            turn(100)
                            robot_drive(1)
                            sleep(1.25)
                            arlo.stop()
                            search_side = "s_left"
                        counter = 0
                        state = 0
                        

main()

############   RALLY CODE   ##############   
#################  END  ##################
