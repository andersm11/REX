from re import I, search
from turtle import right
import robot
from time import sleep 
import Ex3
from Ex3 import search_and_find
arlo = robot.Robot()

landmarks = [2,4]
states = [0,1]
class  Landmark():
    def __init__(self, state, landmarks):
        self.state = state
        self.landmarks = landmarks
    
    
    def find(self, landmark):
        #kode til at checke om kamera kan se aruco kode
        # det her går jo slet ikke.. I know
        for i in range(self.landmarks):
            if landmark == self.landmarks[i]:
                return True
        else: 
            return False
    
    def nextLandmark(self):
        landmarks = landmarks[i+1]
        state = state[i+1]

    def lastLandmark(self):
        if self.state[i] == 1:
            return True
        else: 
            return False

landmark = Landmark(states[0],landmarks[0])

#def searchtarget(landmark):
#    while arlo.read_front_ping_sensor(self) > 200 and arlo.read_left_ping_sensor(self) > 100 and arlo.read_right_ping_sensor(self) > 100:
#        if Ex3.CheckID(landmark.landmarks) == True:
#            Ex3.search_and_find
#        else:
#            changeposition()
#            searchtarget(landmark)
#    obstacleavoid()

def changeposition():
    while arlo.read_front_ping_sensor() > 200 and arlo.read_left_ping_sensor() > 100 and arlo.read_right_ping_sensor() > 100:
        arlo.go_diff(40,40,1,1)
        sleep(1)
        search_and_find()
    obstacleavoid()

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

def obstacleavoid():
    if arlo.read_front_ping_sensor() > 200:
        search_and_find()
    else:
        while arlo.read_front_ping_sensor() < 200:
            if arlo.read_left_ping_sensor() > 200:
                arlo.go_diff(30,30,0,1)
                sleep(0.3)
            elif arlo.read_right_ping_sensor() > 200:
                arlo.go_diff(30,30,1,0)
                sleep(0.3)
            else:
                arlo.go_diff(30,30,0,0)
                sleep(0.5)
                arlo.go_diff(30,30,0,1)
                sleep(0.3)
                arlo.stop
        obstacleavoid()

search_and_find()

############# LANDMARK CLASS ##############
## Er det her nødtvendigt? 
#class  Landmark():
#    def __init__(self, state, landmarks):
#        self.state = state
#        self.landmarks = landmarks
#
#    
#    def find(self, landmark):
#        #kode til at checke om kamera kan se aruco kode
#        # det her går jo slet ikke.. I know
#        for i in range(self.landmarks):
#            if landmark == self.landmarks[i]:
#                return True
#        else: 
#            return False
#    
#    def nextLandmark(self):
#        landmarks = landmarks[i+1]
#        state = state[i+1]
#
#    def lastLandmark(self):
#        if self.state[i] == 3:
#            return True
#        else: 
#            return False
#
############# LANDMARK CLASS ##############
##################  END  ##################
#
#

############ Particle filter #############

############ Particle filter #############
#################  END  ##################



############  Localization  ##############
def localization():
    return angle, dist

############  Localization  ##############
#################  END  ##################


angle, dist = localization()

###########  ROBOT FUNCTIONS  ############
def robot_drive(distance, direction=1):
    arlo.go_diff(30,30,direction,direction)
    sleep(distance*0.028)

###########  ROBOT FUNCTIONS  ############
#################  END  ##################


############   RALLY CODE   ##############



# l1 = Landmark(0, 1)

# def search_Landmark():
#     søg efter targetbox
#     if Landmark.find() == True:
#         drivetotarget(Landmark) 
#     else:
#         changeposition()

# def change_Position():
#     while arlo.read_front_ping_sensor() > 200 and arlo.read_left_ping_sensor() > 100 and arlo.read_right_ping_sensor() > 100:
#         kør 30 cm frem
#         searchtarget(targetbox)
#     opstacleavoid()

# def drivetotarget(targetbox):
#     while arlo.read_front_ping_sensor() > 200 and arlo.read_left_ping_sensor() > 100 and arlo.read_right_ping_sensor() > 100:
#         turn()
#         kør mod targetbox
#         if targetbox.last_targetbox = true
#             arlo.stop()
#         if else dist(targetbox) <= 40 cm 
#             targetbox = nexttargetbox
#             search_Landmark(targetbox)
#     opstacleavoid
#     searchtarget(targetbox)

# i = 0
def opstacle_avoid():
    left_censor = arlo.read_left_ping_sensor()
    right_censor = arlo.read_right_ping_sensor()
    front_censor = arlo.read_front_ping_sensor()
    if left_censor < 30 and front_censor < 30:
        arlo.turn() #90 grader til højre
        search()
    if right_censor < 30 and front_censor < 30:
        arlo.turn() #90 grader til venstre
        search()
    if front_censor < 30 and  right_censor > 30 and left_censor > 30: 
        robot_drive(50,0) 

############   RALLY CODE   ##############   
#################  END  ##################

    

