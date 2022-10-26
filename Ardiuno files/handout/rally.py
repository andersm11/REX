try:
    import robot
    onRobot = True
except ImportError:
    print("selflocalize.py: robot module not present - forcing not running on Arlo!")
    onRobot = False

import cv2

targets = [L1, L2, L3, L1]
states = [0,1,2,3]
class targetbox():
    def __init__(state, target):
        self.state = state
        self.target = target
    def find():
        #kode til at checke om kamera kan se aruco kode
        #hvis ja:
            return True
        #hvis nej: 
            return false
    def nexttarget():
        self.target = target+1
        self.state = state+1
    def lasttarget():
        if state == 3:
            return True
        else: 
            return False

targetbox(state[0],target[0])

def searchtarget(targetbox):
    #søg efter targetbox
    if targetbox.find == True:
        drivetotarget(targetbox) 
    else:
        changeposition()

def changeposition():
    while arlo.read_front_ping_sensor(self) > 200 and arlo.read_left_ping_sensor(self) > 100 and arlo.read_right_ping_sensor(self) > 100:
        #kør 30 cm frem
        searchtarget(targetbox)
    opstacleavoid()

def drivetotarget(targetbox):
    while arlo.read_front_ping_sensor(self) > 200 and arlo.read_left_ping_sensor(self) > 100 and arlo.read_right_ping_sensor(self) > 100:
        #vend mod targetbox
        #kør mod targetbox
        if targetbox.last_targetbox == True:
            arlo.stop
        elif dist(targetbox) <= 400:
            targetbox = nexttargetbox
            searchtarget(targetbox)
    opstacleavoid
    searchtarget(targetbox)

def opstacleavoid():
    if arlo.read_front_ping_sensor(self) > 200:
        searchtarget(targetbox)
    else:
        while arlo.read_front_ping_sensor(self) < 200:
            if arlo.read_left_ping_sensor(self) > 100:
                #drej 10 grader til venstre
            elif arlo.read_right_ping_sensor(self) > 100:
                #drej 10 grader mod højre
            else
                #kør 10 cm tilbage
                #drej 10 grader til venstre
        #kør 10 cm frem
        obstacleavoid()

    


searchtarget(targetbox)