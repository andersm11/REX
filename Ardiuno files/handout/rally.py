targets = [L1, L2, L3, L1]
states = [0,1,2,3]
class targetbox():
    def __init__(state, target):
        self.state = state
        self.target = target
    def find():
        kode til at checke om kamera kan se aruco kode
        hvis ja:
            return true
        hvis nej: 
            return false
    def nexttarget()
        self.target = target+1
        self.state = state+1
    def lasttarget()
        if state = 3:
            return true
        else: 
            return false

targetbox(state[0],target[0])

def searchtarget(targetbox):
    søg efter targetbox
    if targetbox.find = true:
        drivetotarget(targetbox) 
    else:
        changeposition()

def changeposition():
    while read_front_ping_sensor(self) > 200 and read_left_ping_sensor(self) > 100 and read_right_ping_sensor(self) > 100:
        kør 30 cm frem
        searchtarget(targetbox)
    opstacleavoid()

def drivetotarget(targetbox):
    while read_front_ping_sensor(self) > 200 and read_left_ping_sensor(self) > 100 and read_right_ping_sensor(self) > 100:
        vend mod targetbox
        kør mod targetbox
        if targetbox.last_targetbox = true
            stop
        if else dist(targetbox) <= 40 cm 
            targetbox = nexttargetbox
            searchtarget(targetbox)
    opstacleavoid
    searchtarget(targetbox)

def opstacleavoid():
    if read_front_ping_sensor(self) > 200:
        searchtarget(targetbox)
    else read_front_ping_sensor(self) < 200:
        while read_front_ping_sensor(self) < 200:
            if read_left_ping_sensor(self) > 100:
                drej 10 grader til venstre
            else read_right_ping_sensor(self) > 100:
                drej 10 grader til højre
        kør 10 cm frem
        obstacleavoid()

    


searchtarget(targetbox)