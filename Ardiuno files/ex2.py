import robot 
from time import sleep, time  
import matplotlib.pyplot as plt


arlo = robot.Robot()

#subExercise 1)
def subEx1():
    dist = True
    while dist:
        arlo.go_diff(40,40,1,1)
        a = arlo.read_front_ping_sensor()
        c = arlo.read_right_ping_sensor()
        d = arlo.read_left_ping_sensor()
        sleep(0.2)
        print("a =", a, "c=", c, ", d=",d)
        if a < 400 or c < 400 or d < 400:
            #print(arlo.go_diff(40,40,0,0))
            #sleep(1)
            print(arlo.go_diff(64,64,1,0))
            sleep(0.35)
    arlo.stop()
subEx1()

#subExercise 2)
#def subEx2():
#    llist = []
#    for i in range(40):
#        a = arlo.read_front_ping_sensor()
#        llist.insert(i, a)
#    arlo.stop()
#    print(llist)
#
#
#subEx2()

