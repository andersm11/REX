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
        b = arlo.read_back_ping_sensor()
        c = arlo.read_right_ping_sensor()
        d = arlo.read_left_ping_sensor()
        sleep(0.2)
        print("a =", a, ", b=", b, ", c=", c, ", d=",d)
        if a < 300 or b < 300 or c < 300 or d < 300:
            print(arlo.go_diff(40,40,0,0))
            sleep(1)
            print(arlo.go_diff(64,64,1,0))
            sleep(0.7)
    arlo.stop()

#subExercise 2)
def subEx2():
    llist = []
    for i in range(40):
        a = arlo.read_front_ping_sensor()
        llist.insert(i, a)
    arlo.stop()
    print(llist)
    plt.plot(llist)
    plt.show()

subEx2() 