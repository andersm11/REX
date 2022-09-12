import robot 
from time import sleep 

#venstre hjul m/s pr tik: 0.0066 
#højre hjul m/s pr tik: 0.0069 

arlo = robot.Robot()
print(arlo.go_diff(40,40,1,1))
#a = arlo.read_front_ping_sensor()
#while a > 300:
#    a = arlo.read_front_ping_sensor()
#    sleep(0.2)
#    print(a)
#arlo.stop()
#
dist = True
while dist:
    a = arlo.read_front_ping_sensor()
    b = arlo.read_back_ping_sensor()
    c = arlo.read_right_ping_sensor()
    d = arlo.read_left_ping_sensor()
    sleep(0.2)
    print("a =", a, ", b=", b, ", c=", c, ", d=",d)
    if a < 300 or b < 300 or c < 300 or d < 300:
        print(arlo.go(40,40,0,0))
        sleep(0.5)
        print(arlo.go_diff(64,64,1,0))
        sleep(0.7)


arlo.stop()


# 1 meter
#print(arlo.go_diff(50,48.5,1,1))
#sleep(3)
#print(arlo.go_diff(0,30,1,1))
#sleep(0.1)
#print(arlo.stop)
#for i in range(12):
#    print(arlo.go_diff(68,64,1,1))
#    sleep(2)
#
#    print(arlo.go_diff(63.5,63.5,1,0))
#    sleep(0.7)

# Square
# print(arlo.go_diff(66,64,1,1))
# sleep(2)

# print(arlo.go_diff(64,64,1,0))
# sleep(0.7)

# print(arlo.go_diff(66,64,1,1))
# sleep(2)

# print(arlo.go_diff(64,64,1,0))
# sleep(0.7)

# print(arlo.go_diff(66,64,1,1))
# sleep(2)

# print(arlo.go_diff(64,64,1,0))
# sleep(0.7)

# print(arlo.go_diff(66,64,1,1))
# sleep(2)

# print(arlo.go_diff(64,64,1,0))
# sleep(0.7)
# print(arlo.stop)

# print("finish")

# Continuous motion (8)

# Med uret
#print(arlo.go_diff(84,47,1,1))
#sleep(10)
#print(arlo.go_diff(0,30,1,1))
#sleep(0.1)

# Mod uret
#print(arlo.go_diff(52,80,1,1))
#sleep(10)
#print(arlo.go_diff(0,30,1,1))
#sleep(0.1)

#print(arlo.go_diff(85,47,1,1))
#sleep(10)
#print(arlo.go_diff(0,30,1,1))
#sleep(0.1)

# Mod uret
#print(arlo.go_diff(51.7,79,1,1))
#sleep(10)
#print(arlo.go_diff(0,30,1,1))
#sleep(0.1)



#print(arlo.go_diff(31,62,1,1))
#sleep(10)
print(arlo.stop)
