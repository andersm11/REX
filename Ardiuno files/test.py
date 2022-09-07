import robot 
from time import sleep 

#venstre hjul m/s pr tik: 0.0066 
#højre hjul m/s pr tik: 0.0069 

arlo = robot.Robot()

# 1 meter
#print(arlo.go_diff(50,48.5,1,1))
#sleep(3)
#print(arlo.go_diff(0,30,1,1))
#sleep(0.1)
#print(arlo.stop)

# Square
#print(arlo.go_diff(66,64,1,1))
#sleep(2)

#print(arlo.go_diff(64,64,1,0))
#sleep(0.7)

#print(arlo.go_diff(66,64,1,1))
#sleep(2)

#print(arlo.go_diff(64,64,1,0))
#sleep(0.7)

#print(arlo.go_diff(66,64,1,1))
#sleep(2)

#print(arlo.go_diff(64,64,1,0))
#sleep(0.7)

#print(arlo.go_diff(66,64,1,1))
#sleep(2)

#print(arlo.go_diff(64,64,1,0))
#sleep(0.7)
#print(arlo.stop)

#print("finish")

# Continuous motion (8)
#

print(arlo.go_diff(83,45.5,1,1))
sleep(5)

#print(arlo.go_diff(31,62,1,1))
#sleep(10)
print(arlo.stop)
