import robot 
from time import sleep 

arlo = robot.Robot()

# 1 meter
print(arlo.go_diff(50,50,1,1))
sleep(4)
print(arlo.stop)

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
#print(arlo.go_diff(66,64,1,1))
#sleep(2)

#print(arlo.go_diff(64,0,1,1))
#sleep(3.3)

#print(arlo.go_diff(66,64,1,1))
#sleep(2)

#print(arlo.go_diff(0,64,1,1))
#sleep(3.3)

#print(arlo.go_diff(64,0,1,1))
#sleep(3.3)
#print(arlo.stop)
