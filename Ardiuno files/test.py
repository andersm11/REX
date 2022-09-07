import robot 
from time import sleep 

arlo = robot.Robot()

# 1 meter
#print(arlo.go_diff(49,49,1,1))
#sleep(3)
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

#print(arlo.go_diff(66,31,1,1))
#sleep(10)

#print(arlo.go_diff(31,66,1,1))
#sleep(10)

print(arlo.go_diff(31,66,1,1))
sleep(10)
print(arlo.stop)
