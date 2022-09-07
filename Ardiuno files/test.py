import robot 
from time import sleep 

arlo = robot.Robot()

print(arlo.go_diff(66,64,1,1))
sleep(2)

print(arlo.go_diff(64,64,1,0))
sleep(0.7)

print(arlo.go_diff(66,64,1,1))
sleep(2)

print(arlo.go_diff(64,64,1,0))
sleep(0.7)

print(arlo.go_diff(66,64,1,1))
sleep(2)

print(arlo.go_diff(64,64,1,0))
sleep(0.7)

print(arlo.go_diff(66,64,1,1))
sleep(2)

print(arlo.go_diff(64,64,1,0))
sleep(0.7)
print(arlo.stop)

print("finish")
