import robot 
from time import sleep 

arlo = robot.Robot()

print(arlo.go_diff(64,64,1,0))
sleep(0.6)
print(arlo.stop)
print("finish")
