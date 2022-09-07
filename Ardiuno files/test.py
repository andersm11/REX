import robot 
from time import sleep 

arlo = robot.Robot()

print(arlo.go_diff(64,64,0,0))
sleep(2)
print(arlo.stop)
print("finish")
