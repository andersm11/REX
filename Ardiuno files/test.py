import robot 
from time import sleep 

arlo = robot.Robot()

print(arlo.go_diff(68,64,1,1))
sleep(2)
print(arlo.stop)
print("finish")
