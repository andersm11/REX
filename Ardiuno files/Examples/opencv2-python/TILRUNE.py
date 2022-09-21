# This script shows how to open a camera in OpenCV and grab frames and show these.
# Kim S. Pedersen, 2022

import cv2 # Import the OpenCV library
import cv2.aruco

import numpy as np

print("OpenCV version = " + cv2.__version__)

# Open a camera device for capturing
cam = cv2.VideoCapture(0)

if not cam.isOpened(): # Error
    print("Could not open camera")
    exit(-1)

# Open a window
WIN_RF = "Example 1"
cv2.namedWindow(WIN_RF)
cv2.moveWindow(WIN_RF, 100, 100)

arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
arucoParams = cv2.aruco.DetectorParameters_create()
dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)




while cv2.waitKey(4) == -1: # Wait for a key pressed event
    retval, frameReference = cam.read() # Read frame
    
    if not retval: # Error
        print(" < < <  Game over!  > > > ")
        exit(-1)
    (corners, ids, rejected) = cv2.aruco.detectMarkers(frameReference, arucoDict,
        parameters=arucoParams)

    cv2.aruco.drawDetectedMarkers(frameReference,corners)
    print(ids)
    cv2.imshow("billede",frameReference)

# Finished successfully



