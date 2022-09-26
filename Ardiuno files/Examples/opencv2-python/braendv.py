
import cv2 # Import the OpenCV library
import cv2.aruco

import numpy as np

print("OpenCV version = " + cv2.__version__)

def gstreamer_pipeline(capture_width=1024, capture_height=720, framerate=30):
    """Utility function for setting parameters for the gstreamer camera pipeline"""
    return (
        "libcamerasrc !"
        "video/x-raw, width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "videoconvert ! "
        "appsink"
        % (
            capture_width,
            capture_height,
            framerate,
        )
    )

# Open a camera device for capturing
cam = cv2.VideoCapture(gstreamer_pipeline(), apiPreference=cv2.CAP_GSTREAMER)

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




#while cv2.waitKey(4) == -1: # Wait for a key pressed event
retval, frameReference = cam.read() # Read frame
    
if not retval: # Error
    print(" < < <  Game over!  > > > ")
    exit(-1)
(corners, ids, rejected) = cv2.aruco.detectMarkers(frameReference, arucoDict,
    parameters=arucoParams)

cv2.aruco.drawDetectedMarkers(frameReference,corners)
print(ids)
cv2.imshow("billede",frameReference)
cv2.imwrite("test1.png", frameReference)
