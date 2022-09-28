# This script shows how to open a camera in OpenCV and grab frames and show these.
# Kim S. Pedersen, 2022

from time import sleep
#from types import NoneType
import cv2 # Import the OpenCV library
import cv2.aruco
import numpy as np

print("OpenCV version = " + cv2.__version__)

def gstreamer_pipeline(capture_width=1024, capture_height=720, framerate=30):
    """Utility function for setting parameters for the gstreamer camera pipeline"""
    return (
        "libcamerasrc !"
        "videobox autocrop=true"
        "video/x-raw, width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "videoconvert ! "
        "appsink"
        % (
            capture_width,
            capture_height,
            framerate,
        )
    )

# Open a camera device for capturing gstreamer_pipeline(), apiPreference=cv2.CAP_GSTREAMER
#cam = cv2.VideoCapture(gstreamer_pipeline(), apiPreference=cv2.CAP_GSTREAMER)
cam = cv2.VideoCapture(0)

cam_matrix = np.array([[1.628,0,680],[0,1.628,360],[0,0,1]])

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

z = np.array([0,0,1])
x = np.array([1,0,0])



while cv2.waitKey(4) == -1: # Wait for a key pressed event
    sleep(1)
    retval, frameReference = cam.read() # Read frame
    
    if not retval: # Error
        print(" < < <  Game over!  > > > ")
        exit(-1)
    (corners, ids, rejected) = cv2.aruco.detectMarkers(frameReference, arucoDict,
        parameters=arucoParams)

    cv2.aruco.drawDetectedMarkers(frameReference,corners)
    print(ids)
    rvec, tvec, objPoints = cv2.aruco.estimatePoseSingleMarkers(corners,15,cam_matrix,0)
    if tvec is not None:
        tvec = np.reshape(tvec,(3,))
        tvec_norm = tvec/np.linalg.norm(tvec)
        beta = (np.rad2deg(np.arccos(np.dot(tvec_norm,z)))-89)*100
        print("dot:",np.dot(tvec_norm, z) )
        sign = np.sign(np.dot(np.transpose(x),tvec))
        print("beta", beta, "sign:", sign, "k:", tvec_norm, "\n")
        
    cv2.imshow("billede",frameReference)

# Finished successfully



