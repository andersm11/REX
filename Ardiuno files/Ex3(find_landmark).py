# This script shows how to open a camera in OpenCV and grab frames and show these.
# Kim S. Pedersen, 2022
import robot 
from pickle import FALSE, TRUE
from time import sleep, time  
#from types import None
import cv2 # Import the OpenCV library
import cv2.aruco
import ctypes
import io
from contextlib import contextmanager
import os
import sys
import tempfile
import numpy as np


libc = ctypes.CDLL(None)
c_stderr = ctypes.c_void_p.in_dll(libc, 'stderr')

@contextmanager
def stderr_redirector(stream):
    original_stderr_fd = sys.stderr.fileno()

    def _redirect_stderr(to_fd):
        libc.fflush(c_stderr)
        sys.stderr.close()
        os.dup2(to_fd, original_stderr_fd)
        sys.stderr = io.TextIOWrapper(os.fdopen(original_stderr_fd, 'wb'))

    saved_stderr_fd = os.dup(original_stderr_fd)
    try:
        tfile = tempfile.TemporaryFile(mode='w+b')
        _redirect_stderr(tfile.fileno())
        yield
        _redirect_stderr(saved_stderr_fd)
        tfile.flush()
        tfile.seek(0, io.SEEK_SET)
        stream.write(tfile.read().decode())
    finally:
        tfile.close()
        os.close(saved_stderr_fd)

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

# Open a camera device for capturing gstreamer_pipeline(), apiPreference=cv2.CAP_GSTREAMER

#cam = cv2.VideoCapture(0)
arlo = robot.Robot()

cam_matrix = np.array([[1672,0,540],[0,1672,400],[0,0,1]])





arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
arucoParams = cv2.aruco.DetectorParameters_create()
dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

z = np.array([0,0,1],dtype=float)
x = np.array([1,0,0],dtype=float)
box_id = 8
def CheckID(id):
    if id is not None:
        if id[0] == box_id:
            return True
        else:
            return False
    return False
def Turn(angle):
    if angle <= 0:
        arlo.go_diff(30,30,0,1)
        print(0.019*abs(angle))
        sleep(0.019*abs(angle))
        arlo.stop()
    else:
        arlo.go_diff(30,30,1,0)
        print(0.019*abs(angle))
        sleep(0.019*abs(angle))
        arlo.stop()


def Take_pic():
    with stderr_redirector(io.StringIO()):
        cam = cv2.VideoCapture(gstreamer_pipeline(), apiPreference=cv2.CAP_GSTREAMER)
        #cam = cv2.VideoCapture(0)
    if not cam.isOpened(): # Error
        print("Could not open camera")
        exit(-1)
    retval, frameReference = cam.read() 
    cam.release()
    return retval, frameReference
arlo.stop()
while cv2.waitKey(4) == -1: # Wait for a key pressed event
    # Read frame
    retval, frameReference = Take_pic()
    if not retval: # Error
        print(" < < <  Game over!  > > > ")
        exit(-1)
    (corners, ids, rejected) = cv2.aruco.detectMarkers(frameReference, arucoDict,
        parameters=arucoParams)
    cv2.aruco.drawDetectedMarkers(frameReference,corners)
    print(ids)
    rvec, tvec, objPoints = cv2.aruco.estimatePoseSingleMarkers(corners,15,cam_matrix,0)
    if tvec is not None:
        #tvec2 = np.reshape(tvec[0,:],(3,))
        tvec = tvec[0].reshape((3,))
        tvec_norm = tvec/np.linalg.norm(tvec)
        print(tvec.shape)
        beta = np.rad2deg(np.arccos(np.dot(tvec_norm,z)))
        print("dot:",np.dot(tvec_norm, z) )
        sign = np.sign(np.dot(np.transpose(x),tvec))
        angle = beta*sign
        print("beta", angle, "sign:", sign)

    if CheckID(ids) is True:
        if 3 < beta:
            Turn(angle)
        else:
            length = np.linalg.norm(tvec)
            if length < 30:
                exit(1)
            arlo.go_diff(50,50,1,1)
            print(length)
            sleep(0.028*(length))
            arlo.stop()
            exit(1)
    else:
        arlo.go_diff(30,30,1,0)
        sleep(0.5)
        arlo.stop()

 
 
 #Finished successfully



