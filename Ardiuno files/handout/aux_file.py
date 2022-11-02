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


def gstreamer_pipeline(capture_width=1280, capture_height=720, framerate=30):
    """Utility function for setting parameters for the gstreamer camera pipeline"""
    return (
        "libcamerasrc !"
        "videobox autocrop=true !"
        "video/x-raw, width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "videoconvert ! "
        "appsink"
        % (
            capture_width,
            capture_height,
            framerate,
        )
    )

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