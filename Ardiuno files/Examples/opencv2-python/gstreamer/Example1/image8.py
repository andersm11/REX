import cv2 as cv2
from cv2 import aruco 

marker_dict=aruco.Dictionary_get(aruco.DICT_6X6_250)

MARKER_SIZE = 400

for id in range(20):
    marker_image = aruco.drawMarker(marker_dict, id, MARKER_SIZE)
    cv.imwrite(f"markers/markers_{id}.png", marker_image)