import cv2
import cv2.aruco as aruco

VideoCap = False
cap=cv2.VideoCapture(0)

def findAruco(img, marker_size=6, total_markers=250,draw=True):
    gray=cv2.cvtColor(img,cv2.COLOR_BGRA2GRAY)
    key = getattr(aruco,f"DICT_{marker_size}X{marker_size}_{total_markers}")
    arucoDict=aruco.Dictionary_get(key)

while True:
    if VideoCap: _, img=cap.read()
    else:
        img=cv2.imread("test.png")
        img=cv2.resize(img,(0,0),fx=0.7,fy=0.7)
    if cv2.waitKey(1)==113:
        break
    cv2.imshow("img",img)
