import numpy as np
import cv2


cap = cv2.VideoCapture('../videos/walk.mp4')

ret,front_frame = cap.read()
front_frame = cv2.cvtColor(front_frame, cv2.COLOR_BGR2GRAY)
front_frame = cv2.medianBlur(front_frame,5)
while(cap.isOpened()):

    ret, back_frame = cap.read()
    # 读到结尾就退出
    if (ret == False):
        break;
    back_frame = cv2.cvtColor(back_frame, cv2.COLOR_BGR2GRAY)
    back_frame = cv2.medianBlur(back_frame, 5)
    dir = cv2.absdiff(back_frame,front_frame)

    dir[dir>10]=255
    dir[dir<10]=0
    #res = binary(back_frame,dir)

    cv2.imshow('frame',dir)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    front_frame = back_frame
cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()