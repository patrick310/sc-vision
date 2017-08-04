import cv2
from PIL import Image


def set_resolution(cap, x, y):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(x))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(y))
    return str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


cv2.namedWindow("preview")
vc = cv2.VideoCapture(1)

if vc.isOpened():  # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

counter = 0

set_resolution(vc, 1920, 1080)

while rval:
    file_name = str(counter) + ".jpg"
    rval, frame = vc.read()
    oframe = frame.copy()
    cv2.imshow("preview", frame)

    key = cv2.waitKey(20)
    if key == 97:
        cv2.imwrite("newData/pos/" + file_name, oframe)
        counter = counter + 1
    if key == 108:
        cv2.imwrite("newData/neg/" + file_name, oframe)
        counter = counter + 1
    if key == 27:  # exit on ESC
        break
cv2.destroyWindow("preview")