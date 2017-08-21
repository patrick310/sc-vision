import cv2

from vdp.helpers import set_resolution

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

classes = ['2_bolt_configuration', '4_bolt_configuration', 'between_cars', 'vehicle_with_no_bolts']
initial_key = 49

if vc.isOpened():  # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

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