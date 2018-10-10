import cv2
from barcode_detection_functions import cam_setup_settings, frame_adjustment_canny_application, \
    find_label_scan_and_save_img
import time
import numpy as np


if __name__ == '__main__':
    cam = cv2.VideoCapture(0)
    cam2 = cv2.VideoCapture(1)
    cam2.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
    exposure = cam.get(cv2.CAP_PROP_EXPOSURE)
    fps = cam.get(cv2.CAP_PROP_FPS)
    width2 = cam2.get(cv2.CAP_PROP_FRAME_WIDTH)
    height2 = cam2.get(cv2.CAP_PROP_FRAME_HEIGHT)
    exposure2 = cam2.get(cv2.CAP_PROP_EXPOSURE)
    fps2 = cam2.get(cv2.CAP_PROP_FPS)

    print(width, height, exposure, fps)  # prints 640 480 -4.0
    print(width2, height2, exposure2, fps2)  # prints 640 480 -4.0

    # Set new parameters for the size of the window
    fpsr = cam.set(cv2.CAP_PROP_FPS, 3)
    hr = cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 2048)
    wr = cam.set(cv2.CAP_PROP_FRAME_WIDTH, 2048)
    # Print out what you set
    print("Setting resolution ", hr, wr, fpsr)  # prints  True True True
    # Show the settins again, remember that exposure time has to be set in camera manager
    exposure3 = cam2.set(cv2.CAP_PROP_EXPOSURE, -9)
    width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cam.get(cv2.CAP_PROP_FPS)
    exposure = cam.get(cv2.CAP_PROP_EXPOSURE)
    exposure2 = cam2.get(cv2.CAP_PROP_EXPOSURE)
    print(width, height, exposure, fps, exposure2)  # 2048.0 2048.0 -9.0 3.000000
    # create trackbar for canny edge detection threshold changes
    cv2.namedWindow('canny')
    def nothing():
        pass
    # add ON/OFF switch to "canny"
    switch = '0 : OFF \n1 : ON'
    cv2.createTrackbar(switch, 'canny', 0, 1, nothing)

    # add lower and upper threshold slidebars to "canny"
    cv2.createTrackbar('lower', 'canny', 0, 255, nothing)
    cv2.createTrackbar('upper', 'canny', 0, 255, nothing)
    while 1:
        _, frame = cam.read()
        _, label = cam2.read()
        lower = cv2.getTrackbarPos('lower', 'canny')
        upper = cv2.getTrackbarPos('upper', 'canny')
        s = cv2.getTrackbarPos(switch, 'canny')
        frame = cv2.flip(frame, 0)
        print('Reading Frame')
        cv2.imshow("Frame", frame)
        p = find_label_scan_and_save_img(frame, label, s, lower, upper)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()
