import numpy as np
import cv2
from barcode_detection_functions import find_label_scan_and_save_img


if __name__ == '__main__':

    #read in image
    raw_image = cv2.imread('test4.5.jpg')
    cv2.imshow('Original Image', raw_image)
    cv2.waitKey(0)
    find_label_scan_and_save_img(raw_image)
    cv2.waitKey(0)