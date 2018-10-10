import numpy as np
import cv2
from barcode_detection_functions import number_from_zbar
from pyzbar import pyzbar


if __name__ == '__main__':

    #read in image
    raw_image = cv2.imread('BarcodeTest.jpg')
    # gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2HSV)
    # blurred = cv2.GaussianBlur(raw_image, (1, 1), 5)
    # print('The Image is blurred')
    # smooth = cv2.addWeighted(blurred, 2.5, raw_image, -0.5, 0)
    _, thresh = cv2.threshold(raw_image, 0, 30, cv2.THRESH_TOZERO)
    cv2.imshow('Original Image', raw_image)
    cv2.waitKey(0)
    cv2.imshow('Original Image', thresh)
    cv2.waitKey(0)
    print(pyzbar.decode(thresh))
    barcode = pyzbar.decode(thresh)
    print(str(barcode))