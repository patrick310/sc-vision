import cv2
import PIL.Image, PIL.ImageTk
import pyzbar.pyzbar as pyzbar
import numpy as np
import sys
import logging
import time
# from testreal import Deployment, VideoCapture
import keyboard
from vision_sys_gui import *
from vision_gui import *


def barcode_filter(image, value=True):
    image = np.asarray(image)
    def decode(im):
        # Find barcodes and QR codes
        decodedObjects = pyzbar.decode(im)
        # Print results
        for obj in decodedObjects:
            print('Type : ', obj.type)
            print('Data : ', obj.data, '\n')
        return decodedObjects
    # Display barcode and QR code location
    def display(im, decodedObjects):
        # Loop over all decoded objects
        for decodedObject in decodedObjects:
            points = decodedObject.polygon
            # If the points do not form a quad, find convex hull
            if len(points) > 4:
                hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
                hull = list(map(tuple, np.squeeze(hull)))
            else:
                hull = points;
            # Number of points in the convex hull
            n = len(hull)
            # Draw the convex hull
            for j in range(0, n):
                cv2.line(im, hull[j], hull[(j + 1) % n], (255, 0, 0), 3)
    decoded_objects = decode(image)
    if len(decoded_objects) is not 0:
        logger.info(decoded_objects)
        # App.save(decoded_objects, image)
    return [image, value]
def grayscale_filter(image, value=None):
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return [image, value]
def flip_filter(image, value=None):
    image = np.asarray(image)
    image = cv2.flip(image, -1)
    return [image, value]
def canny_filter(image, value=None):
    image = np.asarray(image)
    sigma = 0.33
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    image = cv2.Canny(image, lower, upper)
    return [image, value]
def space_true(image, value=None): 
    image = np.asarray(image)
    if keyboard.is_pressed('a'): #if key 'a' is pressed
            value = "IMG.jpg"
    return[image, value]
#f_end_string

filter_dict = {"barcode_filter" : barcode_filter,
                "flip_filter" : flip_filter,
                "grayscale_filter" : grayscale_filter,
				"canny_filter" : canny_filter,
				"space_true" : space_true,
#d_end_string
            }