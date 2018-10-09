import cv2
from pyzbar import pyzbar
import numpy as np


def read_production_number_from_image(image):
    """ Returns the production number of a vehicle from an image including a barcode.


    Args:
        image: The input image, to be searched for the barcode, Must be in jpg format.

    Returns:
        production_number: The production number. May be None if the process was not
        successfully launched.
        For Example:

        4252811

        If the function retuns None, then there was no barcode found in the image
        """
    if isinstance(image, str):
        image = cv2.imread(image)
    img = image
    num_zbar = number_from_zbar(img)
    print("zbar initial number " + str(num_zbar))

    if number_from_zbar(img):
        print("main loop, barcode detected")

        if type(num_zbar) is None or type(num_zbar) is False:
            print("major error! How did you get here with a None?")
        print("main loop " + num_zbar + " was the number after we pull from zbar")
        cv2.imwrite(str(num_zbar) + ".jpg", img)
        return number_from_zbar(img)

    #if locate_label_in_image(image) is not None:
    #    label_cropped = four_point_transform(img, locate_label_in_image(img))
    #   return True

    else:
        return None

    # cropped_labels = [four_point_transform(coordinate) for coordinate in detected_labels]


def number_from_zbar(image):
    """Uses Zbar to identify, read, and return the first value from a barcode found in an image"""

    barcode_data = None
    production_number = False
    barcodes = pyzbar.decode(image)

    # loop over the detected barcodes
    for barcode in barcodes:
        # extract the bounding box location of the barcode and draw the
        # bounding box surrounding the barcode on the image
        (x, y, w, h) = barcode.rect
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # the barcode data is a bytes object so if we want to draw it on
        # our output image we need to convert it to a string first
        barcode_data = barcode.data.decode("utf-8")
        # TODO make function only look for code128 barcodes

        barcodeType = barcode.type

        # draw the barcode data and barcode type on the image
        text = "{} ({})".format(barcode_data, barcodeType)
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 2)

        # print the barcode type and data to the terminal

        if len(barcode_data.split()[0]) == 8:
            print("We found a barcode ")
            production_number = barcode_data.split()[0]
            print(production_number)
            return str(production_number)

        print("[INFO] Found {} barcode: {}".format(barcodeType, barcode_data))


def order_points(pts):
    """
    Initialize a list of coordinates that will be ordered
    such that the first entry in the list is the top-left,
    the second entry is the top-right, the third is the
    bottom-right, and the fourth is the bottom-left
    """
    if type(pts) == type("a string"):
        pts = pts.split()
    pts = np.asarray(pts).reshape((4,2))
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=0)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    """Crops an image from four points"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def locate_label_in_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)[1]

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[1]

    # loop over the contours
    reduced_list = [c for c in cnts if is_label(c)]

    if len(reduced_list) == 0:
        return None
    else:
        for c in reduced_list:
            # compute the center of the contour, then detect the name of the
            # shape using only the contour

            shape = shape_detector(c)

            if shape == "label":
                return points_of_label(c)
            else:
                return None


def is_label_in_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)[1]

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[1]

    # loop over the contours
    reduced_list = [c for c in cnts if is_label(c)]

    if len(reduced_list) < 1:
        return False
    else:
        for c in reduced_list:
            # compute the center of the contour, then detect the name of the
            # shape using only the contour

            shape = shape_detector(c)

            if shape == "label":
                return True
            else:
                return False


def shape_detector(c):
    """Approximates the contour shape"""

    peri = cv2.arcLength(c, True)

    if peri > 600:
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        # if the shape is a triangle, it will have 3 vertices
        if len(approx) == 3:
            shape = "triangle"

        elif len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)

            shape = "square" if 0.95 <= ar <= 1.05 else "rectangle"

            if 3.7 <= ar <= 4.0:
                shape = "label"

        elif len(approx) == 5:
            shape = "pentagon"

        else:
            shape = None

        return shape
    else:
        return None


def points_of_label(label_contour):
    c = label_contour
    """Estimates the 4 points of the label"""
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    (x, y, w, h) = cv2.boundingRect(approx)
    return [[x,y], [x+w,y], [x,y+h], [x+w,y+h]]


def is_label(c):
    if shape_detector(c) == 'label':
        return True
    else:
        return False




