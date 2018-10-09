import numpy as np
import cv2
from read_from_image import read_production_number_from_image,number_from_zbar, is_label_in_image



def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def crop_from_points(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


def find_countour_export_pts(raw_image):
    #We start out by putting a filter ont he Image to make the edges more distinct
    bilateral_filtered_image = cv2.bilateralFilter(raw_image, 5, 175, 175)
    cv2.imshow('Bilateral', bilateral_filtered_image)
    cv2.waitKey(0)
    #after that we apply the canny edge detection to find all the edges in the photo
    edge_detected_image = cv2.Canny(bilateral_filtered_image, 75, 200)
    cv2.imshow('Edge', edge_detected_image)
    cv2.waitKey(0)
    #Now that the edges are all white lines and everything else is balck we cam apply a canny edge detector
    _, contours, hierarchy = cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #Establish a list to store all fo the contores and then call on a for loop to go over each contore and make sure it is the one that we need
    #based off the number of sides, aspect ratio, after some time on the line we might add area of pixles also
    contour_list = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True,True)
        area = cv2.contourArea(contour)
        (x,y,w,h)=cv2.boundingRect(contour)
        aspect_ratio = w/float(h)

        if (len(approx) == 4) & (aspect_ratio>3.6) & (aspect_ratio<3.8):
            print('The Approx Len is: ' + str(len(approx)))
            print('The area is: ' + str(area))
            print('Apsect ratio: ' + str(aspect_ratio))
            print('X Coordnate: ' + str(x))
            print('Y Coordnate: ' + str(y))
            print('H Coordnate: ' + str(h))
            print('W Coordnate: ' + str(w))
            contour_list.append(contour)
            #establishing an array to store the foru corners of the contore and then be returend to used in future functions
            pts = np.array(([x,y],[x+w, y],[x+w, y+h],[x, y+h]), dtype="float32")
    #pts = np.array(([1027,448],[1152,456],[1150,482],[1025,457]), dtype="float32")
    #print("This is the array we get from contour" + str(pts))
    # show the original and warped images
    cv2.imshow("Original", raw_image)
    cv2.waitKey(0)

    cv2.drawContours(raw_image, contour_list,  -1, (255,0,0), 2)
    cv2.imshow('Objects Detected',raw_image)
    cv2.waitKey(0)
    return pts




if __name__ == '__main__':

    #read in image
    raw_image = cv2.imread('FTest.jpg')
    #set apart orgin to be croped form later
    orgin =raw_image.copy()
    cv2.imshow('Original Image', raw_image)
    cv2.waitKey(0)
    read_production_number_from_image(raw_image)
    #use finction to find the four points that are the corners of the conour
    pts=find_countour_export_pts(raw_image)
    #take int he points and the original image and crop out the barcode form this image
    warped = crop_from_points(orgin, pts)
    #show me the the barcode
    cv2.imshow("Warped", warped)
    cv2.waitKey(0)