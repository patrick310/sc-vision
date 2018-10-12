try:
    import cv2
except:
    import pip
    pip.main('install', 'cv2')
    import cv2

try:
    import os
except:
    import pip
    pip.main('install', 'os')
    import cv2

try:
    import numpy as np
except:
    import pip
    pip.main('install', 'numpy')
    import numpy as np

try:
    from pyzbar import pyzbar
except:
    import pip
    pip.main('install', 'pyzbar')
    from pyzbar import pyzbar

try:
    import time
except:
    import pip
    pip.main('install', 'time')
    import time

try:
    import os
except:
    import pip
    pip.main('install', 'os')
    import os


def cam_setup_settings(cam):

    """ It puts into place the the settings for the IDS Camera with the 6mm focal length


    Args:
        cam. the cam is put in to be read and altered

    Returns:
        The cam with different settings is returned from this Function
        """
    #Find the original settings for the camera and print them
    width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
    exposure = cam.get(cv2.CAP_PROP_EXPOSURE)
    fps = cam.get(cv2.CAP_PROP_FPS)
    print(width, height, exposure, fps)  # prints 640 480 -4.0
    #Set new parameters for the size of the window
    fpsr = cam.set(cv2.CAP_PROP_FPS, 3)
    hr = cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 2048)
    wr = cam.set(cv2.CAP_PROP_FRAME_WIDTH, 2048)
    #Print out what you set
    print("Setting resolution ", hr, wr, fpsr)  # prints  True True True
    #Show the settins again, remember that exposure time has to be set in camera manager
    width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cam.get(cv2.CAP_PROP_FPS)
    exposure = cam.get(cv2.CAP_PROP_EXPOSURE)
    print(width, height, exposure, fps)  # 2048.0 2048.0 -9.0 3.0000000
    return cam

def auto_canny(image, sigma = 0.33):
    """ It takes care of our canny image detection for us
        Args:
            It takes in an image aka the frame and also a sigma value which can be adjusted to cange the detection

        Returns:
            It returns an edged image or frame to be further processed
            """

    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

def frame_adjustment_canny_application(frame):
    """ This simply adjust the frame of the camera, for some reason the frame was turned upside down so made
    this function to flip it and apply grey scale, gaussianBlur, and the auto canny image.


        Args:
            cam. the cam is put in to be read and altered

        Returns:
            The cam with different settings is returned from this Function
            """
    frame = cv2.flip(frame, 0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = auto_canny(blurred, sigma=0.33)
    cv2.imshow('Original', frame)
    cv2.imshow('Edges', edges)
    return edges

def find_label_scan_and_save_img(frame, label, s, lower, upper):
    """ Takes in an image and contour list and decides if it is the right contour and then tries to read the barcode
    and save the image as the barcode
        Args:
            Contours, contour_list, and big image
        Returns:
            Nothing but it will save the image that you input as the production number
            """
    cv2.imshow("label", label)
    frame = frame
    origin = label.copy()
    save = frame.copy()
    # print('copying images to other names for editing and future saving...')
    pts = find_countour_export_pts(label, s, lower, upper)
    # print('Finding pts using the contors of image...')
    # print(pts)
    localtime = time.localtime(time.time())
    print(localtime)
    if np.all(pts):
        cropped_image = crop_from_points(origin, pts)
        cv2.imwrite("label.jpg", cropped_image)
        cv2.imshow('Crop', cropped_image)
        #Form here make coppies and resize the image to be easir to read by the program and from there        #put it into the read_from_zbar function that will return the barcode which is saved and used to name
        #the image that is saved and was coppied form earlier
        raw_image = cv2.imread('label.jpg')
        blur_score = blur_detection(raw_image)
        print("The Blur Score = " + str(blur_score))
        big_raw_image = cv2.resize(raw_image, (913, 249))
        barcode_number = number_from_zbar(big_raw_image)
        barcode_number = str(barcode_number)
        cv2.imshow("biggon", big_raw_image)
        path = "Users\MBUSI\Desktop\\ratiodetect\share\\" + str(barcode_number)

        #cv2.imwrite(path + "\\" + str(x) + ',' + str(y) + ',' + str(h) + ',' + str(w) + ',' + str(aspect_ratio) + ',' + str(area), raw_image)
        if (len(barcode_number) == 8):
            if not os.path.isfile(path + "\\" + "frame" + ".jpg"):
                cv2.imwrite(path + "\\" + "frame" + ".jpg", frame)
            if not os.path.isfile(path + "\\" + "label" + ".jpg"):
                cv2.imwrite(path + "\\" + "label" + ".jpg", label)
        elif not os.path.isfile(str(blur_score) + ".jpg"):
            cv2.imwrite(str(blur_score) + ".jpg", label)

def scale_image(input, factor):
        w, h = cv2.GetSize(input)
        # print('w=' + str(w))
        # print('h=' + str(w))
        output = cv2.resize(input, ((int(h)*int(factor)), (int(w)*int(factor))), interpolation=cv2.INTER_CUBIC )
        return output

def order_points(pts):
    """ Makes sure that your points are in order when theya are put into the cropping and adjusting function


            Args:
                Points

            Returns:
                those same points only now in order
                """

    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # print(type(pts))
    # print(pts)
    # pts = pts
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

    """ Takes in an image and points and retruns an upright corrected image that is no longer shifted in one direction
    so it is easier to read


            Args:
                image, pts

            Returns:
                the barcode label image
                """

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

def find_countour_export_pts(raw_image, s, lower, upper):
    #We start out by putting a filter ont he Image to make the edges more distinct
    gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2HSV)
    # print('The Image is grayscaled')
    blurred = cv2.GaussianBlur(gray, (1, 1), 0)
    # print('The Image is blurred')
    smooth = cv2.addWeighted(blurred, 1.5, raw_image, -0.5, 0)
    # print('The image is smooth')
    #after that we apply the canny edge detection to find all the edges in the photo
    edge_detected_image = cv2.Canny(smooth, lower, upper)
    # print('autocanny detection complete')
    cv2.imshow('Edge', edge_detected_image)
    #cv2.waitKey(0)
    #Now that the edges are all white lines and everything else is balck we cam apply a canny edge detector
    _, contours, hierarchy = cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]
    # print('Contours found')
    # print(contours)
    #Establish a list to store all fo the contores and then call on a for loop to go over each contore and make sure it is the one that we need
    #based off the number of sides, aspect ratio, after some time on the line we might add area of pixles also
    contour_list = []
    global aspect_ratio
    global x, y, w, h
    global area
    global approx
    for contour in contours:
        approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True,True)
        area = cv2.contourArea(contour)
        (x, y, w, h)=cv2.boundingRect(contour)
        aspect_ratio = w/float(h)


        if (len(approx) > 3) & (len(approx) < 10) & (area > 2000) & (aspect_ratio > 3) & (aspect_ratio < 6):
            # print('The Approx Len is: ' + str(len(approx)))
            # print('The area is: ' + str(area))
            # print('Apsect ratio: ' + str(aspect_ratio))
            # print('X Coordnate: ' + str(x))
            # print('Y Coordnate: ' + str(y))
            # print('H Coordnate: ' + str(h))
            # print('W Coordnate: ' + str(w))
            contour_list.append(contour)
            #establishing an array to store the foru corners of the contore and then be returend to used in future functions
            #pts = np.array(([x,y],[x+w, y],[x+w, y+h],[x, y+h]), dtype="float32")
            rect = cv2.minAreaRect(contour)
            # print(str(rect))
            box = cv2.boxPoints(rect)
            pts = np.int0(box)
            #cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
        #pts = np.array(([1027,448],[1152,456],[1150,482],[1025,457]), dtype="float32")
        #print("This is the array we get from contour" + str(pts))
        # show the original and warped images
            cv2.drawContours(raw_image, contour_list,  -1, (255, 0, 0), 2)
            cv2.imshow('Objects Detected', raw_image)
            # print(str(pts) + 'find_contour_return_pts')
            return pts

def number_from_zbar(image):
    """Uses Zbar to identify, read, and return the first value from a barcode found in an image"""
    barcode_data = None
    production_number = None
    barcodes = pyzbar.decode(image)
    print("Barcodes collected")
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
            len_pro_num = len(str(production_number))
            print(str(len_pro_num))
            return str(production_number)

        print("[INFO] Found {} barcode: {}".format(barcodeType, barcode_data))

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

    if (len(str(number_from_zbar(image)))==8):
        print("main loop, barcode detected")

        if type(num_zbar) is None or type(num_zbar) is False:
            print("major error! How did you get here with a None?")
        print("main loop " + num_zbar + " was the number after we pull from zbar")
        cv2.imwrite(str(num_zbar) + ".jpg", img)
        return str(number_from_zbar(img))

    #if locate_label_in_image(image) is not None:
    #    label_cropped = four_point_transform(img, locate_label_in_image(img))
    #   return True

    else:
        print("read_production_number_from_image is returning false")
        return False

    # cropped_labels = [four_point_transform(coordinate) for coordinate in detected_labels]

def blur_detection(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()