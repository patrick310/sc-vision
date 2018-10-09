import cv2
from pyzbar import pyzbar
import numpy as np
from img_edge_detection import order_points, crop_from_points, find_countour_export_pts
from read_from_image import read_production_number_from_image,number_from_zbar, is_label_in_image



if __name__ == '__main__':
    cap = cv2.VideoCapture(1)
    #Captures video using the While Loop
    while (1):
        _, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red = np.array([30, 150, 50])
        upper_red = np.array([255, 255, 180])

        mask = cv2.inRange(hsv, lower_red, upper_red)
        res = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.imshow('Original', frame)
        edges = cv2.Canny(frame, 10,200)
        cv2.imshow('Edges', edges)

        _, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contour_list = []

        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True, True)
            area = cv2.contourArea(contour)
            (x, y, w, h) = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            if (len(approx) == 4) & (aspect_ratio > 3.65) & (aspect_ratio < 3.75):
                print('The Approx Len is: ' + str(len(approx)))
                print('Apsect ratio: ' + str(aspect_ratio))
                print('X Coordnate: ' + str(x))
                print('Y Coordnate: ' + str(y))
                contour_list.append(contour)
                pts = np.array(([x,y],[x+w, y],[x+w, y+h],[x, y+h]), dtype="float32")
                print(str(pts))
                orgin = frame.copy()
                croped_image = crop_from_points(orgin,pts)
                cv2.imshow('Crop', croped_image)
                production_number = read_production_number_from_image(croped_image)

                # print the barcode type and data to the terminal
                if is_label_in_image(croped_image):
                    print("[INFO] Found {} barcode: {}".format("Production number", read_production_number_from_image(croped_image)))
            cv2.drawContours(frame, contour_list, -1, (255, 0, 0), 2)
            cv2.imshow('Box', frame)



        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()
    cap.release()