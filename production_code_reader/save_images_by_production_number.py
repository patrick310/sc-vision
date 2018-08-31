import numpy as np
import cv2
from pyzbar.pyzbar import decode

from production_code_reader.read_from_image import read_production_number_from_image, is_label_in_image

cap = cv2.VideoCapture(1)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    image = frame

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    retval, threshold = cv2.threshold(gray, 50, 150, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    production_number = read_production_number_from_image(frame)

    # print the barcode type and data to the terminal
    if is_label_in_image(frame):
        print("[INFO] Found {} barcode: {}".format("Production number", read_production_number_from_image(frame)))

    # Display the resulting frame
    cv2.imshow('frame',image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()